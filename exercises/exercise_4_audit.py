"""Durable HITL graph with structured audit logging."""
import argparse
import asyncio
import os
import time
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command, interrupt
from common.db import db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.schemas import AuditEntry, ReviewState, risk_level_for
from exercises.exercise_1_confidence import node_route as route
from exercises.review import (review_llm, analysis_messages, synthesis_messages,
    approval_payload, validate_response, validate_answers, render_comment)

AGENT_ID = 'pr-review-agent@v0.1'

async def audit(state, entry):
    await write_audit_event(thread_id=state['thread_id'], pr_url=state['pr_url'], entry=entry)

async def log(state, action, decision, start, reason=None, analysis=None, human=False):
    a = analysis or state.get('analysis')
    c = a.confidence if a else 0.0
    await audit(state, AuditEntry(agent_id=AGENT_ID, action=action, confidence=c,
        risk_level=risk_level_for(c) if a else 'med', decision=decision, reason=reason,
        reviewer_id=os.getenv('GITHUB_USER') if human else None,
        execution_time_ms=int((time.monotonic()-start)*1000)))

async def node_fetch_pr(state):
    start = time.monotonic()
    pr = await asyncio.to_thread(fetch_pr, state['pr_url'])
    await log(state, 'fetch_pr', 'pending', start, f'Fetched {len(pr.files_changed)} files; head={pr.head_sha}')
    return dict(pr_title=pr.title, pr_author=pr.author, pr_diff=pr.diff,
                pr_files=pr.files_changed, pr_head_sha=pr.head_sha)

async def node_analyze(state):
    start = time.monotonic()
    a = await review_llm().ainvoke(analysis_messages(state))
    await log(state, 'analyze', 'pending', start, a.confidence_reasoning, a)
    return {'analysis': a}

async def node_route(state):
    start = time.monotonic()
    result = route(state)
    d = result['decision']
    await log(state, 'route', {'auto_approve':'auto', 'human_approval':'pending', 'escalate':'escalate'}[d], start, d)
    return result

async def node_approval_pending(state):
    await log(state, 'human_approval', 'pending', time.monotonic(), 'Waiting for confirmation')
    return {}

async def node_human_approval(state):
    # Writes live after interrupt, since this node runs again on resume.
    response = validate_response(interrupt(approval_payload(state)))
    await log(state, 'human_approval', response['human_choice'], time.monotonic(), response.get('human_feedback'), human=True)
    return response

def escalation_payload(state):
    a = state['analysis']
    return dict(kind='escalation', pr_url=state['pr_url'], confidence=a.confidence,
        confidence_reasoning=a.confidence_reasoning, summary=a.summary,
        risk_factors=a.risk_factors, questions=a.escalation_questions or
        ['What is the intended behavior of these changes, and what risks remain?'])

async def node_escalation_pending(state):
    await log(state, 'escalate', 'pending', time.monotonic(), 'Waiting for answers')
    return {}

async def node_escalate(state):
    payload = escalation_payload(state)
    answers = validate_answers(interrupt(payload), payload['questions'])
    await log(state, 'escalate', 'escalate', time.monotonic(), str(answers), human=True)
    return {'escalation_answers': answers}

async def node_synthesize(state):
    start = time.monotonic()
    a = await review_llm().ainvoke(synthesis_messages(state))
    await log(state, 'synthesize', 'pending', start, a.confidence_reasoning, a)
    return {'analysis': a}

async def node_commit(state):
    start = time.monotonic()
    choice = state.get('human_choice')
    auto = state['decision'] == 'auto_approve'
    if not auto and choice not in {'approve', 'edit'}:
        await log(state, 'commit', 'reject', start, state.get('human_feedback'), human=True)
        return {'final_action': 'rejected'}
    body = render_comment(state)
    pr = await asyncio.to_thread(fetch_pr, state['pr_url'])
    if pr.head_sha != state['pr_head_sha']:
        await log(state, 'commit', 'pending', start, 'PR changed; start a new review')
        return {'final_action': 'stale_pr'}
    try:
        await asyncio.to_thread(post_review_comment, state['pr_url'], body)
    except Exception as exc:
        await log(state, 'commit', 'pending', start, f'Posting failed ({type(exc).__name__})')
        return {'final_action': 'commit_failed'}
    await log(state, 'commit', 'auto' if auto else choice, start, body, human=not auto)
    return {'final_action': 'auto_committed' if auto else 'committed', 'posted_comment_body': body}

async def node_auto_approve(state):
    await log(state, 'auto_approve', 'auto', time.monotonic(), state['analysis'].confidence_reasoning)
    return {}

def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name in ('fetch_pr', 'analyze', 'route', 'auto_approve', 'approval_pending',
                 'human_approval', 'escalation_pending', 'escalate', 'synthesize', 'commit'):
        g.add_node(name, globals()[f'node_{name}'])
    for source, target in ((START,'fetch_pr'), ('fetch_pr','analyze'), ('analyze','route'),
        ('auto_approve','commit'), ('approval_pending','human_approval'),
        ('human_approval','commit'), ('escalation_pending','escalate'),
        ('escalate','synthesize'), ('synthesize','approval_pending'), ('commit',END)):
        g.add_edge(source,target)
    g.add_conditional_edges('route', lambda s:s['decision'],
        dict(auto_approve='auto_approve', human_approval='approval_pending', escalate='escalation_pending'))
    return g.compile(checkpointer=checkpointer)

async def session(pr_url, thread_id, resume_value=None, restore=False):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        cp.serde = JsonPlusSerializer(allowed_msgpack_modules=[("common.schemas", "PRAnalysis"), ("common.schemas", "ReviewComment")])
        await cp.setup()
        app = build_graph(cp)
        cfg = {'configurable': {'thread_id': thread_id}}
        snapshot = await app.aget_state(cfg)
        pending = [i for task in snapshot.tasks for i in task.interrupts]
        if restore:
            if not snapshot.values:
                raise ValueError('Session not found')
            return {**snapshot.values, **({'__interrupt__': pending} if pending else {}),
                    '_can_continue': bool(snapshot.next) and not pending}
        if resume_value is not None:
            if not pending:
                raise ValueError('No pending question. Reload this session.')
            payload = pending[0].value
            if payload['kind'] == 'approval_request':
                validate_response(resume_value)
            else:
                validate_answers(resume_value, payload['questions'])
            value = Command(resume=resume_value)
        elif snapshot.values:
            if not snapshot.next:
                return dict(snapshot.values)
            value = None
        else:
            value = dict(pr_url=pr_url, thread_id=thread_id)
        return await app.ainvoke(value, cfg)

def handle_interrupt(payload):
    print(f"\nConfidence: {payload['confidence']:.0%}\n{payload['summary']}\n{payload['confidence_reasoning']}")
    if payload['kind'] == 'escalation':
        answers = {}
        for q in payload['questions']:
            answer = ''
            while not answer:
                answer = input(f'{q}\nAnswer: ').strip()
            answers[q] = answer
        return answers
    print(payload.get('diff_preview', ''))
    for c in payload.get('comments', []):
        print(f"[{c['severity']}] {c['file']}: {c['body']}")
    while True:
        choice = input('approve/reject/edit? ').strip().lower()
        feedback = input('Edited comment body / feedback: ').strip() if choice != 'approve' else ''
        response = dict(choice=choice, feedback=feedback)
        try:
            validate_response(response)
            return response
        except ValueError as exc:
            print(exc)

async def run(pr_url, thread_id=None):
    thread_id = thread_id or str(uuid.uuid4())
    print(f'thread_id = {thread_id}')
    result = await session(pr_url, thread_id)
    while result.get('__interrupt__'):
        result = await session(pr_url, thread_id, handle_interrupt(result['__interrupt__'][0].value))
    print(f"final_action = {result.get('final_action')}")
    print(f'Replay: python -m audit.replay --thread {thread_id}')

def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument('--pr', required=True)
    p.add_argument('--thread')
    args = p.parse_args()
    asyncio.run(run(args.pr, args.thread))

if __name__ == '__main__':
    main()
