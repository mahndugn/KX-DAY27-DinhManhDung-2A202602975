"""Streamlit UI: python -m streamlit run app.py."""
import asyncio
import hashlib
import uuid
import streamlit as st
from dotenv import load_dotenv
from common.github import parse_pr_url
from audit.sessions import recent_sessions
from exercises.exercise_4_audit import session

load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")
st.caption("Gemini · LangGraph · SQLite audit trail")
for key, default in dict(thread_id=None, pr_url="", interrupt_payload=None, final=None, can_continue=False).items():
    st.session_state.setdefault(key, default)


def apply_result(result):
    st.session_state.pr_url = result.get("pr_url", st.session_state.pr_url)
    st.session_state.can_continue = result.get("_can_continue", False)
    pending = result.get("__interrupt__")
    st.session_state.interrupt_payload = pending[0].value if pending else None
    st.session_state.final = None if pending or st.session_state.can_continue else result
    st.query_params["thread"] = st.session_state.thread_id


def drive(resume=None, restore=False):
    try:
        with st.spinner("Processing review…"):
            result = asyncio.run(session(st.session_state.pr_url, st.session_state.thread_id, resume, restore))
        apply_result(result)
        return True
    except Exception as exc:
        st.error(f"Review could not continue ({type(exc).__name__}). Check API configuration, quota and network; then reload this session.")
        return False


with st.sidebar:
    st.header("Recent sessions")
    try:
        for row in asyncio.run(recent_sessions()):
            with st.expander(f"{row['thread_id'][:8]} · {row['worst_risk']}"):
                st.caption(row["pr_url"])
                st.caption(f"Last event: {row['last_event']} · {row['events']} events")
                if st.button("Open session", key=row["thread_id"]):
                    st.session_state.thread_id = row["thread_id"]
                    st.session_state.pr_url = row["pr_url"]
                    if drive(restore=True):
                        st.rerun()
    except Exception as exc:
        st.error(f"Cannot load sessions ({type(exc).__name__}).")
    if st.session_state.thread_id:
        st.caption(f"Session: {st.session_state.thread_id}")
        if st.button("Reload current session") and drive(restore=True):
            st.rerun()

# A query parameter preserves the session after a browser refresh.
requested = st.query_params.get("thread")
if requested and not st.session_state.thread_id:
    st.session_state.thread_id = requested
    drive(restore=True)

with st.form("start"):
    pr_url = st.text_input("PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1")
    submitted = st.form_submit_button("Run review", type="primary")
if submitted:
    try:
        parse_pr_url(pr_url)
    except ValueError:
        st.error("Enter a GitHub pull request URL.")
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.pr_url = pr_url
        st.session_state.interrupt_payload = None
        st.session_state.final = None
        drive()


def render_approval_card(payload):
    st.subheader(f"Approval requested · confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for risk in payload.get("risk_factors", []):
        st.warning(risk)
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")
    with st.expander("Full diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    body = "\n".join([payload["summary"], ""] + [
        f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}"
        for c in payload.get("comments", [])])
    key = st.session_state.thread_id + hashlib.sha256(body.encode()).hexdigest()[:12]
    edited = st.text_area("Edited comment body (used by Edit & post)", value=body, height=200, key=key)
    feedback = st.text_input("Reason for rejection (optional)", key=key+"reason")
    a, b, c = st.columns(3)
    if a.button("Approve & post", type="primary"):
        return dict(choice="approve", feedback="")
    if b.button("Reject"):
        return dict(choice="reject", feedback=feedback)
    if c.button("Edit & post"):
        if edited.strip():
            return dict(choice="edit", feedback=edited)
        st.error("Enter a comment body.")


def render_escalation_card(payload):
    st.subheader(f"Strong escalation · confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for risk in payload.get("risk_factors", []):
        st.error(risk)
    with st.form("escalation_"+st.session_state.thread_id):
        answers = {q: st.text_area(q, key=f"{st.session_state.thread_id}_q{i}") for i,q in enumerate(payload["questions"])}
        if st.form_submit_button("Submit answers"):
            if all(a.strip() for a in answers.values()):
                return answers
            st.error("Answer every question.")


if st.session_state.can_continue:
    st.warning("This session stopped before completion.")
    if st.button("Continue session") and drive():
        st.rerun()
payload = st.session_state.interrupt_payload
if payload:
    answer = render_approval_card(payload) if payload["kind"] == "approval_request" else render_escalation_card(payload)
    if answer is not None and drive(answer):
        st.rerun()
final = st.session_state.final
if final is not None:
    action = final.get("final_action")
    if action in {"committed", "auto_committed"}:
        st.success("Review comment posted.")
        st.link_button("View discussion on GitHub", st.session_state.pr_url)
    elif action == "rejected":
        st.warning("Review rejected. No comment posted.")
    elif action == "stale_pr":
        st.warning("PR changed while pending. Start a new review.")
    elif action == "commit_failed":
        st.error("Posting failed. Check GitHub token permissions and connectivity.")
    else:
        st.info("Session is incomplete. Reload or start a new review.")
    if final.get("analysis"):
        a = final["analysis"]
        st.metric("Confidence", f"{a.confidence:.0%}")
        st.markdown(a.summary)
    if final.get("posted_comment_body"):
        with st.expander("Posted review"):
            st.markdown(final["posted_comment_body"])
