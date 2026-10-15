import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from unittest.mock import Mock
from langgraph.types import Command

from common.db import replay_events
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as workflow
from exercises.exercise_1_confidence import node_route
from audit.sessions import recent_sessions
from exercises import exercise_1_confidence as ex1, exercise_2_hitl as ex2, exercise_3_escalation as ex3


def analysis(confidence):
    return PRAnalysis(summary="Test review", confidence=confidence,
        confidence_reasoning="Test rationale", escalation_questions=["Why this change?"])


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, HITL_DB_PATH=os.path.join(self.tmp.name, "audit.db"), GITHUB_USER="reviewer")
        self.env.start()
        self.pr = SimpleNamespace(title="Example", author="author", diff="+example",
            files_changed=["example.py"], head_sha="abc")
        self.fetch = patch.object(workflow, "fetch_pr", return_value=self.pr)
        self.fetch.start()
        self.post = patch.object(workflow, "post_review_comment")
        self.post_mock = self.post.start()
        self.model = SimpleNamespace(ainvoke=AsyncMock(return_value=analysis(.65)))
        self.llm = patch.object(workflow, "review_llm", return_value=self.model)
        self.llm.start()

    async def asyncTearDown(self):
        self.llm.stop()
        self.post.stop()
        self.fetch.stop()
        self.env.stop()
        self.tmp.cleanup()

    async def test_boundaries(self):
        for score, branch in [(0,"escalate"),(.5799,"escalate"),(.58,"human_approval"),(.7299,"human_approval"),(.73,"auto_approve"),(1,"auto_approve")]:
            self.assertEqual(node_route({"analysis":analysis(score)})["decision"],branch)

    async def test_restore_approve_no_duplicate_audit_or_post(self):
        initial = await workflow.session("https://github.com/o/r/pull/1", "approve")
        self.assertEqual(initial["__interrupt__"][0].value["kind"], "approval_request")
        restored = await workflow.session("", "approve", restore=True)
        self.assertEqual(restored["pr_head_sha"], "abc")
        final = await workflow.session("", "approve", {"choice":"approve"})
        self.assertEqual(final["final_action"], "committed")
        await workflow.session("", "approve")
        self.post_mock.assert_called_once()
        events = await replay_events("approve")
        self.assertEqual(sum(e["action"]=="human_approval" and e["decision"]=="pending" for e in events),1)
        self.assertTrue(any(e["reviewer_id"]=="reviewer" for e in events))

    async def test_reject(self):
        await workflow.session("https://github.com/o/r/pull/1", "reject")
        final = await workflow.session("", "reject", {"choice":"reject"})
        self.assertEqual(final["final_action"], "rejected")
        self.post_mock.assert_not_called()

    async def test_edit(self):
        await workflow.session("https://github.com/o/r/pull/1", "edit")
        with self.assertRaises(ValueError):
            await workflow.session("", "edit", {"choice":"edit", "feedback":""})
        await workflow.session("", "edit", {"choice":"edit", "feedback":"My edited review"})
        self.assertEqual(self.post_mock.call_args.args[1],"My edited review")

    async def test_escalation_final_confirmation_and_risk_order(self):
        self.model.ainvoke.side_effect = [analysis(.4),analysis(.8)]
        result = await workflow.session("https://github.com/o/r/pull/2", "escalate")
        self.assertEqual(result["__interrupt__"][0].value["kind"],"escalation")
        with self.assertRaises(ValueError):
            await workflow.session("", "escalate", {})
        result = await workflow.session("", "escalate", {"Why this change?":"Intent explained"})
        self.assertEqual(result["__interrupt__"][0].value["kind"],"approval_request")
        self.post_mock.assert_not_called()
        await workflow.session("", "escalate", {"choice":"reject"})
        self.post_mock.assert_not_called()
        self.assertEqual((await recent_sessions())[0]["worst_risk"],"high")

    async def test_auto(self):
        self.model.ainvoke.return_value = analysis(.9)
        result = await workflow.session("https://github.com/o/r/pull/1", "auto")
        self.assertEqual(result["final_action"],"auto_committed")
        self.post_mock.assert_called_once()

    async def test_stale_head(self):
        await workflow.session("https://github.com/o/r/pull/1", "stale")
        self.pr.head_sha = "changed"
        result = await workflow.session("", "stale", {"choice":"approve"})
        self.assertEqual(result["final_action"],"stale_pr")
        self.post_mock.assert_not_called()

    async def test_post_failure(self):
        self.model.ainvoke.return_value = analysis(.9)
        self.post_mock.side_effect = RuntimeError("offline")
        result = await workflow.session("https://github.com/o/r/pull/1", "fail")
        self.assertEqual(result["final_action"],"commit_failed")

    async def test_progressive_exercises(self):
        model = SimpleNamespace(invoke=Mock(return_value=analysis(.65)))
        with patch("exercises.review.review_llm", return_value=model):
            for module in (ex1, ex2, ex3):
                with self.subTest(module=module.__name__), patch.object(module,"fetch_pr",return_value=self.pr):
                    graph = module.build_graph()
                    cfg = {"configurable":{"thread_id":"progressive"}}
                    result = graph.invoke({"pr_url":"https://github.com/o/r/pull/1"},cfg)
                    if module is ex1:
                        self.assertEqual(result["final_action"],"pending_human_approval")
                    else:
                        self.assertTrue(result["__interrupt__"])
                        with patch.object(module,"post_review_comment") as post:
                            result = graph.invoke(Command(resume={"choice":"edit","feedback":"Edited"}),cfg)
                            self.assertEqual(result["final_action"],"committed")
                            self.assertEqual(post.call_args.args[1],"Edited")


if __name__ == "__main__":
    unittest.main()
