import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
from streamlit.testing.v1 import AppTest
from exercises.review import approval_payload
from test_workflow import analysis


class UiTests(unittest.TestCase):
    def test_approval_edit_posts_resume_value(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, HITL_DB_PATH=os.path.join(tmp,"ui.db")):
            state = dict(pr_url="https://github.com/o/r/pull/1", analysis=analysis(.65), pr_diff="+example")
            response = {**state, "__interrupt__":[SimpleNamespace(value=approval_payload(state))]}
            with patch("exercises.exercise_4_audit.session", new=AsyncMock(side_effect=[response,{**state,"final_action":"committed","posted_comment_body":"Edited review"}])) as invoke:
                app = AppTest.from_file("../app.py").run()
                app.text_input[0].set_value(state["pr_url"])
                app.button(key="FormSubmitter:start-Run review").click().run()
                self.assertFalse(app.exception)
                app.text_area[0].set_value("Edited review")
                next(b for b in app.button if b.label=="Edit & post").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(invoke.call_args.args[2], {"choice":"edit","feedback":"Edited review"})
                self.assertTrue(app.success)

    def test_escalation_requires_answers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, HITL_DB_PATH=os.path.join(tmp,"ui.db")):
            payload = dict(kind="escalation", confidence=.4, confidence_reasoning="Risk", summary="Review", risk_factors=["Unsafe"], questions=["Why?"])
            with patch("exercises.exercise_4_audit.session", new=AsyncMock(return_value={"__interrupt__":[SimpleNamespace(value=payload)]})) as invoke:
                app = AppTest.from_file("../app.py").run()
                app.text_input[0].set_value("https://github.com/o/r/pull/2")
                app.button(key="FormSubmitter:start-Run review").click().run()
                next(b for b in app.button if b.label=="Submit answers").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(invoke.await_count,1)
                self.assertTrue(any("Answer every" in e.value for e in app.error))
