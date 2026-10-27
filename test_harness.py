"""Tests for harness.py — run with: python -m unittest -v test_harness"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import harness
from harness import ANY, DONE, RETRY, Flow, HarnessError, Step, StepError, discover, get, register, registry, step, unregister


def setUpModule():
    os.chdir(Path(__file__).parent)  # so the default ./plugins discovery is deterministic


@step("t.add")
class Add(Step):
    def prep(self, memory):
        return memory.get("terms", [])

    def exec(self, work):
        return sum(work)

    def post(self, memory, work, value):
        memory["sum"] = value
        return "ok"


@step("t.double")
class Double(Step):
    def prep(self, memory):
        return memory.get("sum")

    def exec(self, work):
        return work * 2

    def post(self, memory, work, value):
        memory["doubled"] = value
        return DONE


@step("t.halve")
class Halve(Step):
    def prep(self, memory):
        return memory.get("sum")

    def exec(self, work):
        return work // 2

    def post(self, memory, work, value):
        memory["halved"] = value
        return DONE


@step("t.stop")
class Stop(Step):
    def prep(self, memory):
        return memory.get("nothing")  # None -> end the run


@step("t.boom")
class Boom(Step):
    def prep(self, memory):
        return memory.get("input")

    def exec(self, work):
        raise ValueError("boom")


@step("t.looper")
class Looper(Step):
    def prep(self, memory):
        return memory.get("input")

    def post(self, memory, work, value):
        return False  # -> "Retry", forever


SPEC = {
    "name": "math",
    "start": "add",
    "steps": {"add": "t.add", "next": "t.double"},
    "edges": {"add": {"ok": "next"}},
}


class FlowTest(unittest.TestCase):
    def test_linear_chain_writes_shared_memory(self):
        report = Flow.from_spec(SPEC).run({"terms": [1, 2, 3]})
        self.assertTrue(report.ok)
        self.assertEqual(report.memory["sum"], 6)
        self.assertEqual(report.memory["doubled"], 12)
        self.assertEqual(report.trail, "add→next")

    def test_prep_none_ends_the_run(self):
        report = Flow.from_spec({"start": "s", "steps": {"s": "t.stop"}}).run({})
        self.assertTrue(report.ok)
        self.assertEqual(len(report.steps), 1)
        self.assertIsNone(report.steps[0].role)

    def test_wildcard_role_matches_anything(self):
        spec = {
            "start": "add",
            "steps": {"add": "t.add", "any": "t.halve"},
            "edges": {"add": {ANY: "any"}},
        }
        report = Flow.from_spec(spec).run({"terms": [4, 4]})
        self.assertEqual(report.memory["halved"], 4)

    def test_step_local_success_is_used_without_flow_edges(self):
        @step("t.local")
        class Local(Step):
            success = {"ok": "t.halve"}

            def prep(self, memory):
                return memory.get("terms")

            def post(self, memory, work, value):
                memory["sum"] = sum(work)
                return "ok"

        report = Flow.from_spec({"start": "a", "steps": {"a": "t.local"}}).run({"terms": [9, 1]})
        self.assertEqual(report.trail, "a→t.halve")
        self.assertEqual(report.memory["halved"], 5)

    def test_failure_routed_to_handler_is_recovered(self):
        spec = {"start": "a", "steps": {"a": "t.boom", "fix": "t.halve"}, "edges": {"a": {"Fail": "fix"}}}
        report = Flow.from_spec(spec).run({"input": 8, "sum": 8})
        self.assertTrue(report.ok)
        self.assertEqual(report.trail, "a→fix")
        self.assertEqual(report.memory["halved"], 4)
        self.assertIsNone(report.error)
        self.assertIsNone(report.steps[0].value)
        self.assertIn('File "', report.steps[0].error)
        self.assertIn("ValueError: boom", report.steps[0].error)

    def test_unwired_failure_is_an_error_status(self):
        report = Flow.from_spec({"start": "a", "steps": {"a": "t.boom"}}).run({"input": 1})
        self.assertEqual(report.status, "error:Fail")
        self.assertIsInstance(report.error, ValueError)
        with self.assertRaises(StepError):
            report.raise_for_status()

    def test_unwired_success_role_stalls(self):
        report = Flow.from_spec({"start": "a", "steps": {"a": "t.add"}}).run({"terms": [1]})
        self.assertEqual(report.status, "stalled:ok")
        self.assertIsNone(report.error)

    def test_retry_loops_until_the_budget_stops_it(self):
        spec = {"start": "a", "steps": {"a": "t.looper"}, "budget": 4}
        report = Flow.from_spec(spec).run({"input": 1})
        self.assertEqual(report.status, "budget-exhausted")
        self.assertEqual(len(report.steps), 4)
        self.assertEqual({r.round for r in report.steps}, {0, 1, 2, 3})

    def test_role_true_and_false_are_normalised(self):
        @step("t.roles")
        class Roles(Step):
            def post(self, memory, work, value):
                memory["seen"] = memory.get("seen", 0) + 1
                return False if memory["seen"] == 1 else True

        report = Flow.from_spec({"start": "a", "steps": {"a": "t.roles"}, "budget": 3}).run({})
        self.assertEqual([r.role for r in report.steps], [RETRY, DONE])
        self.assertEqual([r.round for r in report.steps], [0, 1])

    def test_trace_callback_sees_every_round(self):
        seen = []
        Flow.from_spec(SPEC).run({"terms": [1]}, trace=seen.append)
        self.assertEqual([r.name for r in seen], ["add", "next"])
        self.assertTrue(all(r.seconds >= 0 for r in seen))

    def test_nested_flow_runs_as_a_single_step(self):
        inner = {"name": "inner", "start": "d", "steps": {"d": "t.double"}}
        spec = {"start": "add", "steps": {"add": "t.add"}, "edges": {"add": {"ok": inner}}}
        report = Flow.from_spec(spec).run({"terms": [2, 2]})
        self.assertTrue(report.ok)
        self.assertEqual(report.memory["doubled"], 8)
        self.assertEqual(report.steps[-1].value.status, "ok")
        self.assertEqual(report.steps[-1].value.trail, "d")

    def test_flow_object_as_a_step_shares_memory(self):
        spec = {"start": "add", "steps": {"add": "t.add"}, "edges": {"add": {"ok": Flow.from_spec({"start": "h", "steps": {"h": "t.halve"}})}}}
        report = Flow.from_spec(spec).run({"terms": [7, 1]})
        self.assertEqual(report.memory["halved"], 4)

    def test_reusable_flow_runs_twice_with_fresh_memory(self):
        flow = Flow.from_spec(SPEC)
        for value in (3, 10):
            report = flow.run({"terms": [value]})
            self.assertEqual(report.memory["doubled"], value * 2)

    def test_wiring_errors(self):
        with self.assertRaises(HarnessError):
            Flow.from_spec({"start": "a", "steps": {"a": "no.such.step"}})
        with self.assertRaises(HarnessError):
            Flow.from_spec({"start": "ghost", "steps": {"a": "t.add"}})
        with self.assertRaises(HarnessError):
            Flow.from_spec({"start": "a", "steps": {"a": "t.add"}, "edges": {"missing": {"ok": "a"}}})
        with self.assertRaises(HarnessError):
            Flow.from_spec({"start": "a", "steps": {"a": "t.add"}, "edges": {"a": {"ok": "a"}}}).add(Add(), name="a")
        with self.assertRaises(HarnessError):
            register("t.bad", dict)
        flow = Flow(name="empty")
        with self.assertRaises(HarnessError):
            flow.run({})

    def test_unknown_role_name_in_registry_reports_known_steps(self):
        with self.assertRaises(HarnessError) as caught:
            get("t.not.here")
        self.assertIn("t.add", str(caught.exception))

    def test_edge_can_name_a_step_that_is_plugged_in_later(self):
        @step("t.plain")
        class Plain(Step):
            def post(self, memory, work, value):
                return "ok"

        spec = {"start": "a", "steps": {"a": "t.plain"}, "edges": {"a": {"ok": "t.late"}}}
        flow = Flow.from_spec(spec)
        with self.assertRaises(HarnessError):
            flow.run({})

        @step("t.late")
        class Late(Step):
            def prep(self, memory):
                return memory.get("sum")

            def post(self, memory, work, value):
                memory["halved"] = work // 2
                return DONE

        report = flow.run({"sum": 10})
        self.assertEqual(report.trail, "a→t.late")
        self.assertEqual(report.memory["halved"], 5)
        self.assertIs(flow.get_step("t.late"), flow.get_step("t.late"))  # resolved once
        self.assertEqual(flow.run({"sum": 4}).memory["halved"], 2)
        unregister("t.late")

    def test_a_broken_failure_handler_is_a_hard_failure(self):
        """A handler that raises is a bug, not data — it must not be swallowed."""

        @step("t.badhandler")
        class BadHandler(Step):
            def exec(self, work):
                raise RuntimeError("the work failed")

            def fail(self, memory, work, error):
                raise ValueError("the handler failed")

        spec = {"start": "a", "steps": {"a": "t.badhandler"}}
        with self.assertRaises(ValueError):
            Flow.from_spec(spec).run({})


class PluginTest(unittest.TestCase):
    def test_discover_loads_the_default_plugins_directory(self):
        found = discover()
        self.assertEqual(found["errors"], {})
        self.assertIn("demo.fetch", found["loaded"])
        self.assertIn("text.split", registry())
        # only the repo's own packs, ignoring any distribution installed elsewhere
        self.assertEqual(discover(group=None)["modules"], ["flaky_source", "text_tools"])

    def test_discover_is_idempotent(self):
        first = discover()
        second = discover()
        self.assertEqual(first["loaded"], second["loaded"])
        self.assertTrue(set(first["loaded"]) <= set(registry()))

    def test_a_broken_plugin_does_not_break_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "good.py").write_text("from harness import Step, step\n\n@step('x.good')\nclass G(Step):\n    pass\n", encoding="utf-8")
            Path(tmp, "bad.py").write_text("import nonexistent_module_xyz\n", encoding="utf-8")
            Path(tmp, "_ignored.py").write_text("raise Exception('should not run')\n", encoding="utf-8")
            found = discover(tmp, group=None)
            self.assertIn("x.good", found["loaded"])
            self.assertIn("bad", found["errors"])
            self.assertNotIn("_ignored", found["modules"])
        self.addCleanup(unregister, "x.good")

    def test_swapping_the_plug_changes_what_the_same_spec_does(self):
        discover()
        original = registry()["text.split"]
        self.addCleanup(register, "text.split", original, replace=True)
        spec = {
            "start": "split",
            "steps": {"split": "text.split", "count": "text.count"},
            "edges": {"split": {"ok": "count"}},
        }

        self.assertEqual(Flow.from_spec(spec).run({"text": "a b c"}).memory["count"], 3)

        @step("text.split", replace=True)
        class CommaSplit(Step):
            def prep(self, memory):
                return memory.get("text")

            def exec(self, work):
                return [part for part in work.split(",") if part]

            def post(self, memory, work, value):
                memory["words"] = value
                return "ok" if value else "empty"

        # identical spec, different plug: whitespace is now a single token
        swapped = Flow.from_spec(spec).run({"text": "a b c"})
        self.assertEqual(swapped.memory["count"], 1)
        self.assertEqual(swapped.memory["words"], ["a b c"])
        self.assertEqual(Flow.from_spec(spec).run({"text": "a,b,c"}).memory["count"], 3)
        self.assertEqual(Flow.from_spec(spec).run({"text": ""}).status, "stalled:empty")
        self.assertEqual(Flow.from_spec(spec).run({"text": ""}).memory["words"], [])
        self.assertIn(("text.split", "text_tools", __name__), harness.overrides())

    def test_duplicate_registration_needs_replace(self):
        with self.assertRaises(HarnessError):
            register("t.add", Double)
        register("t.add", Double, replace=True)
        self.addCleanup(register, "t.add", Add, replace=True)
        self.assertIs(get("t.add"), Double)

    def test_entry_point_plugins_are_scanned_and_exported(self):
        class FakeEP:
            name = "ep.pack"

            def load(self):
                return {"ep.shout": Shout, "ep.whisper": Whisper}

        with mock.patch("harness.importlib.metadata.entry_points", return_value=[FakeEP()]):
            found = discover(group="harness.steps")
        self.assertIn("ep.shout", found["loaded"])
        self.assertIn("ep.pack", found["modules"])
        for name in ("ep.shout", "ep.whisper"):
            self.addCleanup(unregister, name)

    def test_entry_point_can_return_a_bare_class(self):
        class FakeEP:
            name = "ep.single"

            def load(self):
                return Shout

        with mock.patch("harness.importlib.metadata.entry_points", return_value=[FakeEP()]):
            found = discover(paths=[], group="harness.steps")
        self.assertEqual(found["loaded"], ["ep.single"])
        self.addCleanup(unregister, "ep.single")

    def test_failing_entry_point_is_reported_not_raised(self):
        class FakeEP:
            name = "ep.broken"

            def load(self):
                raise RuntimeError("bad distribution")

        with mock.patch("harness.importlib.metadata.entry_points", return_value=[FakeEP()]):
            found = discover(paths=[], group="harness.steps")
        self.assertIn("ep.broken", found["errors"])

    def test_end_to_end_pipeline_from_plugs_only(self):
        found = discover()
        self.assertEqual(found["errors"], {})
        Fetch = registry()["demo.fetch"]
        spec = {
            "name": "demo",
            "start": "fetch",
            "budget": 10,
            "steps": {
                "fetch": "demo.fetch",
                "split": "text.split",
                "count": "text.count",
                "gate": "demo.gate",
                "fallback": "demo.noop",
                "longest": "text.longest",
            },
            "edges": {
                "fetch": {"ok": "split", "Fail": "fetch"},
                "split": {"ok": "count", "empty": "fallback"},
                "count": {"ok": "gate"},
                "gate": {"ok": "longest", "bad": "fallback"},
                "longest": {"ok": "count"},
            },
        }
        report = Flow.from_spec(spec).run({"url": "https://example.com/x"})
        self.assertTrue(report.ok, str(report))
        self.assertEqual(report.trail, "fetch→fetch→fetch→split→count→gate→longest")
        self.assertEqual(report.memory["count"], 9)
        self.assertEqual(report.memory["longest"], "quick")
        self.assertNotIn("degraded", report.memory)
        self.assertIn("connection reset by https://example.com/x (attempt 1)", report.steps[0].error)
        self.assertIn("attempt 2", report.steps[1].error)
        self.assertIsNone(report.steps[2].error)
        self.assertEqual(report.steps[0].round, 0)
        self.assertEqual(report.steps[1].round, 1)

        # the same graph, with the flaky source swapped for a reliable one
        @step("demo.fetch", replace=True)
        class ReliableFetch(Step):
            def prep(self, memory):
                return memory.get("url")

            def exec(self, work):
                return "one two three"

            def post(self, memory, work, value):
                memory["text"] = value
                return "ok"

        self.addCleanup(register, "demo.fetch", Fetch, replace=True)
        fast = Flow.from_spec(spec).run({"url": "https://example.com/y"})
        self.assertEqual(fast.trail, "fetch→split→count→gate→longest")
        self.assertEqual(fast.memory["count"], 3)

    def test_example_in_the_module_docstring_runs(self):
        discover()
        spec = {
            "name": "pipeline",
            "start": "fetch",
            "steps": {"fetch": "demo.fetch", "split": "text.split", "top": "text.longest"},
            "edges": {
                "fetch": {"ok": "split", "Fail": "fetch"},
                "split": {"ok": "top", "empty": "fetch"},
            },
        }
        report = Flow.from_spec(spec).run({"url": "https://example.com/feed.json"})
        self.assertEqual(report.trail, "fetch→fetch→fetch→split→top")
        self.assertEqual(report.memory["longest"], "quick")
        report.raise_for_status()


class Shout(Step):
    """Two undecorated classes: an entry point can export a dict of these."""

    def exec(self, work):
        return "SHOUT"


class Whisper(Step):
    def exec(self, work):
        return "whisper"


if __name__ == "__main__":
    unittest.main(verbosity=2)
