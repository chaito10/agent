"""Tests for agent.py — python -m unittest -v test_agent"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent))

import agent as A
from agent import Agent, Context, Hooks, Memory, Policy, Registry, Skills, Tasks, Scheduler, Prompts, Tool, ToolError, first_json


def workspace() -> Path:
    return Path(tempfile.mkdtemp(prefix="agent-test-"))


def make_agent(root=None, **kw) -> Agent:
    return Agent(root=root or workspace(), skills_dir=None, memory_path=None, **kw)


class FunnelTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_tool_call_roundtrip(self):
        record = self.agent.call("echo", {"text": "hi"})
        self.assertTrue(record.ok)
        self.assertEqual(record.value, {"echo": "hi"})
        self.assertEqual(record.kind, "tool")

    def test_unknown_tool_is_a_record_not_a_crash(self):
        record = self.agent.call("no.such.tool", {})
        self.assertFalse(record.ok)
        self.assertIn("no tool", record.error)
        self.assertEqual(self.agent.journal[-1], record)

    def test_missing_and_unexpected_args(self):
        self.assertIn("missing required", self.agent.call("echo", {}).error)
        self.assertIn("unexpected arg", self.agent.call("echo", {"text": "x", "nope": 1}).error)

    def test_arguments_are_coerced_from_json(self):
        reg = Registry()
        reg.add(Tool("t.coerce", "echo typed args",
                     args={"n": int, "f": float, "b": bool, "l": list, "d": dict, "s": str},
                     handler=lambda **kw: kw))
        seen = reg.get("t.coerce").call({"n": "42", "f": "1.5", "b": "yes", "l": "a, b", "d": '{"k": 1}', "s": 7})
        self.assertEqual(seen["n"], 42)
        self.assertEqual(seen["f"], 1.5)
        self.assertIs(seen["b"], True)
        self.assertEqual(seen["l"], ["a", "b"])
        self.assertEqual(seen["d"], {"k": 1})
        self.assertEqual(seen["s"], "7")

    def test_operator_defaults_merge_into_every_call(self):
        (self.agent.root / "notes.md").write_text("a", encoding="utf-8")
        self.assertEqual("fs.read is missing required arg(s): path", self.agent.call("fs.read", {}).error)
        self.agent.tool_defaults["fs.read"] = {"path": "notes.md"}
        self.assertIn("notes.md", self.agent.call("fs.read", {}).value["path"])
        (self.agent.root / "other.txt").write_text("b", encoding="utf-8")
        self.assertIn("other.txt", self.agent.call("fs.read", {"path": "other.txt"}).value["path"])
        self.assertEqual("other.txt", self.agent.journal[-1].args["path"])   # merged args are what policy sees

    def test_plugin_exception_becomes_data(self):
        @A.tool("t.explode", "always raises", registry=self.agent.registry)
        def explode():
            raise ZeroDivisionError("nope")

        record = self.agent.call("t.explode", {})
        self.assertFalse(record.ok)
        self.assertIn("ZeroDivisionError: nope", record.error)

    def test_agents_get_isolated_registries(self):
        other = make_agent()
        self.agent.registry.add(Tool("t.only_here", "x", handler=lambda: 1))
        self.assertFalse(other.registry.has("t.only_here"))
        self.assertTrue(self.agent.registry.has("echo"))

    def test_tools_call_meta_tool_goes_through_the_funnel(self):
        record = self.agent.call("tools.call", {"tool": "echo", "args": {"text": "meta"}})
        self.assertTrue(record.ok)
        self.assertEqual(record.value["result"], {"echo": "meta"})


class ActionContractTest(unittest.TestCase):
    def test_first_json_from_fenced_and_noisy_replies(self):
        self.assertEqual(A.first_json('```json\n{"thought": "a", "final": "b"}\n```')["final"], "b")
        self.assertEqual(A.first_json('sure! {"call": {"tool": "echo", "args": {"x": 1}}} done')["call"]["tool"], "echo")
        self.assertEqual(A.first_json('{"a": {"b": "}"}}')["a"]["b"], "}")
        self.assertIsNone(A.first_json("no object here"))
        self.assertIsNone(A.first_json('{"broken": '))


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        self.agent.grant("fs.write", repeats=True)

    def test_deny_wins_and_reports_the_rule(self):
        ok, reason = self.agent.policy.check(self.agent.registry.get("fs.delete"), {"path": "x"})
        self.assertFalse(ok)
        self.assertIn("denied by guardrail rule 'fs.delete'", reason)

    def test_allow_list_locks_the_surface(self):
        self.agent.policy.set(allow=["echo", "fs.*"])
        self.assertTrue(self.agent.policy.check(self.agent.registry.get("echo"), {})[0])
        ok, reason = self.agent.policy.check(self.agent.registry.get("memory.list"), {})
        self.assertFalse(ok)
        self.assertIn("allow list", reason)

    def test_paths_outside_the_sandbox_are_refused(self):
        ok, reason = self.agent.policy.check(self.agent.registry.get("fs.read"), {"path": "../../etc/passwd"})
        self.assertFalse(ok)
        self.assertIn("escapes the sandbox", reason)

    def test_forbidden_command_patterns_are_refused(self):
        # shell.* is denied by name out of the box; lift that to exercise the content rule
        self.agent.policy.set(confirm_required=[], deny=["agent.run", "fs.delete"])
        ok, reason = self.agent.policy.check(self.agent.registry.get("shell.run"), {"cmd": "sudo rm -rf /tmp/x"})
        self.assertFalse(ok)
        self.assertIn("is forbidden", reason)
        self.assertIn("is forbidden", self.agent.call("shell.run", {"cmd": "shutdown now"}).reason)
        self.assertTrue(self.agent.policy.check(self.agent.registry.get("shell.run"), {"cmd": "python -V"})[0])

    def test_extra_dirs_widen_the_sandbox(self):
        extra = workspace()
        self.agent.policy.set(extra_dirs=[str(extra)])
        self.assertTrue(self.agent.policy.check(self.agent.registry.get("fs.read"), {"path": str(extra / "a.txt")})[0])

    def test_write_and_delete_budgets_are_separate(self):
        self.agent.policy.set(max_writes=2, max_deletes=0, deny=["agent.run", "shell.*"])
        self.assertTrue(self.agent.call("fs.write", {"path": "a.txt", "content": "1"}).ok)
        self.assertTrue(self.agent.call("fs.write", {"path": "b.txt", "content": "2"}).ok)
        blocked = self.agent.call("fs.write", {"path": "c.txt", "content": "3"})
        self.assertIn("write budget 2", blocked.reason)
        self.assertEqual(self.agent.policy.writes, 2)
        self.assertEqual(self.agent.policy.deletes, 0)

    def test_delete_budget_charges_deletes(self):
        self.agent.policy.set(deny=["agent.run", "shell.*"], max_deletes=1, confirm_required=[])
        (self.agent.root / "gone.txt").write_text("x", encoding="utf-8")
        self.assertTrue(self.agent.call("fs.delete", {"path": "gone.txt"}).ok)
        self.assertEqual(self.agent.policy.deletes, 1)

    def test_dry_run_refuses_every_mutating_tool(self):
        self.agent.policy.set(dry_run=True)
        record = self.agent.call("fs.write", {"path": "a.txt", "content": "x"})
        self.assertTrue(record.blocked)
        self.assertIn("dry-run", record.reason)
        self.assertFalse((self.agent.root / "a.txt").exists())
        self.assertTrue(self.agent.call("fs.read", {"path": "nope"}).error)  # read-only tools still run

    def test_network_tools_are_off_by_default(self):
        record = self.agent.call("model.ollama", {"messages": []})
        self.assertTrue(record.blocked)
        self.assertIn("network tools are off", record.reason)

    def test_confirmation_is_one_shot_unless_repeats(self):
        self.agent.policy.set(confirm_required=["echo"])
        self.assertTrue(self.agent.call("echo", {"text": "x"}).blocked)
        self.agent.grant("echo")
        self.assertTrue(self.agent.call("echo", {"text": "x"}).ok)
        self.assertTrue(self.agent.call("echo", {"text": "x"}).blocked)
        self.agent.grant("echo", repeats=True)
        for _ in range(3):
            self.assertTrue(self.agent.call("echo", {"text": "x"}).ok)
        self.agent.call("confirm.revoke", {"tool": "echo"})
        self.assertTrue(self.agent.call("echo", {"text": "x"}).blocked)

    def test_wall_clock_limit(self):
        self.agent.policy.set(max_seconds=1)
        self.agent.policy.started_at = time.time() - 10
        ok, reason = self.agent.policy.check(self.agent.registry.get("echo"), {})
        self.assertFalse(ok)
        self.assertIn("wall-clock", reason)

    def test_policy_check_tool_does_not_charge_the_probe_target(self):
        for _ in range(3):
            probe = self.agent.call("policy.check", {"tool": "fs.write", "args": {"path": "a.txt"}})
            self.assertTrue(probe.ok)
            self.assertTrue(probe.value["allowed"])
        used = self.agent.policy.summary()["used"]
        self.assertEqual(used["writes"], 0)  # the fs.write it evaluated was never charged
        self.assertEqual(used["deletes"], 0)
        self.assertEqual(used["calls"], 3)  # only the probes themselves, one each

    def test_unknown_policy_knob(self):
        with self.assertRaises(ToolError):
            self.agent.policy.set(max_duck=3)

    def test_prompt_block_summarises_state(self):
        text = self.agent.policy.prompt_block()
        self.assertIn("network=off", text)
        self.assertIn("needs approval=", text)


class BuiltinsTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()
        (self.agent.root / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

    def test_fs_edit_patches_one_unique_match(self):
        self.agent.grant("fs.edit", repeats=True)
        record = self.agent.call("fs.edit", {"path": "app.py", "find": "return a - b", "replace": "return a + b"})
        self.assertTrue(record.ok)
        self.assertEqual(record.value["replacements"], 1)
        self.assertIn("a + b", (self.agent.root / "app.py").read_text(encoding="utf-8"))

    def test_fs_edit_refuses_an_ambiguous_needle_unless_all(self):
        self.agent.grant("fs.edit", repeats=True)
        (self.agent.root / "dup.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        vague = self.agent.call("fs.edit", {"path": "dup.py", "find": "x = 1", "replace": "y = 2"})
        self.assertIn("matches 2 times", vague.error)
        self.assertEqual(self.agent.call("fs.edit", {"path": "dup.py", "find": "gone", "replace": "z"}).error,
                         "no match for 'gone' in dup.py")
        self.assertEqual(self.agent.call("fs.edit", {"path": "dup.py", "find": "x = 1",
                                                     "replace": "y = 2", "all": True}).value["replacements"], 2)

    def test_fs_edit_is_an_approval_and_write_budget_tool(self):
        self.assertIn("fs.edit", self.agent.policy.confirm_required)
        blocked = self.agent.call("fs.edit", {"path": "app.py", "find": "a - b", "replace": "a + b"})
        self.assertTrue(blocked.blocked)
        self.assertIn("needs approval", blocked.reason)
        self.assertIn("a - b", (self.agent.root / "app.py").read_text(encoding="utf-8"))

    def test_fs_edit_stays_inside_the_sandbox(self):
        self.agent.grant("fs.edit", repeats=True)
        record = self.agent.call("fs.edit", {"path": "../elsewhere.py", "find": "a", "replace": "b"})
        self.assertTrue(record.blocked)
        self.assertIn("escapes the sandbox", record.reason)

    def test_browser_open_is_a_network_tool(self):
        record = self.agent.call("browser.open", {"url": "data:text/plain,hello"})
        self.assertTrue(record.blocked)
        self.assertIn("network tools are off", record.reason)

    def test_html_reduce_keeps_title_text_and_links(self):
        page = ("<html><head><title>Notes &amp; more</title><style>p{color:red}</style></head>"
                "<body><script>secret()</script><p>one   funnel</p><a href=\"/next\">Next</a></body></html>")
        url = "data:text/html;charset=utf-8," + A.urllib.parse.quote(page)
        self.agent.policy.set(allow_network=True)
        seen = self.agent.call("browser.open", {"url": url}).value
        self.assertEqual(seen["status"], 200)
        self.assertEqual(seen["title"], "Notes & more")
        self.assertEqual(seen["text"], "one funnel Next")
        self.assertEqual(seen["links"], [{"href": "/next", "text": "Next"}])
        self.assertNotIn("secret()", seen["text"])


class HookTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_pre_tool_hook_blocks(self):
        self.agent.hook("pre_tool", "echo", call=lambda payload: {"block": True, "reason": "no echoes"})
        record = self.agent.call("echo", {"text": "x"})
        self.assertTrue(record.blocked)
        self.assertEqual(record.reason, "no echoes")
        self.assertEqual(self.agent.hooks.fired[-1]["hook"], "pre_tool:echo")
        self.assertEqual(self.agent.policy.blocks[-1]["tool"], "echo")

    def test_pre_tool_hook_can_rewrite_args(self):
        self.agent.hook("pre_tool", "echo", call=lambda payload: {"args": {"text": "rewritten"}})
        record = self.agent.call("echo", {"text": "original"})
        self.assertEqual(record.value, {"echo": "rewritten"})
        self.assertEqual(record.args["text"], "rewritten")

    def test_post_tool_veto_turns_success_into_a_block(self):
        self.agent.hook("post_tool", "echo", call=lambda payload: {"block": True, "reason": "result is not fit"})
        record = self.agent.call("echo", {"text": "x"})
        self.assertFalse(record.ok)
        self.assertIn("post_tool veto", record.reason)

    def test_a_broken_hook_blocks_instead_of_hanging(self):
        def bad(payload):
            raise RuntimeError("hook bug")

        self.agent.hook("pre_tool", "*", call=bad)
        record = self.agent.call("echo", {"text": "x"})
        self.assertIn("raised RuntimeError: hook bug", record.reason)

    def test_pattern_matches_tool_names_only(self):
        blocked = self.agent.hook("pre_tool", "fs.*", call=lambda payload: {"block": True, "reason": "no fs"})
        self.assertTrue(self.agent.call("fs.read", {"path": "x"}).blocked)
        self.assertTrue(self.agent.call("echo", {"text": "x"}).ok)
        self.assertEqual(blocked.pattern, "fs.*")

    def test_hook_handler_can_be_a_registered_tool(self):
        calls = []

        def audit(payload):
            calls.append(payload["tool"])
            return None

        self.agent.registry.add(Tool("audit.record", "record the tool name", args={"payload": dict}, handler=audit))
        self.agent.hook("pre_tool", "*", tool="audit.record")
        self.agent.call("echo", {"text": "x"})
        self.assertEqual(calls, ["echo"])

    def test_tool_error_event_fires(self):
        seen = []
        self.agent.hook("tool_error", "*", call=lambda payload: seen.append(payload["error"]))
        self.agent.call("echo", {})
        self.assertEqual(len(seen), 1)

    def test_hook_tools_can_only_attach_registered_tool_handlers(self):
        self.agent.registry.add(Tool("veto.run", "always blocks", args={"payload": dict},
                                     handler=lambda payload: {"block": True, "reason": f"vetoed {payload['tool']}"}))
        added = self.agent.call("hook.add", {"event": "pre_tool", "pattern": "echo", "tool": "veto.run"})
        self.assertTrue(added.ok, added.error)
        self.assertEqual(added.value, {"attached": "veto.run", "event": "pre_tool", "pattern": "echo"})

        unregistered = self.agent.call("hook.add", {"event": "pre_tool", "tool": "audit.missing"})
        self.assertIn("not registered", unregistered.error)
        handlerless = self.agent.call("hook.add", {"event": "pre_tool"})
        self.assertIn("needs either a callable or a registered tool", handlerless.error)
        bogus = self.agent.call("hook.add", {"event": "pre_dance"})
        self.assertIn("unknown hook event", bogus.error)

        self.assertTrue(self.agent.call("echo", {"text": "x"}).blocked)
        self.assertEqual(self.agent.call("hook.list", {}).value,
                         [{"name": "veto.run", "event": "pre_tool", "pattern": "echo", "tool": "veto.run"}])
        history = self.agent.call("hook.history", {"limit": 5})
        self.assertEqual(history.value[0]["outcome"]["reason"], "vetoed echo")
        self.assertEqual(history.value[0]["hook"], "veto.run")
        self.assertEqual(self.agent.call("hook.remove", {"name": "veto.run"}).value, 1)
        self.assertTrue(self.agent.call("echo", {"text": "x"}).ok)

    def test_invalid_event(self):
        with self.assertRaises(ToolError):
            self.agent.hooks.add("pre_dance", "*")


class MemoryTest(unittest.TestCase):
    def test_write_read_update(self):
        memory = Memory()
        memory.write("lang", "python", tags="stack", kind="preference")
        entry = memory.read("lang")
        self.assertEqual(entry["value"], "python")
        memory.write("lang", "python and sqlite", kind="preference")
        self.assertEqual(len(memory.entries), 1)
        self.assertEqual(memory.read("lang")["hits"], 2)

    def test_missing_key_is_a_tool_error(self):
        with self.assertRaises(ToolError):
            Memory().read("ghost")

    def test_search_ranks_by_hits_and_recency(self):
        clock = [1000.0]
        memory = Memory(clock=lambda: clock[0])
        memory.write("a", "the loop runs tools")
        clock[0] += 10
        memory.write("b", "tools include the loop and tools")
        clock[0] += 10
        memory.write("c", "unrelated")
        found = memory.search("tools loop")
        self.assertEqual([e["key"] for e in found], ["b", "a"])
        self.assertEqual(memory.search("nothing"), [])

    def test_persistence_across_instances(self):
        path = workspace() / "memory.json"
        Memory(path).write("kept", "yes", kind="fact")
        self.assertEqual(Memory(path).read("kept")["value"], "yes")
        self.assertEqual(Memory(workspace() / "broken.json").entries, [])
        path.write_text("not json", encoding="utf-8")
        self.assertEqual(Memory(path).entries, [])

    def test_forget_and_notes(self):
        memory = Memory()
        memory.write("x", "1")
        memory.write("y", "2")
        self.assertEqual(memory.forget("x"), {"forgotten": 1, "left": 1})
        self.assertIn("y", memory.notes())

    def test_memory_through_the_agent_funnel(self):
        agent = make_agent()
        self.assertTrue(agent.call("memory.write", {"key": "k", "value": "v"}).ok)
        self.assertEqual(agent.call("memory.search", {"query": "k"}).value[0]["value"], "v")


class ContextTest(unittest.TestCase):
    def test_push_and_stats(self):
        context = Context(budget_tokens=8)
        context.reset("system prompt")
        context.push("user", "hello there this is a goal")
        stats = context.stats()
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["tokens"], 9)
        self.assertTrue(stats["over"])
        self.assertEqual(stats["used_pct"], 112.5)

    def test_compact_folds_only_old_turns(self):
        context = Context(keep_last=2)
        for index in range(6):
            context.push("tool", f"observation {index} " + "x" * 80, name=f"t{index}")
        outcome = context.compact()
        self.assertTrue(outcome["compacted"])
        self.assertEqual(outcome["folded"], 4)
        self.assertLess(outcome["after"], outcome["before"])
        self.assertEqual(context.messages[-1]["name"], "t5")
        self.assertEqual(context.compact()["compacted"], False)

    def test_summary_text_is_used_when_given(self):
        context = Context(keep_last=1)
        for index in range(4):
            context.push("tool", f"turn {index}")
        context.compact("the user wanted a report")
        self.assertIn("the user wanted a report", context.messages[0]["content"])

    def test_as_messages_carries_names(self):
        context = Context()
        context.reset("sys")
        context.push("tool", "out", name="fs.read")
        self.assertEqual(context.as_messages(), [{"role": "system", "content": "sys"},
                                                 {"role": "tool", "content": "out", "name": "fs.read"}])


class PromptsTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_rendered_prompt_is_layered(self):
        self.agent.call("prompt.set", {"layer": "goal", "text": "count the files"})
        rendered = self.agent.call("prompt.get", {"rendered": True}).value["rendered"]
        self.assertIn("harness agent", rendered.splitlines()[0])
        self.assertIn("# goal\ncount the files", rendered)
        self.assertIn("# guardrails", rendered)
        self.assertIn("# tools", rendered)
        self.assertIn(" echo <text>", rendered)

    def test_denied_tools_are_hidden_from_the_prompt(self):
        tools_block = self.agent.call("prompt.get", {"rendered": True}).value["rendered"].split("# tools\n", 1)[1]
        for hidden in ("fs.delete", "shell.run", "agent.run"):
            self.assertNotIn(hidden, tools_block)
        self.assertIn("echo", tools_block)

    def test_layers_are_addressable(self):
        bad = self.agent.call("prompt.set", {"layer": "vibes", "text": "x"})
        self.assertIn("unknown prompt layer", bad.error)
        self.assertEqual(self.agent.call("prompt.set", {"layer": "persona", "text": "Style: dry."}).value,
                         {"layer": "persona", "tokens": 2})
        self.agent.prompts.attach("tasks", lambda: "2 tasks open")
        rendered = self.agent.call("prompt.get", {"rendered": True}).value["rendered"]
        self.assertIn("# tasks\n2 tasks open", rendered)
        self.assertEqual(self.agent.call("prompt.get", {}).value["dynamic"], ["tasks"])

    def test_memory_and_skills_join_the_render(self):
        self.agent.call("memory.write", {"key": "pref", "value": "likes tables", "kind": "preference"})
        rendered = self.agent.call("prompt.get", {"rendered": True}).value["rendered"]
        self.assertIn("# memory", rendered)
        self.assertIn("likes tables", rendered)


class SkillsTest(unittest.TestCase):
    NOTE = """---
name: note-taking
description: capture and cite notes
when_to_use: the user wants notes written down
tools: fs.write, notes.summarise
---
Read first, then write one file per note.
"""

    def setUp(self):
        self.dir = workspace()
        (self.dir / "note-taking.md").write_text(self.NOTE, encoding="utf-8")
        (self.dir / "extra").mkdir()
        (self.dir / "extra" / "plain-skill.md").write_text("# plain\nno front matter here\n", encoding="utf-8")
        self.agent = Agent(root=workspace(), skills_dir=[self.dir], memory_path=None)

    def test_scan_reads_front_matter_and_nested_dirs(self):
        names = {s["name"] for s in self.agent.call("skills.list", {}).value}
        self.assertEqual(names, {"note-taking", "plain-skill"})
        pack = next(s for s in self.agent.skills.list() if s["name"] == "note-taking")
        self.assertEqual(pack["description"], "capture and cite notes")
        self.assertEqual(pack["when"], "the user wants notes written down")
        self.assertEqual(pack["tools"], ["fs.write", "notes.summarise"])

    def test_use_reports_missing_tools(self):
        used = self.agent.call("skills.use", {"name": "note-taking"})
        self.assertEqual(used.value, {"active": ["note-taking"], "requires": ["fs.write", "notes.summarise"],
                                      "missing_tools": ["notes.summarise"]})
        self.assertIn("Read first, then write one file per note.",
                      self.agent.call("prompt.get", {"rendered": True}).value["rendered"])
        self.assertEqual(self.agent.call("skills.stop", {"name": "note-taking"}).value, {"active": []})
        self.assertNotIn("Read first", self.agent.call("prompt.get", {"rendered": True}).value["rendered"])

    def test_unknown_skill_names_the_installed_ones(self):
        self.assertIn("no skill 'nope'", self.agent.call("skills.read", {"name": "nope"}).error)

    def test_scan_tool_picks_up_new_files(self):
        (self.dir / "late.md").write_text("name: late\nbody\n", encoding="utf-8")
        self.assertIn("late", self.agent.call("skills.scan", {}).value["found"])

    def test_skill_without_a_directory_is_empty(self):
        self.assertEqual(make_agent().skills.list(), [])


class TasksTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_dependency_chain_gates_readiness(self):
        first = self.agent.call("task.create", {"subject": "read the file"})
        second = self.agent.call("task.create", {"subject": "summarise it", "blocks": str(first.value["id"])})
        self.assertEqual([t["subject"] for t in self.agent.call("task.ready", {}).value], ["read the file"])
        self.agent.call("task.update", {"id": first.value["id"], "status": "completed"})
        self.assertEqual([t["subject"] for t in self.agent.call("task.ready", {}).value], ["summarise it"])
        self.assertEqual(second.value["history"], ["pending"])

    def test_status_is_validated(self):
        task = self.agent.call("task.create", {"subject": "a"}).value
        self.assertIn("bad task status", self.agent.call("task.update", {"id": task["id"], "status": "mostly-done"}).error)
        self.assertIn("no task 42", self.agent.call("task.get", {"id": 42}).error)

    def test_summary_counts(self):
        self.agent.call("task.create", {"subject": "a"})
        self.agent.call("task.create", {"subject": "b", "status": "completed"})
        self.assertEqual(self.agent.call("task.list", {}).value,
                         {"items": self.agent.tasks.items, "tasks": 2, "pending": 1, "completed": 1, "ready": ["a"]})


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.agent = Agent(root=workspace(), skills_dir=None, memory_path=None, clock=lambda: self.now[0])

    def test_add_validates_the_job(self):
        self.assertIn("must name a tool",
                      self.agent.call("schedule.add", {"name": "j", "tool": "", "every": 60}).error)
        self.assertIn("give either", self.agent.call("schedule.add", {"name": "j", "tool": "echo"}).error)

    def test_interval_job_fires_through_the_funnel(self):
        job = self.agent.call("schedule.add", {"name": "beat", "tool": "echo",
                                               "args": {"text": "tick"}, "every": 60}).value
        self.assertEqual(job["next"], 1060.0)
        self.assertEqual(self.agent.call("schedule.due", {}).value, {"due": []})
        self.now[0] = 1060.0
        self.assertEqual(self.agent.call("schedule.due", {}).value, {"due": ["beat"]})
        fired = self.agent.tick()
        self.assertEqual(fired[0]["result"], {"tool": "echo", "result": {"echo": "tick"}})
        self.assertEqual(self.agent.scheduler.jobs[0]["next"], 1120.0)
        self.assertEqual(self.agent.scheduler.jobs[0]["runs"], 1)
        scheduled = [c for c in self.agent.journal if c.source == "schedule"]
        self.assertEqual([c.tool for c in scheduled], ["echo"])

    def test_one_shot_job_fires_once(self):
        self.agent.call("schedule.add", {"name": "once", "tool": "echo", "args": {"text": "x"},
                                         "at": 900.0, "max_runs": 1})
        self.assertEqual(len(self.agent.tick()), 1)
        self.assertEqual(self.agent.tick(), [])
        self.assertEqual(self.agent.scheduler.due(), [])

    def test_readding_a_name_replaces_the_job(self):
        self.agent.call("schedule.add", {"name": "beat", "tool": "echo", "every": 60})
        self.agent.call("schedule.add", {"name": "beat", "tool": "echo", "every": 30})
        self.assertEqual(len(self.agent.scheduler.jobs), 1)
        self.assertEqual(self.agent.scheduler.jobs[0]["every"], 30.0)

    def test_a_blocked_job_reports_instead_of_raising(self):
        self.agent.call("schedule.add", {"name": "wipe", "tool": "fs.delete", "args": {"path": "x"}, "at": 1.0})
        fired = self.agent.tick()
        self.assertTrue(fired[0]["result"]["blocked"])
        self.assertIn("denied by guardrail", fired[0]["result"]["reason"])

    def test_remove(self):
        self.agent.call("schedule.add", {"name": "beat", "tool": "echo", "every": 60})
        self.assertEqual(self.agent.call("schedule.remove", {"name": "beat"}).value, {"removed": 1, "left": 0})
        self.assertEqual(self.agent.call("schedule.remove", {"name": "beat"}).value, {"removed": 0, "left": 0})

    def test_the_loop_ticks_due_jobs(self):
        self.agent.call("schedule.add", {"name": "beat", "tool": "echo", "args": {"text": "from the job"}, "at": 1.0})
        self.agent.script([{"thought": "work", "call": {"tool": "echo", "args": {"text": "loop"}}},
                           {"thought": "done", "final": "ok"}])
        run = self.agent.run("drive the loop")
        self.assertTrue(run.ok)
        self.assertIn("beat", [f["job"] for f in self.agent.scheduler.fired])


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_scripted_queue_is_deterministic(self):
        self.agent.script([{"thought": "one", "final": "a"}, {"thought": "two", "final": "b"}], name="s")
        self.assertEqual(self.agent.call("model.scripted", {"script": "s"}).value["final"], "a")
        self.assertEqual(self.agent.call("model.scripted", {"script": "s"}).value["final"], "b")
        ran_out = self.agent.call("model.scripted", {"script": "s", "goal": "keep going"}).value
        self.assertIn("ran out after 2 turns", ran_out["final"])
        self.assertEqual(ran_out["thought"], "script exhausted")

    def test_named_scripts_do_not_share_a_queue(self):
        self.agent.script([{"final": "from a"}], name="a")
        self.agent.script([{"final": "from b"}], name="b")
        self.assertEqual(self.agent.call("model.scripted", {"script": "b"}).value["final"], "from b")

    def test_set_script_tool_takes_json_and_appends(self):
        self.agent.call("model.set_script", {"turns": '[{"final": "one"}]'})
        self.assertEqual(self.agent.call("model.set_script", {"turns": [{"final": "two"}], "append": True}).value,
                         {"script": "default", "turns": 2})
        self.assertEqual(self.agent.call("model.scripted", {}).value["final"], "one")

    def test_action_from_tool_calls_and_json(self):
        models = A.Models()
        called = models._action_from("", [{"function": {"name": "echo", "arguments": '{"text": "hi"}'}}])
        self.assertEqual(called, {"thought": "echo", "call": {"tool": "echo", "args": {"text": "hi"}}})
        aliased = models._action_from('{"thought": "t", "action": {"tool": "echo", "args": {}}}')
        self.assertEqual(aliased["call"]["tool"], "echo")
        with self.assertRaises(ToolError):
            models._action_from("I think we should stop")

    def test_chatml_render(self):
        self.assertEqual(A.render_chatml([{"role": "system", "content": "hi"}]), "system: hi\n\nassistant:")

    def test_network_models_are_guardrailed_by_default(self):
        for name in ("model.ollama", "model.openai", "model.llamacpp"):
            record = self.agent.call(name, {})
            self.assertTrue(record.blocked, name)
            self.assertIn("network tools are off", record.reason)

    def test_an_unreachable_endpoint_is_reported_not_raised(self):
        agent = Agent(root=workspace(), skills_dir=None, memory_path=None, allow_network=True,
                      env={"AGENT_TIMEOUT": "2", "OLLAMA_HOST": "http://127.0.0.1:9"})
        record = agent.call("model.ollama", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertFalse(record.ok)
        self.assertIn("unreachable", record.error)

    def test_model_log_records_offers(self):
        self.agent.call("model.scripted", {"tools": ["echo", "fs.read"]})
        self.assertEqual(self.agent.call("model.log", {}).value[-1]["tools"], 2)

    def test_openai_spec_is_a_json_schema(self):
        spec = self.agent.registry.get("fs.write").openai_spec()
        self.assertEqual(spec["function"]["name"], "fs.write")
        self.assertEqual(spec["function"]["parameters"]["required"], ["path"])
        self.assertEqual(spec["function"]["parameters"]["properties"]["append"]["type"], "boolean")


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.agent = make_agent()

    def test_run_follows_the_script(self):
        self.agent.script([{"thought": "first", "call": {"tool": "echo", "args": {"text": "a"}}},
                           {"thought": "second", "call": {"tool": "echo", "args": {"text": "b"}}},
                           {"thought": "answer", "final": "two echoes"}])
        run = self.agent.run("echo twice")
        self.assertEqual((run.status, run.answer, run.trail), ("done", "two echoes", "echo -> echo"))
        self.assertEqual(len(run.turns), 3)
        self.assertIn(str(run), "[model.scripted] done: two echoes")

    def test_run_stops_at_the_step_limit(self):
        self.agent.script([{"call": {"tool": "echo", "args": {"text": "a"}}}] * 4)
        run = self.agent.run("never answers", max_steps=2)
        self.assertEqual(run.status, "steps-exhausted")
        self.assertEqual(len(run.turns), 2)

    def test_a_blocked_model_fails_the_run(self):
        self.agent.call("policy.set", {"deny": ["model.scripted"]})
        run = self.agent.run("no brain")
        self.assertEqual(run.status, "model-failed")
        self.assertIn("denied by guardrail", run.error)
        self.assertEqual(run.turns, [])

    def test_side_channels_are_filed(self):
        self.agent.script([{"thought": "note it", "final": "done",
                            "memory": {"key": "who", "value": "the harness", "kind": "fact"},
                            "task": {"subject": "check later"}}])
        run = self.agent.run("remember and schedule")
        self.assertEqual((run.memory_writes, run.tasks_created), (1, 1))
        self.assertEqual(self.agent.call("memory.read", {"key": "who"}).value["value"], "the harness")
        self.assertEqual(self.agent.call("task.list", {}).value["tasks"], 1)

    def test_a_block_comes_back_as_an_observation(self):
        self.agent.script([{"thought": "wipe", "call": {"tool": "fs.delete", "args": {"path": "notes.md"}}},
                           {"thought": "give up", "final": "delete was denied"}])
        run = self.agent.run("delete the notes")
        self.assertEqual(run.blocked[0]["tool"], "fs.delete")
        tool_turns = [m for m in self.agent.context.messages if m["role"] == "tool" and m["name"] == "fs.delete"]
        self.assertIn('"blocked": true', tool_turns[0]["content"])
        self.assertIn("denied by guardrail", tool_turns[0]["content"])

    def test_an_action_without_call_or_final_is_reported_to_the_model(self):
        self.agent.script([{"thought": "empty"}, {"thought": "answer", "final": "recovered"}])
        run = self.agent.run("be vague")
        self.assertTrue(run.ok)
        loop_turns = [m for m in self.agent.context.messages if m["name"] == "loop"]
        self.assertIn("neither call nor final", loop_turns[0]["content"])

    def test_the_window_compacts_itself_during_a_run(self):
        agent = Agent(root=workspace(), skills_dir=None, memory_path=None, budget_tokens=900, keep_last=2)
        agent.script([{"call": {"tool": "echo", "args": {"text": "x" * 800}}}] * 3 + [{"final": "done"}])
        run = agent.run("shout a lot")
        self.assertTrue(run.ok)
        self.assertGreaterEqual(run.compactions, 1)
        self.assertEqual(agent.context.messages[0]["name"], "compact")
        self.assertIn("summary of", agent.context.messages[0]["content"])

    def test_grants_are_one_shot_by_default(self):
        self.agent.grant("fs.write")
        self.agent.script([{"call": {"tool": "fs.write", "args": {"path": "a.txt", "content": "1"}}},
                           {"call": {"tool": "fs.write", "args": {"path": "b.txt", "content": "2"}}},
                           {"final": "done"}])
        run = self.agent.run("write twice")
        self.assertEqual([t.call.blocked for t in run.turns[:2]], [False, True])
        self.assertIn("needs approval", run.blocked[0]["reason"])

    def test_a_repeating_grant_survives_across_runs(self):
        self.agent.grant("fs.write", repeats=True)
        for _ in range(2):
            self.agent.script([{"call": {"tool": "fs.write", "args": {"path": "a.txt", "content": "1"}}},
                               {"final": "written"}])
            run = self.agent.run("write")
            self.assertTrue(run.ok)
            self.assertEqual(run.blocked, [])

    def test_step_drives_one_cycle_at_a_time(self):
        self.agent.script([{"thought": "look", "call": {"tool": "echo", "args": {"text": "hi"}}},
                           {"thought": "answer", "final": "saw hi"}])
        first = self.agent.step("say hi")
        self.assertEqual(first["call"]["tool"], "echo")
        self.assertEqual([c.source for c in self.agent.journal[-2:]], ["model", "step"])
        second = self.agent.step()
        self.assertEqual(second["final"], "saw hi")
        self.assertEqual(self.agent.context.messages[0]["content"], "say hi")

    def test_step_reports_a_blocked_model(self):
        self.agent.call("policy.set", {"deny": ["model.scripted"]})
        self.assertEqual(self.agent.step("goal")["reason"], "denied by guardrail rule 'model.scripted'")

    def test_tool_call_budget_applies_to_mutations(self):
        self.agent.call("policy.set", {"max_tool_calls": 1})
        self.agent.policy.start_run()
        self.assertTrue(self.agent.call("memory.write", {"key": "a", "value": "1"}).ok)
        blocked = self.agent.call("memory.write", {"key": "b", "value": "2"})
        self.assertIn("tool-call budget 1 already spent", blocked.reason)
        self.assertTrue(self.agent.call("memory.search", {"query": "a"}).ok)

    def test_meta_tool_nests_a_whole_run(self):
        inner = make_agent()
        inner.script([{"call": {"tool": "echo", "args": {"text": "from inner"}}}, {"final": "inner done"}])
        self.agent.registry.add(Tool("agent.inner", "hand a goal to a nested agent", args={"goal": str},
                                     required=("goal",), kind="tool", mutating=True,
                                     handler=lambda goal: inner.run(goal).answer))
        self.agent.script([{"call": {"tool": "agent.inner", "args": {"goal": "do it"}}},
                           {"final": "nested answer relayed"}])
        run = self.agent.run("delegate")
        self.assertEqual(run.answer, "nested answer relayed")
        self.assertEqual([c.source for c in inner.journal if c.tool == "echo"], ["loop"])

    def test_the_loop_hooks_see_plan_and_stop(self):
        seen = []
        self.agent.hook("plan", "*", call=lambda payload: seen.append(("plan", payload["args"]["steps"])))
        self.agent.hook("stop", "*", call=lambda payload: seen.append(("stop", payload["tool"])))
        self.agent.script([{"final": "ok"}])
        self.agent.run("tiny")
        self.assertEqual(seen, [("plan", 12), ("stop", "done")])


class DiscoverTest(unittest.TestCase):
    counter = iter(range(1, 1000))

    def plugin(self, body: str) -> str:
        return f"from agent import tool\n\n{body}\n"

    def discover(self, directory, **kw):
        with mock.patch.object(A.importlib.metadata, "entry_points", lambda group=None: []):
            return A.discover([directory], registry=kw.pop("registry", Registry()), group=None, **kw)

    def test_a_plugin_pack_loads_into_a_fresh_agent(self):
        n = next(self.counter)
        directory = workspace()
        (directory / f"pack{n}.py").write_text(self.plugin(f"""
@tool("pack{n}.shout", "shout the text back", args={{"text": str}}, required=("text",))
def shout(text):
    return {{"shouted": text.upper()}}
"""), encoding="utf-8")
        agent = Agent(root=workspace(), tools_dir=directory, skills_dir=None, memory_path=None)
        self.assertIn(f"pack{n}.shout", agent.plugins["loaded"])
        self.assertTrue(agent.call(f"pack{n}.shout", {"text": "hi"}).ok)
        names = [t["name"] for t in agent.call("tools.list", {"kind": "tool"}).value["tools"]]
        self.assertIn(f"pack{n}.shout", names)

    def test_declared_schema_survives_the_plugin_boundary(self):
        n = next(self.counter)
        directory = workspace()
        (directory / f"typed{n}.py").write_text(self.plugin(f"""
@tool("typed{n}.add", "add b to a", args={{"a": int, "b": (int, "optional addend")}}, required=("a",))
def add(a, b=1):
    return {{"sum": a + b}}
"""), encoding="utf-8")
        registry = Registry()
        self.assertEqual(self.discover(directory, registry=registry)["loaded"], [f"typed{n}.add"])
        brief = registry.get(f"typed{n}.add").brief()
        self.assertEqual(brief["args"], {"a": "int", "b": "int"})
        self.assertEqual(brief["required"], ["a"])
        self.assertEqual(registry.get(f"typed{n}.add").call({"a": "2", "b": "3"}), {"sum": 5})

    def test_a_broken_plugin_does_not_stop_the_rest(self):
        n = next(self.counter)
        directory = workspace()
        (directory / f"good{n}.py").write_text(self.plugin(f'@tool("good{n}.ok")\ndef ok():\n    """fine"""'),
                                               encoding="utf-8")
        (directory / f"bad{n}.py").write_text("raise ValueError('this plugin is broken')\n", encoding="utf-8")
        (directory / f"_private{n}.py").write_text("import os\n", encoding="utf-8")
        found = self.discover(directory)
        self.assertEqual(found["modules"], [f"good{n}"])
        self.assertIn("ValueError: this plugin is broken", found["errors"][f"bad{n}"])
        self.assertEqual(found["loaded"], [f"good{n}.ok"])

    def test_a_missing_directory_is_quiet(self):
        self.assertEqual(self.discover(workspace() / "nope")["loaded"], [])

    def test_entry_points_are_adopted(self):
        class FakeEntryPoint:
            def __init__(self, name, obj):
                self.name, self._obj = name, obj

            def load(self):
                return self._obj

        class BrokenEntryPoint(FakeEntryPoint):
            def load(self):
                raise RuntimeError("this entry point is broken")

        def tagged(text):
            return {"tagged": text}

        tagged.tool_name = "tagged.by.attr"

        def by_name(text):
            return {"said": text}

        eps = [FakeEntryPoint("pack", {"ep.hello": lambda text: {"hello": text}}),
               FakeEntryPoint("listing", [by_name]),
               FakeEntryPoint("attr", tagged),
               BrokenEntryPoint("broken", None)]
        registry = Registry()

        def fake(group=None):
            return eps if group == A.ENTRY_POINT_GROUP else []

        with mock.patch.object(A.importlib.metadata, "entry_points", fake):
            found = A.discover([workspace() / "absent"], registry=registry)
        self.assertEqual(sorted(found["loaded"]), ["by_name", "ep.hello", "tagged.by.attr"])
        self.assertIn("this entry point is broken", found["errors"]["broken"])
        self.assertEqual(registry.get("tagged.by.attr").call({"text": "x"}), {"tagged": "x"})
        self.assertEqual(registry.get("ep.hello").call({"text": "x"}), {"hello": "x"})


class CliTest(unittest.TestCase):
    def invoke(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = A.main(list(argv))
        return code, out.getvalue()

    def test_demo_shows_the_guardrails_biting(self):
        code, out = self.invoke("demo")
        self.assertEqual(code, 0)
        self.assertIn("status: done", out)
        self.assertIn("BLOCK fs.read", out)
        self.assertIn("BLOCK fs.delete", out)
        self.assertIn("note-length", out)

    def test_run_command_echoes(self):
        code, out = self.invoke("run", "hello")
        self.assertEqual(code, 0)
        self.assertIn("echoed", out)

    def test_tools_and_panel_commands(self):
        code, out = self.invoke("tools")
        self.assertEqual(code, 0)
        self.assertIn("model.scripted", out)
        self.assertRegex(out, r"\d+ tools, kinds=")
        code, out = self.invoke("panel")
        self.assertEqual(code, 0)
        self.assertGreater(json.loads(out)["tools"], 40)
        code, out = self.invoke("skills")
        self.assertEqual(code, 0)
        self.assertIsInstance(json.loads(out), list)

    def test_an_unknown_command_prints_the_help(self):
        code, out = self.invoke("--help")
        self.assertEqual(code, 2)
        self.assertIn("python agent.py", out)


class SingleFileTest(unittest.TestCase):
    SOURCE = (Path(__file__).parent / "agent.py").read_text(encoding="utf-8")

    def test_only_stdlib_is_imported(self):
        imported = {m.split(".")[0] for m in re.findall(r"^(?:import|from)\s+([\w.]+)", self.SOURCE, re.M)}
        self.assertEqual(imported - set(sys.stdlib_module_names), set())

    def test_public_surface_lives_in_that_one_file(self):
        for name in ("class Agent", "class Tool", "class Registry", "class Policy", "class Hooks",
                     "class Memory", "class Context", "class Prompts", "class Skills", "class Tasks",
                     "class Scheduler", "class Models", "def tool", "def discover", "def main"):
            self.assertIn(name, self.SOURCE)


class ComponentSurfaceTest(unittest.TestCase):
    """Every subsystem must be reachable as a tool - in all three implementations."""

    KINDS = ("skill", "tool", "memory", "context", "hook", "policy", "prompt", "task", "schedule", "model", "loop")

    def test_each_component_registers_its_own_tools(self):
        kinds = make_agent().panel()["kinds"]
        for kind in self.KINDS:
            self.assertGreaterEqual(kinds.get(kind, 0), 2, kind)

    def test_every_component_kind_is_listable(self):
        agent = make_agent()
        for kind in self.KINDS:
            listed = agent.call("tools.list", {"kind": kind}).value
            self.assertEqual(listed["count"], len(agent.registry))
            self.assertEqual(len(listed["tools"]), agent.panel()["kinds"][kind], kind)

    def test_tool_names_are_shared_by_the_three_implementations(self):
        names = make_agent().registry.names()
        self.assertIn("agent.run", names)
        self.assertIn("model.set_script", names)
        self.assertIn("echo", names)
        for twin in ("agent.js", "agent.html"):
            path = Path(__file__).parent / twin
            if not path.exists():
                continue
            source = path.read_text(encoding="utf-8")
            for name in ("agent.run", "model.set_script", "fs.write", "memory.write", "hook.add",
                         "policy.check", "task.create", "schedule.add", "context.compact", "skills.use"):
                self.assertIn(name, source, f"{twin} is missing {name}")

    def test_the_node_twin_registers_exactly_the_same_tools(self):
        try:
            proc = subprocess.run(
                ["node", "-e", "const {Agent} = require('./agent.js');"
                               "console.log(JSON.stringify(new Agent({root: process.cwd(),"
                               "skills_dir: null, memory_path: null}).registry.names()))"],
                cwd=str(Path(__file__).parent), capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            self.skipTest("node is not installed")
        if proc.returncode:
            self.skipTest(f"node could not load agent.js: {proc.stderr.strip()[:200]}")
        self.assertEqual(sorted(json.loads(proc.stdout.strip())), sorted(make_agent().registry.names()))

    def test_the_browser_twin_adds_only_the_manual_model(self):
        source = (Path(__file__).parent / "agent.html").read_text(encoding="utf-8")
        # only the engine section registers tools; the panel and its presets do not
        engine = source[source.index("// -- engine"):source.index("const PRESETS")]
        found = set(re.findall(r"new Tool\(\s*\{\s*name: '([a-z_]+(?:\.[a-z_]+)*)'", engine))
        expected = set(make_agent().registry.names()) | {"model.manual"}
        self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
