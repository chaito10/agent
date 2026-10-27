"""agent.py — a pluggable agent harness where every subsystem is a tool.

Single file, stdlib only. Nine components — skills, tools, memory, context,
hooks, control/guardrails, orchestration loop, prompts, tasks/scheduler — each
register their own tools into one flat registry, so the loop can drive (and a
hook can veto) any of them by name. Model endpoints are tools too:
`model.scripted`, `model.ollama`, `model.openai`, `model.llamacpp`.
Built-in tools are ordinary plugs too: `fs.read/write/edit/list/exists/delete`,
`shell.run` (argv, no shell interpolation), `http.get`, `browser.open`.

    from agent import Agent

    agent = Agent()                       # offline scripted model by default
    agent.script([
        {"thought": "look", "call": {"tool": "fs.read", "args": {"path": "notes.md"}}},
        {"thought": "answered", "final": "two notes"},
    ])
    run = agent.run("what is in the notes?")
    print(run.status, run.trail)

Every turn a model must answer with one JSON object:

    {"thought": "...", "call": {"tool": "<name>", "args": {...}}}   # act
    {"thought": "...", "final": "..."}                              # stop

Either form may also carry "memory": {"key": ..., "value": ...} or
"task": {"subject": ...}; the loop files them, so a model can extend harness
state without a second tool call.

Plugging in your own:

    from agent import tool, discover

    @tool("csv.rows", "read a csv into rows", args={"path": "the file"}, required=("path",))
    def rows(path):
        ...

    discover()          # ./tools/*.py and installed entry points, group "agent.tools"

Nothing executes outside one funnel: registry -> pre_tool hooks -> policy -> handler
-> post_tool hooks -> context. The guardrails (Policy) are deny-first name patterns, a
filesystem sandbox around `root`, an approval list for destructive tools, forbidden
shell command patterns, per-run write/delete/call/wall-clock budgets, network off by
default, and a dry-run that refuses every mutating tool.

CLI:  python agent.py demo | tools | skills | panel | examples [name] | run "<goal>" [--model model.ollama]
"""

import fnmatch
import html
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_TOOLS_DIR = "tools"
ENTRY_POINT_GROUP = "agent.tools"
MODEL_GROUP = "agent.models"
PATH_KEYS = ("path", "file", "dir", "from", "to", "cwd", "root")
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
TASK_STATUSES = ("pending", "in_progress", "completed", "blocked", "deleted")
HOOK_EVENTS = ("plan", "pre_tool", "post_tool", "tool_error", "stop")


def estimate_tokens(text: Any) -> int:
    return max(1, len(str(text)) // 4)


def resolve(path: Any, root: Path) -> Path:
    candidate = Path(str(path)).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def coerce(value: Any, typ: type) -> Any:
    """Tool args arrive as JSON from a model; fit them to the declared type."""
    if value is None or typ in (Any, None):
        return value
    if typ is not bool and isinstance(value, typ):
        return value
    if typ is str:
        return value if isinstance(value, str) else json.dumps(value, default=str)
    if typ is bool:
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    if typ is int:
        return int(float(value))
    if typ is float:
        return float(value)
    if typ is list:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                return [p.strip() for p in value.split(",") if p.strip()]
            return parsed if isinstance(parsed, list) else [parsed]
        return list(value)
    if typ is dict:
        if not isinstance(value, str):
            raise ValueError(f"cannot use {type(value).__name__} as an object")
        parsed = json.loads(value) if value.strip() else {}
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return parsed
    return typ(value)


def error_text(exc: BaseException) -> str:
    """KeyError str() quotes its argument, which would swallow the harness message."""
    if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def first_json(text: Any) -> dict | None:
    """Pull the first balanced {...} object out of a model reply."""
    raw = str(text)
    start = raw.find("{")
    while start != -1:
        depth, in_string, escape = 0, False, False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start:index + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
        start = raw.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# tools — the one interface everything speaks
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """A tool ran and failed; the message goes back into the transcript."""


class Blocked(Exception):
    """A hook or guardrail refused the call."""


class UnknownTool(KeyError):
    pass


class SandboxError(Exception):
    """Path or budget violation detected by Policy."""


@dataclass(slots=True)
class Tool:
    name: str
    desc: str
    handler: Callable
    args: dict = field(default_factory=dict)
    required: tuple = ()
    kind: str = "tool"  # owning component: tool|skill|memory|context|hook|policy|prompt|loop|task|schedule|model
    mutating: bool = False
    network: bool = False
    danger: bool = False
    deletes: bool = False  # charged against max_deletes rather than the write budget
    plugin: str = "core"

    @staticmethod
    def arg_type(hint: Any) -> type:
        if isinstance(hint, tuple):
            return hint[0] if isinstance(hint[0], type) else str
        return hint if isinstance(hint, type) else str

    def schema(self) -> dict:
        props = {}
        for key, hint in self.args.items():
            if isinstance(hint, str):
                description = hint
            elif isinstance(hint, tuple):
                description = hint[1] if len(hint) > 1 else ""
            else:
                description = ""
            props[key] = {"type": JSON_TYPES.get(self.arg_type(hint), "string"),
                          "description": description or self.desc}
        return {"type": "object", "properties": props, "required": list(self.required)}

    def openai_spec(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.desc,
                                                 "parameters": self.schema()}}

    def call(self, args: dict) -> Any:
        bound = {}
        for key, value in (args or {}).items():
            if key not in self.args:
                raise ToolError(f"{self.name} got unexpected arg {key!r}; it takes {sorted(self.args) or 'nothing'}")
            try:
                bound[key] = coerce(value, self.arg_type(self.args[key]))
            except (ValueError, TypeError) as exc:
                raise ToolError(f"{self.name} arg {key!r}: {exc}") from None
        missing = [key for key in self.required if key not in bound]
        if missing:
            raise ToolError(f"{self.name} is missing required arg(s): {', '.join(missing)}")
        return self.handler(**bound)

    def brief(self) -> dict:
        return {"name": self.name, "kind": self.kind, "plugin": self.plugin, "desc": self.desc,
                "args": {k: (v if isinstance(v, str) else (v[0].__name__ if isinstance(v, tuple) else v.__name__))
                         for k, v in self.args.items()},
                "required": list(self.required), "mutating": self.mutating,
                "network": self.network, "danger": self.danger, "deletes": self.deletes}


class Registry:
    """Flat name -> Tool. Components register here; the loop dispatches only from here."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add(self, spec: Tool) -> Tool:
        self.tools[spec.name] = spec
        return spec

    def remove(self, name: str) -> bool:
        return self.tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        try:
            return self.tools[name]
        except KeyError:
            raise UnknownTool(f"no tool {name!r}. {len(self.tools)} are registered; tools.list shows them") from None

    def has(self, name: str) -> bool:
        return name in self.tools

    def names(self, kind: str | None = None) -> list[str]:
        return sorted(n for n, t in self.tools.items() if kind is None or t.kind == kind)

    def brief(self, kind: str | None = None) -> list[dict]:
        return [self.tools[n].brief() for n in self.names(kind)]

    def __len__(self) -> int:
        return len(self.tools)

    def __bool__(self) -> bool:
        return True  # an empty registry is still a registry; never fall back silently


_default_registry = Registry()


def tool(name: str | None = None, desc: str = "", *, args: dict | None = None, required: tuple = (),
         kind: str = "tool", mutating: bool = False, network: bool = False, danger: bool = False,
         registry: Registry | None = None) -> Callable:
    """Decorator registering a function as a named, callable, pluggable tool."""

    def wrap(fn: Callable) -> Callable:
        summary = ((fn.__doc__ or "").strip().splitlines() or [""])[0]
        (registry or _default_registry).add(Tool(
            name=name or fn.__name__, desc=desc or summary or fn.__name__, handler=fn, args=args or {},
            required=required, kind=kind, mutating=mutating, network=network, danger=danger,
            plugin=getattr(fn, "__module__", "core")))
        return fn

    return wrap


def discover(paths: Any = None, *, group: str | None = ENTRY_POINT_GROUP, registry: Registry | None = None) -> dict:
    """Import plugin files from ./tools and installed entry points, collecting tool names."""
    target = registry or _default_registry
    if paths is None:
        paths = [DEFAULT_TOOLS_DIR]
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    loaded: list[str] = []
    modules: list[str] = []
    errors: dict[str, str] = {}

    def adopt(candidate: Any, fallback: str) -> None:
        if not callable(candidate):
            return
        key = next((n for n, t in target.tools.items() if t.handler is candidate),
                   getattr(candidate, "tool_name", None) or fallback)
        summary = ((candidate.__doc__ or "").strip().splitlines() or [""])[0]
        signature = inspect.signature(candidate)
        args: dict = {}
        required: list = []
        for parameter in signature.parameters.values():
            if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD:
                continue
            hint = parameter.annotation
            args[parameter.name] = hint if isinstance(hint, type) and hint is not inspect.Parameter.empty else str
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter.name)
        target.add(Tool(name=key, desc=summary or key, handler=candidate, args=args, required=tuple(required)))
        loaded.append(key)

    for raw in paths:
        root = Path(str(raw)).expanduser()
        if not root.is_dir():
            continue
        directory = str(root.resolve())
        if directory not in sys.path:
            sys.path.append(directory)
        for file in sorted(root.glob("*.py")):
            if file.name.startswith("_"):
                continue
            before = set(target.tools)
            try:
                importlib.import_module(file.stem)
            except BaseException as exc:  # noqa: BLE001 - one broken plugin must not stop the rest
                errors[file.stem] = f"{type(exc).__name__}: {exc}"
                continue
            modules.append(file.stem)
            # a plain @tool plugin registers in the process registry; hand its tools to this one
            for name, spec in _default_registry.tools.items():
                if spec.plugin == file.stem:
                    target.tools.setdefault(name, spec)
            loaded += sorted(set(target.tools) - before)
    for ep_group in filter(None, (group, MODEL_GROUP)):
        for ep in importlib.metadata.entry_points(group=ep_group):
            try:
                obj = ep.load()
            except BaseException as exc:  # noqa: BLE001
                errors[ep.name] = f"{type(exc).__name__}: {exc}"
                continue
            if isinstance(obj, dict):
                for key, candidate in obj.items():
                    adopt(candidate, key)
            elif isinstance(obj, (list, tuple)):
                for candidate in obj:
                    adopt(candidate, getattr(candidate, "__name__", ep.name))
            else:
                adopt(obj, ep.name)
            modules.append(ep.name)
    return {"loaded": sorted(set(loaded)), "modules": modules, "errors": errors}


# ---------------------------------------------------------------------------
# control and guard rails
# ---------------------------------------------------------------------------

# The operator's own handles. A rule list must never be able to lock the person who
# typed it out of the guardrails, so calls from these sources skip the gate for these
# tools. Everything the model drives (loop, model, step, tools.call, schedule) is
# enforced in full, so an agent can never re-open its own network switch by name.
GUARDRAIL_TOOLS = ("policy.", "confirm.")
OPERATOR_SOURCES = ("panel", "cli", "operator")


@dataclass
class Policy:
    """Deny wins over allow, always."""

    root: Path = field(default_factory=Path.cwd)
    allow: list = field(default_factory=list)  # empty means "everything not denied"
    deny: list = field(default_factory=lambda: ["agent.run", "fs.delete", "shell.*"])
    confirm_required: list = field(default_factory=lambda: ["fs.write", "fs.edit", "shell.run"])
    forbidden_commands: list = field(default_factory=lambda: [
        "rm -rf /", "rm -rf ~", "mkfs", "dd if=", ":(){", "sudo", "shutdown", "reboot", "reg delete",
        "> /dev/", "|sh", "| powershell", "del /f /s /q", "format ", "del c:\\", "chmod -R 777"])
    extra_dirs: list = field(default_factory=list)
    max_steps: int = 12
    max_tool_calls: int = 60
    max_writes: int = 20
    max_deletes: int = 2
    max_seconds: float = 300.0
    allow_network: bool = False
    dry_run: bool = False
    granted: dict = field(default_factory=dict)
    calls: int = 0
    writes: int = 0
    deletes: int = 0
    started_at: float = 0.0
    blocks: list = field(default_factory=list)

    def start_run(self) -> None:
        self.calls = self.writes = self.deletes = 0
        self.started_at = time.time()

    def grant(self, name: str, repeats: bool = False) -> None:
        """Operator approval for a confirm-required tool; it survives runs until revoked."""
        self.granted[name] = -1 if repeats else 0

    def revoke(self, name: str) -> bool:
        return self.granted.pop(name, None) is not None

    def block(self, name: str, reason: str) -> tuple[bool, str]:
        self.blocks.append({"tool": name, "reason": reason, "at": time.time()})
        return False, reason

    def check(self, spec: Tool, args: dict, source: str = "loop") -> tuple[bool, str]:
        name = spec.name
        if source in OPERATOR_SOURCES and name.startswith(GUARDRAIL_TOOLS):
            return True, ""
        for pattern in self.deny:
            if fnmatch.fnmatchcase(name, pattern):
                return self.block(name, f"denied by guardrail rule {pattern!r}")
        if self.allow and not any(fnmatch.fnmatchcase(name, pattern) for pattern in self.allow):
            return self.block(name, f"not in the allow list {self.allow}")
        if spec.network and not self.allow_network:
            return self.block(name, "network tools are off (set policy.allow_network=True)")
        command = str((args or {}).get("cmd", ""))
        if command:
            hit = next((pattern for pattern in self.forbidden_commands if pattern in command), None)
            if hit:
                return self.block(name, f"command pattern {hit!r} is forbidden")
        if self.dry_run and spec.mutating:
            return self.block(name, "dry-run refuses mutating tools")
        if name in self.confirm_required:
            left = self.granted.get(name)
            if left is None:
                return self.block(name, f"{name} needs approval: call confirm.grant(tool={name!r})")
            if left == 0:
                self.granted.pop(name, None)
            else:
                self.granted[name] = left - 1
        if spec.mutating and self.calls >= self.max_tool_calls:
            return self.block(name, f"tool-call budget {self.max_tool_calls} already spent")
        try:
            self._charge(spec, args or {})
        except SandboxError as exc:
            return self.block(name, str(exc))
        if self.started_at and self.max_seconds and time.time() - self.started_at > self.max_seconds:
            return self.block(name, f"wall-clock limit {self.max_seconds}s reached")
        self.calls += 1
        return True, ""

    def _charge(self, spec: Tool, args: dict) -> None:
        bases = [Path(str(self.root)), *[Path(str(d)) for d in self.extra_dirs]]
        for key in PATH_KEYS:
            if key in args and args[key] not in (None, ""):
                target = resolve(args[key], Path(str(self.root)))
                if not any(under(target, base) for base in bases):
                    raise SandboxError(f"arg {key}={args[key]!r} escapes the sandbox {self.root}")
        if not spec.mutating:
            return
        if spec.deletes:
            if self.deletes >= self.max_deletes:
                raise SandboxError(f"delete budget {self.max_deletes} already spent")
            self.deletes += 1
        else:
            if self.writes >= self.max_writes:
                raise SandboxError(f"write budget {self.max_writes} already spent")
            self.writes += 1

    def set(self, **changes) -> dict:
        for key, value in changes.items():
            if not hasattr(self, key):
                raise ToolError(f"no such policy knob {key!r}")
            setattr(self, key, Path(str(value)) if key == "root" else value)
        return self.summary()

    def summary(self) -> dict:
        return {"root": str(self.root), "allow": self.allow, "deny": self.deny,
                "confirm_required": self.confirm_required, "granted": dict(self.granted),
                "allow_network": self.allow_network, "dry_run": self.dry_run,
                "max_steps": self.max_steps, "max_tool_calls": self.max_tool_calls,
                "max_writes": self.max_writes, "max_deletes": self.max_deletes,
                "max_seconds": self.max_seconds,
                "used": {"calls": self.calls, "writes": self.writes, "deletes": self.deletes},
                "blocked": len(self.blocks)}

    def prompt_block(self) -> str:
        return (f"root={self.root} | network={'on' if self.allow_network else 'off'}"
                f" | dry_run={'on' if self.dry_run else 'off'}"
                f" | denied={', '.join(self.deny) or 'none'}"
                f" | needs approval={', '.join(self.confirm_required) or 'none'}"
                f" | left: {max(0, self.max_tool_calls - self.calls)} calls, "
                f"{max(0, self.max_writes - self.writes)} writes")


# ---------------------------------------------------------------------------
# hooks — an event bus whose handlers can also be tools
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Hook:
    name: str
    event: str
    pattern: str
    tool: str
    call: Callable


class Hooks:
    def __init__(self, registry: Registry | None = None):
        self.hooks: list[Hook] = []
        self.fired: list[dict] = []
        self.registry = registry or _default_registry

    def add(self, event: str, pattern: str = "*", call: Callable | None = None, tool: str = "", name: str = "") -> Hook:
        if event not in HOOK_EVENTS:
            raise ToolError(f"unknown hook event {event!r}; use {', '.join(HOOK_EVENTS)}")
        if tool and not self.registry.has(tool):
            raise ToolError(f"hook target tool {tool!r} is not registered")
        if call is None and not tool:
            raise ToolError("a hook needs either a callable or a registered tool")
        handler = call or (lambda payload, _t=tool: self.registry.get(_t).call({"payload": payload}))
        hook = Hook(name=name or tool or f"{event}:{pattern}", event=event, pattern=pattern, tool=tool, call=handler)
        self.hooks.append(hook)
        return hook

    def remove(self, name: str) -> int:
        before = len(self.hooks)
        self.hooks = [h for h in self.hooks if h.name != name]
        return before - len(self.hooks)

    def describe(self) -> list:
        return [{"name": h.name, "event": h.event, "pattern": h.pattern, "tool": h.tool} for h in self.hooks]

    def emit(self, event: str, payload: dict) -> dict:
        """Run matching hooks. pre_tool may block or rewrite args; others may only block."""
        verdict = {"block": False, "reason": "", "args": dict(payload.get("args") or {})}
        for hook in self.hooks:
            if hook.event != event:
                continue
            if not fnmatch.fnmatchcase(str(payload.get("tool") or ""), hook.pattern):
                continue
            try:
                outcome = hook.call(dict(payload))
            except Blocked as exc:
                outcome = {"block": True, "reason": str(exc)}
            except Exception as exc:  # noqa: BLE001 - a broken hook must not hang the agent
                outcome = {"block": True, "reason": f"hook {hook.name} raised {type(exc).__name__}: {exc}"}
            self.fired.append({"event": event, "hook": hook.name, "tool": payload.get("tool"),
                               "outcome": outcome if isinstance(outcome, dict) else {}, "at": time.time()})
            if not isinstance(outcome, dict):
                continue
            if outcome.get("block"):
                verdict["block"] = True
                verdict["reason"] = (verdict["reason"] + "; " if verdict["reason"] else "") + str(
                    outcome.get("reason") or f"blocked by {hook.name}")
                if event == "pre_tool":
                    return verdict
            elif event == "pre_tool" and isinstance(outcome.get("args"), dict):
                verdict["args"] = {**verdict["args"], **outcome["args"]}
        return verdict

    def register(self, reg: Registry) -> None:
        reg.add(Tool("hook.add", "attach a hook to an event; it can block a call or rewrite its args",
                     args={"event": (str, f"one of {','.join(HOOK_EVENTS)}"),
                           "pattern": (str, "fnmatch on the tool name"),
                           "tool": (str, "registered tool to run as the handler"), "name": str},
                     required=("event",), handler=self._t_add, kind="hook", mutating=True))
        reg.add(Tool("hook.list", "attached hooks", handler=self.describe, kind="hook"))
        reg.add(Tool("hook.remove", "detach hooks by name", args={"name": str}, required=("name",),
                     handler=self.remove, kind="hook", mutating=True))
        reg.add(Tool("hook.history", "recent firings with outcomes", args={"limit": (int, "rows")},
                     handler=lambda limit=20: self.fired[-int(limit):], kind="hook"))

    def _t_add(self, event: str, pattern: str = "*", tool: str = "", name: str = "") -> dict:
        hook = self.add(event, pattern, tool=tool, name=name)
        return {"attached": hook.name, "event": hook.event, "pattern": hook.pattern}


# ---------------------------------------------------------------------------
# memory — durable notes, file backed
# ---------------------------------------------------------------------------


class Memory:
    def __init__(self, path: Any = None, clock: Callable = time.time):
        self.path = Path(str(path)) if path else None
        self.clock = clock
        self.entries: list[dict] = []
        self.load()

    def load(self) -> int:
        if self.path and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = data if isinstance(data, list) else []
            except (ValueError, OSError):
                self.entries = []
        return len(self.entries)

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=2, default=str), encoding="utf-8")

    def write(self, key: str, value: Any = "", tags: str = "", kind: str = "note") -> dict:
        existing = next((e for e in self.entries if e["key"] == key), None)
        entry = {"key": key, "value": value, "tags": tags, "kind": kind, "at": self.clock(),
                 "hits": int(existing.get("hits", 0)) if existing else 0}
        if existing:
            existing.update(entry)
        else:
            self.entries.append(entry)
        self.save()
        return entry

    def read(self, key: str) -> dict:
        for entry in self.entries:
            if entry["key"] == key:
                entry["hits"] = int(entry.get("hits", 0)) + 1
                self.save()
                return entry
        raise ToolError(f"no memory entry {key!r}")

    def forget(self, key: str) -> dict:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["key"] != key]
        self.save()
        return {"forgotten": before - len(self.entries), "left": len(self.entries)}

    def search(self, query: str = "", limit: int = 8, kind: str = "") -> list:
        words = [w.lower() for w in re.split(r"\W+", str(query or "")) if w]
        now = max(1.0, float(self.clock()))
        scored = []
        for entry in self.entries:
            if kind and entry.get("kind") != kind:
                continue
            haystack = f"{entry['key']} {entry.get('tags', '')} {entry.get('kind', '')} {entry['value']}".lower()
            score = sum(haystack.count(word) for word in words) if words else 1
            if score:
                scored.append((score + 0.5 * int(entry.get("hits", 0)) + float(entry.get("at") or 0) / now, entry))
        scored.sort(key=lambda pair: -pair[0])
        return [entry for _, entry in scored[: int(limit)]]

    def notes(self, limit: int = 6) -> str:
        recent = sorted(self.entries, key=lambda e: -float(e.get("at") or 0))[: int(limit)]
        return "\n".join(f"- {e['key']} ({e.get('kind', 'note')}): {str(e['value'])[:200]}" for e in reversed(recent))

    def register(self, reg: Registry) -> None:
        reg.add(Tool("memory.write", "store a note under a key",
                     args={"key": str, "value": (str, "what to remember"), "tags": str,
                           "kind": (str, "note|preference|fact|warning|chunk")},
                     required=("key", "value"), handler=self.write, kind="memory", mutating=True))
        reg.add(Tool("memory.search", "find notes by keyword", args={"query": str, "limit": int, "kind": str},
                     handler=self.search, kind="memory"))
        reg.add(Tool("memory.read", "one note by key", args={"key": str}, required=("key",),
                     handler=self.read, kind="memory"))
        reg.add(Tool("memory.list", "every note",
                     handler=lambda: {"entries": self.entries, "count": len(self.entries)}, kind="memory"))
        reg.add(Tool("memory.forget", "drop a note", args={"key": str}, required=("key",),
                     handler=self.forget, kind="memory", mutating=True))


# ---------------------------------------------------------------------------
# context — the window the model actually sees
# ---------------------------------------------------------------------------


class Context:
    def __init__(self, budget_tokens: int = 4000, keep_last: int = 6):
        self.budget_tokens = int(budget_tokens)
        self.keep_last = int(keep_last)
        self.system = ""
        self.messages: list[dict] = []
        self.compactions = 0

    def reset(self, system: str = "") -> None:
        self.system = system
        self.messages = []

    def push(self, role: str, content: Any, name: str = "") -> dict:
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        message = {"role": role, "content": text, "name": name, "tokens": estimate_tokens(text)}
        self.messages.append(message)
        return message

    def stats(self) -> dict:
        used = sum(m["tokens"] for m in self.messages) + estimate_tokens(self.system)
        return {"messages": len(self.messages), "tokens": used, "budget": self.budget_tokens,
                "used_pct": round(100.0 * used / max(1, self.budget_tokens), 1),
                "over": used > self.budget_tokens, "compactions": self.compactions}

    def needs_compaction(self) -> bool:
        return self.stats()["over"]

    def as_messages(self) -> list:
        out = [{"role": "system", "content": self.system}] if self.system else []
        for message in self.messages:
            entry = {"role": message["role"], "content": message["content"]}
            if message.get("name"):
                entry["name"] = message["name"]
            out.append(entry)
        return out

    def compact(self, summary: str | None = None) -> dict:
        """Fold everything older than keep_last into one summary message."""
        before = self.stats()["tokens"]
        if len(self.messages) <= self.keep_last + 1:
            return {"compacted": False, "reason": "nothing old enough to fold", "tokens": before}
        old = self.messages[: len(self.messages) - self.keep_last]
        if summary is None:
            summary = " | ".join(f"{m.get('name') or m['role']}: {m['content'][:70]}" for m in old)
        folded = [{"role": "system", "content": f"[summary of {len(old)} earlier turns] {summary}",
                   "name": "compact", "tokens": estimate_tokens(summary)}]
        self.messages = folded + self.messages[len(old):]
        self.compactions += 1
        return {"compacted": True, "folded": len(old), "before": before, "after": self.stats()["tokens"]}

    def register(self, reg: Registry) -> None:
        reg.add(Tool("context.stats", "window usage: tokens, messages, budget", handler=self.stats, kind="context"))
        reg.add(Tool("context.read", "the transcript the model would see", args={"limit": int},
                     handler=lambda limit=30: self.messages[-int(limit):], kind="context"))
        reg.add(Tool("context.push", "add a message to the transcript",
                     args={"role": (str, "user|assistant|tool"), "content": str, "name": str},
                     required=("role", "content"), handler=self.push, kind="context", mutating=True))


# ---------------------------------------------------------------------------
# prompts — layered system prompt assembly
# ---------------------------------------------------------------------------

BASE_PROMPT = """You are the {name} harness agent, a loop that acts only through tools.

Reply with ONE JSON object and nothing else:
  {{"thought": "<one line>", "call": {{"tool": "<name>", "args": {{...}}}}}}
  {{"thought": "<one line>", "final": "<answer to the goal>"}}

Use only tools from the list. Read before you write. If a call is blocked by a
guardrail, do not fight it: route around it or explain the block in "final"."""

PERSONA_PROMPT = "Style: terse and concrete. Never invent a path or a tool name. Quote the output that decided something."


class Prompts:
    def __init__(self, layers: dict | None = None):
        self.layers: dict[str, str] = {"base": BASE_PROMPT, "persona": PERSONA_PROMPT, "goal": "",
                                       "guardrails": "", "skills": "", "memory": "", "tools": ""}
        self.dynamic: list[tuple[str, Callable]] = []
        if layers:
            self.layers.update(layers)

    def set(self, layer: str, text: str) -> dict:
        if layer not in self.layers:
            raise ToolError(f"unknown prompt layer {layer!r}; layers are {', '.join(self.layers)}")
        self.layers[layer] = text
        return {"layer": layer, "tokens": estimate_tokens(text)}

    def attach(self, layer: str, producer: Callable) -> dict:
        """A layer recomputed every turn, e.g. ("tasks", lambda: json.dumps(tasks.summary()))."""
        self.layers.setdefault(layer, "")
        self.dynamic.append((layer, producer))
        return {"layer": layer, "dynamic": True}

    def render(self, agent: "Agent") -> str:
        for layer, producer in self.dynamic:
            self.layers[layer] = str(producer() or "")
        self.layers["memory"] = agent.memory.notes()
        self.layers["skills"] = agent.skills.prompt_block()
        self.layers["guardrails"] = agent.policy.prompt_block()
        self.layers["tools"] = agent.tool_prompt()
        fixed = ("base", "persona")
        order = fixed + ("goal", "guardrails", "skills", "memory", "tools")
        order += tuple(k for k in self.layers if k not in order)
        parts = [self.layers["base"].format(name=agent.name)]
        for key in order[1:]:
            text = self.layers.get(key)
            if text:
                parts.append(text if key == "persona" else f"# {key}\n{text}")
        return "\n\n".join(p for p in parts if p.strip())

    def register(self, reg: Registry, agent: "Agent") -> None:
        def get(rendered: bool = False) -> dict:
            out: dict = {"layers": {k: (v[:400] + "..." if len(str(v)) > 400 else v) for k, v in self.layers.items()},
                         "dynamic": [layer for layer, _ in self.dynamic]}
            if rendered:
                out["rendered"] = self.render(agent)
                out["tokens"] = estimate_tokens(out["rendered"])
            return out

        reg.add(Tool("prompt.get", "the prompt layers, and the rendered system prompt on request",
                     args={"rendered": (bool, "include the assembled text")}, handler=get, kind="prompt"))
        reg.add(Tool("prompt.set", "replace one layer (base|persona|goal|guardrails|skills|memory|tools)",
                     args={"layer": str, "text": str}, required=("layer", "text"), handler=self.set,
                     kind="prompt", mutating=True))


# ---------------------------------------------------------------------------
# skills — instruction packs read from markdown
# ---------------------------------------------------------------------------

FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)


class Skills:
    def __init__(self, dirs: Any = ("skills",), registry: Registry | None = None):
        self.dirs = [Path(str(d)) for d in (dirs if isinstance(dirs, (list, tuple)) else [dirs])]
        self.registry = registry or _default_registry
        self.packs: dict[str, dict] = {}
        self.active: list[str] = []
        self.scan()

    def scan(self) -> list[str]:
        found = []
        for directory in self.dirs:
            if not directory.is_dir():
                continue
            for file in sorted(list(directory.glob("*.md")) + list(directory.glob("*/*.md"))):
                pack = self._parse(file)
                if pack:
                    self.packs[pack["name"]] = pack
                    found.append(pack["name"])
        return found

    @staticmethod
    def _parse(file: Path) -> dict:
        try:
            raw = file.read_text(encoding="utf-8")
        except OSError:
            return {}
        match = FRONT_MATTER.match(raw)
        meta, body = (match.group(1), match.group(2)) if match else ("", raw)
        fields = {}
        for line in meta.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip().strip("\"'")
        return {"name": fields.get("name") or file.stem, "description": fields.get("description", ""),
                "when": fields.get("when_to_use", fields.get("when", "")),
                "tools": [t.strip() for t in fields.get("tools", "").split(",") if t.strip()],
                "body": body.strip(), "file": str(file)}

    def list(self) -> list:
        return [{"name": p["name"], "description": p["description"], "when": p["when"], "tools": p["tools"],
                 "active": p["name"] in self.active} for p in self.packs.values()]

    def pack(self, name: str) -> dict:
        if name not in self.packs:
            raise ToolError(f"no skill {name!r}; {len(self.packs)} installed: {', '.join(self.packs) or 'none'}")
        return self.packs[name]

    def read(self, name: str) -> str:
        return self.pack(name)["body"]

    def use(self, name: str) -> dict:
        pack = self.pack(name)
        if name not in self.active:
            self.active.append(name)
        return {"active": list(self.active), "requires": pack["tools"],
                "missing_tools": [t for t in pack["tools"] if not self.registry.has(t)]}

    def stop(self, name: str) -> dict:
        if name in self.active:
            self.active.remove(name)
        return {"active": list(self.active)}

    def prompt_block(self) -> str:
        return "\n\n".join(f"## skill: {name}\n{self.packs[name]['body']}" for name in self.active if name in self.packs)

    def register(self, reg: Registry) -> None:
        reg.add(Tool("skills.list", "installed skill packs", handler=self.list, kind="skill"))
        reg.add(Tool("skills.read", "a skill's full instructions", args={"name": str}, required=("name",),
                     handler=self.read, kind="skill"))
        reg.add(Tool("skills.use", "activate a skill; its body joins the system prompt", args={"name": str},
                    required=("name",), handler=self.use, kind="skill", mutating=True))
        reg.add(Tool("skills.stop", "deactivate a skill", args={"name": str}, required=("name",),
                    handler=self.stop, kind="skill", mutating=True))
        reg.add(Tool("skills.scan", "re-read the skill directories", handler=lambda: {"found": self.scan()},
                     kind="skill", mutating=True))


# ---------------------------------------------------------------------------
# tasks and scheduler
# ---------------------------------------------------------------------------


class Tasks:
    def __init__(self, clock: Callable = time.time):
        self.items: list[dict] = []
        self.clock = clock
        self._next = 1

    def create(self, subject: str, description: str = "", status: str = "pending", blocks: str = "") -> dict:
        task = {"id": self._next, "subject": subject, "description": description, "status": status,
                "blocked_by": [int(b) for b in re.findall(r"\d+", blocks)], "at": self.clock(),
                "history": [status]}
        self._next += 1
        self.items.append(task)
        return task

    def get(self, id: int) -> dict:  # noqa: A002 - the tool arg really is called id
        for task in self.items:
            if task["id"] == int(id):
                return task
        raise ToolError(f"no task {id}")

    def update(self, id: int, status: str = "", subject: str = "", description: str = "") -> dict:  # noqa: A002
        task = self.get(id)
        if status:
            if status not in TASK_STATUSES:
                raise ToolError(f"bad task status {status!r}; use {', '.join(TASK_STATUSES)}")
            task["status"] = status
            task["history"].append(status)
        if subject:
            task["subject"] = subject
        if description:
            task["description"] = description
        return task

    def ready(self) -> list:
        done = {t["id"] for t in self.items if t["status"] == "completed"}
        return [t for t in self.items if t["status"] == "pending" and not set(t["blocked_by"]) - done]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for task in self.items:
            counts[task["status"]] = counts.get(task["status"], 0) + 1
        return {"tasks": len(self.items), **counts, "ready": [t["subject"] for t in self.ready()]}

    def register(self, reg: Registry) -> None:
        reg.add(Tool("task.create", "track a unit of work",
                     args={"subject": str, "description": str, "status": str,
                           "blocks": (str, "comma separated task ids this waits on")},
                     required=("subject",), handler=self.create, kind="task", mutating=True))
        reg.add(Tool("task.list", "every task plus status counts",
                     handler=lambda: {"items": self.items, **self.summary()}, kind="task"))
        reg.add(Tool("task.get", "one task", args={"id": int}, required=("id",), handler=self.get, kind="task"))
        reg.add(Tool("task.update", "change a task's status or text",
                     args={"id": int, "status": str, "subject": str, "description": str},
                     required=("id",), handler=self.update, kind="task", mutating=True))
        reg.add(Tool("task.ready", "tasks with no unfinished blockers", handler=self.ready, kind="task"))


class Scheduler:
    """Interval/at jobs. A job is a tool call through the same funnel — nothing else."""

    def __init__(self, clock: Callable = time.time):
        self.clock = clock
        self.jobs: list[dict] = []
        self.fired: list[dict] = []

    def add(self, name: str, tool: str = "", args: dict | None = None, every: float = 0, at: float = 0,
            max_runs: int = 0) -> dict:
        if not tool:
            raise ToolError("a scheduled job must name a tool to call")
        if not every and not at:
            raise ToolError("give either every=<seconds> or at=<epoch>")
        job = {"name": name, "tool": tool, "args": args or {}, "every": float(every or 0), "at": float(at or 0),
               "max_runs": int(max_runs or 0), "runs": 0, "last": 0.0,
               "next": float(at) if at else self.clock() + float(every)}
        self.jobs = [j for j in self.jobs if j["name"] != name] + [job]
        return job

    def due(self, now: float | None = None) -> list:
        now = float(self.clock() if now is None else now)
        return [j for j in self.jobs if j["next"] and j["runs"] < (j["max_runs"] or 1 << 30) and j["next"] <= now]

    def tick(self, dispatch: Callable, now: float | None = None) -> list:
        """Fire each due job once. dispatch(tool_name, args) returns a result dict."""
        now = float(self.clock() if now is None else now)
        results = []
        for job in self.due(now):
            outcome = {"job": job["name"], "tool": job["tool"], "at": now}
            try:
                outcome["result"] = dispatch(job["tool"], job["args"])
            except Exception as exc:  # noqa: BLE001 - one failing job must not block the rest
                outcome["error"] = f"{type(exc).__name__}: {exc}"
            job["runs"] += 1
            job["last"] = now
            job["next"] = now + job["every"] if job["every"] else 0.0
            self.fired.append(outcome)
            results.append(outcome)
        return results

    def remove(self, name: str) -> dict:
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j["name"] != name]
        return {"removed": before - len(self.jobs), "left": len(self.jobs)}

    def register(self, reg: Registry) -> None:
        reg.add(Tool("schedule.add", "run a tool every N seconds or at an epoch time",
                     args={"name": str, "tool": str, "args": dict, "every": (float, "seconds between runs"),
                           "at": (float, "epoch fire time"), "max_runs": int},
                     required=("name", "tool"), handler=self.add, kind="schedule", mutating=True))
        reg.add(Tool("schedule.list", "jobs, their next run, and recent firings",
                     handler=lambda: {"jobs": self.jobs, "fired": self.fired[-10:]}, kind="schedule"))
        reg.add(Tool("schedule.due", "jobs whose time has come",
                     handler=lambda: {"due": [j["name"] for j in self.due()]}, kind="schedule"))
        reg.add(Tool("schedule.remove", "drop a job", args={"name": str}, required=("name",),
                    handler=self.remove, kind="schedule", mutating=True))


# ---------------------------------------------------------------------------
# models - every endpoint is a tool with one action contract
# ---------------------------------------------------------------------------

NL = chr(10)


def http_json(url: str, payload: dict, timeout: float = 30.0, headers: dict | None = None) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"content-type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as exc:
        raise ToolError(f"{url} -> HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ToolError(f"{url} unreachable: {exc}") from None
    try:
        return json.loads(body)
    except ValueError:
        raise ToolError(f"{url} returned non-JSON: {body[:200]}") from None


def chatml_token(label: str) -> str:
    return chr(60) + "|" + label + "|" + chr(62)


CHATML_START = chatml_token("start")
CHATML_MESSAGE = chatml_token("message")
CHATML_END = chatml_token("end")
CHATML_STOPS = [CHATML_END, chatml_token("im_end")]


def render_chatml(messages: list) -> str:
    """llama.cpp /completion takes a raw prompt; most GGUFs are trained on ChatML.

    llama.cpp's own template is START+role+MESSAGE+content+END per turn, then an open
    START+assistant+MESSAGE for the reply. chatml_token keeps the control sequences out
    of the source so this file stays safe to paste into a chat.
    """
    return ("".join(CHATML_START + str(m.get("role", "user")) + CHATML_MESSAGE
                    + str(m.get("content", "")) + CHATML_END for m in messages)
            + CHATML_START + "assistant" + CHATML_MESSAGE)


TAG_RE = re.compile(r"<[^>]+>")
NOISE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)
LINK_RE = re.compile(r"<a[^>]*href=[\"']?([^\"'> ]+)[\"']?[^>]*>(.*?)</a>", re.S | re.I)


def html_reduce(markup: Any) -> dict:
    """A page, cut down to the three things an agent reads: title, running text, links."""
    stripped = NOISE_RE.sub(" ", str(markup or ""))

    def flat(value: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(TAG_RE.sub(" ", value))).strip()

    found = TITLE_RE.search(stripped)
    body = stripped.replace(found.group(0), " ", 1) if found else stripped
    return {"title": flat(found.group(1)) if found else "",
            "text": flat(body)[:8000],
            "links": [{"href": href, "text": flat(label)} for href, label in LINK_RE.findall(body)][:20]}


class Models:
    """The brain is pluggable and callable: model.* are ordinary tools.

    Every model tool takes {messages, tools, goal, step} and returns the action
    contract, so swapping one is a policy change, not a code change.
    """

    def __init__(self, scripted: list | None = None, registry: Registry | None = None,
                 env: dict | None = None):
        self.registry = registry or _default_registry
        self.env = env if env is not None else os.environ
        self.scripts: dict[str, list] = {}
        self.turns: dict[str, int] = {}
        self.log: list[dict] = []
        if scripted:
            self.scripts["default"] = list(scripted)

    def script(self, script: str = "default", turns: Any = None, append: bool = False) -> dict:
        if isinstance(turns, str):
            turns = json.loads(turns)
        rows = list(turns or [])
        self.scripts[script] = (self.scripts.get(script, []) + rows) if append else rows
        self.turns.pop(script, None)
        return {"script": script, "turns": len(self.scripts[script])}

    def scripted(self, messages: list | None = None, tools: list | None = None, goal: str = "",
                 step: int = 0, script: str = "default", **_) -> dict:
        """Offline, deterministic, and the reason every example runs with no key."""
        queue = self.scripts.setdefault(script, [])
        index = self.turns.get(script, 0)
        self.turns[script] = index + 1
        self.log.append({"model": "scripted", "script": script, "step": step, "tools": len(tools or [])})
        if index < len(queue):
            return dict(queue[index])
        return {"thought": "script exhausted",
                "final": f"(script {script!r} ran out after {index} turns; goal={goal[:60]})"}

    def ollama(self, messages: list | None = None, tools: list | None = None, goal: str = "", step: int = 0,
               model: str = "", url: str = "", format: str = "json", temperature: float = 0.2, **_) -> dict:
        """POST /api/chat on Ollama. format=json keeps small local models on contract."""
        base = url or self.env.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        payload = {"model": model or self.env.get("OLLAMA_MODEL", "qwen3:8b"),
                   "messages": messages or [], "stream": False, "format": format,
                   "options": {"temperature": float(temperature), "num_ctx": 8192}}
        if tools:
            payload["tools"] = [self.registry.get(t).openai_spec() for t in tools if self.registry.has(t)]
        data = http_json(f"{base.rstrip('/')}/api/chat", payload, timeout=float(self.env.get("AGENT_TIMEOUT", 120)))
        content = (data.get("message") or {}).get("content") or data.get("response") or ""
        return self._action_from(content, data.get("message", {}).get("tool_calls"))

    def openai(self, messages: list | None = None, tools: list | None = None, goal: str = "", step: int = 0,
               model: str = "", base_url: str = "", api_key: str = "", **_) -> dict:
        """POST /chat/completions on any OpenAI-compatible server (vLLM, LM Studio, llama.cpp -ngl, OpenAI)."""
        base = base_url or self.env.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
        key = api_key or self.env.get("OPENAI_API_KEY", "ollama")
        specs = [self.registry.get(t).openai_spec() for t in (tools or []) if self.registry.has(t)]
        payload = {"model": model or self.env.get("OPENAI_MODEL", "qwen3:8b"),
                   "messages": messages or [], "temperature": 0.2, "tool_choice": "auto"}
        if specs:
            payload["tools"] = specs
        data = http_json(f"{base.rstrip('/')}/chat/completions", payload, timeout=float(self.env.get("AGENT_TIMEOUT", 120)),
                         headers={"authorization": f"Bearer {key}"})
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise ToolError(f"unexpected reply: {json.dumps(data)[:300]}") from None
        return self._action_from(choice.get("content") or "", choice.get("tool_calls"))

    def llamacpp(self, messages: list | None = None, tools: list | None = None, goal: str = "", step: int = 0,
                 base_url: str = "", model: str = "", temperature: float = 0.1, n_predict: int = 400, **_) -> dict:
        """POST /completion on a llama.cpp server, forcing the action shape with a json_schema."""
        base = base_url or self.env.get("LLAMACPP_URL", "http://127.0.0.1:8080")
        tool_list = "\n".join(f"- {t}: {self.registry.get(t).desc}" for t in (tools or []) if self.registry.has(t))
        prompt = render_chatml([{"role": "system", "content": (messages or [{}])[0].get("content", "")},
                                {"role": "user", "content": f"Tools:{NL}{tool_list}{NL}Goal: {goal}"}])
        payload = {"prompt": prompt, "temperature": float(temperature), "n_predict": int(n_predict),
                   "stream": False, "no_generation": False, "stop": CHATML_STOPS,
                   "json_schema": json.dumps({"type": "object", "properties": {
                       "thought": {"type": "string"},
                       "call": {"type": "object", "properties": {"tool": {"enum": list(tools or [])},
                                                                 "args": {"type": "object"}}},
                       "final": {"type": "string"}}, "required": ["thought"]})}
        if model:
            # a router-mode llama-server hosts several models and cannot guess which one
            payload["model"] = model
        data = http_json(f"{base.rstrip('/')}/completion", payload, timeout=float(self.env.get("AGENT_TIMEOUT", 180)))
        raw = data.get("content") or data.get("generation") or ""
        for stop in CHATML_STOPS:
            raw = raw.split(stop)[0]
        return self._action_from(raw)

    def _action_from(self, content: str, tool_calls: list | None = None) -> dict:
        """Normalise whatever the server said into {thought, call} or {thought, final}."""
        if tool_calls:
            call = tool_calls[0].get("function", {})
            args = call.get("arguments")
            args = json.loads(args) if isinstance(args, str) else (args or {})
            return {"thought": content or call.get("name", ""), "call": {"tool": call.get("name", ""), "args": args}}
        action = first_json(content)
        if not action:
            raise ToolError(f"model did not return the action contract: {content[:200]!r}")
        if "action" in action and "call" not in action:
            action["call"] = action.pop("action")
        return action

    def register(self, reg: Registry) -> None:
        common = {"messages": (list, "transcript"), "tools": (list, "tool names the model may call"),
                  "goal": (str, "what the run is for"), "step": (int, "turn index")}
        reg.add(Tool("model.scripted", "offline deterministic model: replays a queued script of actions",
                     args={**common, "script": (str, "named script queue")},
                     handler=self.scripted, kind="model"))
        reg.add(Tool("model.set_script", "queue the actions model.scripted should return",
                    args={"script": str, "turns": list, "append": (bool, "add to the queue instead of replacing")},
                    required=("turns",), handler=self.script, kind="model", mutating=True))
        reg.add(Tool("model.ollama", "Ollama /api/chat as the decision maker",
                     args={**common, "model": str, "url": str, "format": str, "temperature": float},
                     handler=self.ollama, kind="model", network=True))
        reg.add(Tool("model.openai", "any OpenAI-compatible /chat/completions as the decision maker",
                     args={**common, "model": str, "base_url": str, "api_key": str},
                     handler=self.openai, kind="model", network=True))
        reg.add(Tool("model.llamacpp", "llama.cpp server /completion with json_schema as the decision maker",
                     args={**common, "base_url": str, "model": str, "temperature": float, "n_predict": int},
                     handler=self.llamacpp, kind="model", network=True))
        reg.add(Tool("model.log", "recent model calls: which endpoint, how many tools offered",
                     args={"limit": int}, handler=lambda limit=20: self.log[-int(limit):], kind="model"))


# ---------------------------------------------------------------------------
# built-in tools - the small pack every harness needs, still behind the funnel
# ---------------------------------------------------------------------------


class Builtins:
    def __init__(self, root: Path, clock: Callable = time.time):
        self.root = Path(str(root))
        self.clock = clock

    def _path(self, path: Any) -> Path:
        return resolve(path, self.root)

    def edit(self, path: str, find: str, replace: str = "", all: bool = False) -> dict:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"{path} is not a file under {self.root}")
        content = target.read_text(encoding="utf-8", errors="replace")
        hits = content.count(str(find))
        if not hits:
            raise ToolError(f"no match for {str(find)[:40]!r} in {path}")
        if not all and hits > 1:
            raise ToolError(f"{str(find)[:40]!r} matches {hits} times in {path}; widen it, or pass all=true")
        done = content.replace(str(find), str(replace)) if all else content.replace(str(find), str(replace), 1)
        target.write_text(done, encoding="utf-8")
        return {"path": str(target), "bytes": target.stat().st_size,
                "replacements": hits if all else 1, "matches": hits}

    def read(self, path: str, max_bytes: int = 20000) -> dict:
        target = self._path(path)
        if not target.is_file():
            raise ToolError(f"{path} is not a file under {self.root}")
        raw = target.read_bytes()[: int(max_bytes)]
        return {"path": str(target), "bytes": len(target.read_bytes()), "content": raw.decode("utf-8", "replace")}

    def write(self, path: str, content: str = "", append: bool = False) -> dict:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = str(content)
        if append and target.exists():
            text = "\n" + text
        with target.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(text)
        return {"path": str(target), "bytes": target.stat().st_size, "mode": "append" if append else "write"}

    def listing(self, path: str = ".", pattern: str = "*") -> dict:
        target = self._path(path)
        if not target.is_dir():
            raise ToolError(f"{path} is not a directory")
        rows = sorted(target.glob(pattern))
        return {"path": str(target), "entries": [{"name": p.name, "dir": p.is_dir(),
                                                  "bytes": p.stat().st_size if p.is_file() else 0} for p in rows[:200]]}

    def exists(self, path: str) -> dict:
        target = self._path(path)
        return {"path": str(target), "exists": target.exists(), "dir": target.is_dir(), "file": target.is_file()}

    def delete(self, path: str) -> dict:
        target = self._path(path)
        if target.is_dir():
            raise ToolError("fs.delete removes files only; a directory is too blunt an instrument")
        if not target.exists():
            raise ToolError(f"{path} does not exist")
        target.unlink()
        return {"deleted": str(target)}

    def shell(self, cmd: str, timeout: float = 20.0) -> dict:
        argv = shlex.split(str(cmd))
        if not argv:
            raise ToolError("empty command")
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=float(timeout), cwd=str(self.root))
        except FileNotFoundError:
            raise ToolError(f"program not found: {argv[0]}") from None
        except subprocess.TimeoutExpired:
            raise ToolError(f"{cmd} timed out after {timeout}s") from None
        return {"cmd": cmd, "code": done.returncode, "stdout": done.stdout[:8000], "stderr": done.stderr[:4000]}

    def http_get(self, url: str, timeout: float = 20.0, max_bytes: int = 20000) -> dict:
        try:
            with urllib.request.urlopen(str(url), timeout=float(timeout)) as response:
                body = response.read(int(max_bytes)).decode("utf-8", "replace")
                status = getattr(response, "status", None) or 200
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ToolError(f"{url} unreachable: {exc}") from None
        return {"url": url, "status": status, "body": body}

    def browse(self, url: str, timeout: float = 20.0, max_bytes: int = 40000) -> dict:
        page = self.http_get(url, timeout=timeout, max_bytes=max_bytes)
        reduced = {"url": page["url"], "status": page["status"]}
        reduced.update(html_reduce(page["body"]))
        return reduced

    def register(self, reg: Registry) -> None:
        reg.add(Tool("fs.read", "read a text file inside the sandbox",
                     args={"path": str, "max_bytes": int}, required=("path",), handler=self.read, kind="tool"))
        reg.add(Tool("fs.write", "write or append a file inside the sandbox",
                     args={"path": str, "content": str, "append": bool}, required=("path",),
                     handler=self.write, kind="tool", mutating=True))
        reg.add(Tool("fs.edit", "replace text inside a file; a second match needs all=true",
                     args={"path": str, "find": str, "replace": str, "all": bool},
                     required=("path", "find", "replace"), handler=self.edit, kind="tool", mutating=True))
        reg.add(Tool("fs.list", "list a directory", args={"path": str, "pattern": str}, handler=self.listing, kind="tool"))
        reg.add(Tool("fs.exists", "does this path exist?", args={"path": str}, required=("path",),
                     handler=self.exists, kind="tool"))
        reg.add(Tool("fs.delete", "remove one file (irreversible)", args={"path": str}, required=("path",),
                     handler=self.delete, kind="tool", mutating=True, danger=True, deletes=True))
        reg.add(Tool("shell.run", "run a program with argv; no shell interpolation",
                     args={"cmd": str, "timeout": float}, required=("cmd",), handler=self.shell,
                     kind="tool", mutating=True, danger=True))
        reg.add(Tool("http.get", "fetch a URL", args={"url": str, "timeout": float, "max_bytes": int},
                    required=("url",), handler=self.http_get, kind="tool", network=True))
        reg.add(Tool("browser.open", "fetch a URL and reduce it to title, text and links",
                     args={"url": str, "timeout": float, "max_bytes": int},
                     required=("url",), handler=self.browse, kind="tool", network=True))
        reg.add(Tool("time.now", "current epoch seconds", handler=lambda: {"now": self.clock()}, kind="tool"))
        reg.add(Tool("echo", "return its argument - the hello-world plug", args={"text": str}, required=("text",),
                     handler=lambda text: {"echo": text}, kind="tool"))


# ---------------------------------------------------------------------------
# the loop - one funnel, repeated: registry -> hooks -> policy -> handler -> hooks -> context
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Call:
    tool: str
    kind: str = "tool"
    plugin: str = "core"
    args: dict = field(default_factory=dict)
    ok: bool = False
    value: Any = None
    error: str = ""
    blocked: bool = False
    reason: str = ""
    source: str = "loop"
    ms: float = 0.0

    def observation(self) -> dict:
        if self.blocked:
            return {"blocked": True, "tool": self.tool, "reason": self.reason}
        if not self.ok:
            return {"error": self.error, "tool": self.tool}
        return {"tool": self.tool, "result": self.value}

    def brief(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Turn:
    i: int
    thought: str = ""
    action: dict = field(default_factory=dict)
    call: Call | None = None
    final: str = ""

    def brief(self) -> dict:
        return {"i": self.i, "thought": self.thought, "tool": self.call.tool if self.call else "",
                "ok": self.call.ok if self.call else None, "blocked": self.call.blocked if self.call else None,
                "final": self.final, "action": self.action}


@dataclass(slots=True)
class Run:
    goal: str
    model: str
    status: str = "running"
    turns: list = field(default_factory=list)
    started: float = 0.0
    seconds: float = 0.0
    memory_writes: int = 0
    tasks_created: int = 0
    compactions: int = 0
    error: str = ""

    @property
    def trail(self) -> str:
        return " -> ".join(t.call.tool for t in self.turns if t.call)

    @property
    def blocked(self) -> list:
        return [{"i": t.i, "tool": t.call.tool, "reason": t.call.reason} for t in self.turns if t.call and t.call.blocked]

    @property
    def answer(self) -> str:
        return next((t.final for t in reversed(self.turns) if t.final), "")

    @property
    def ok(self) -> bool:
        return self.status == "done"

    def __str__(self) -> str:
        return f"[{self.model}] {self.status}: {self.answer[:120] or self.trail}"


class Agent:
    """A wired harness: nine components, one registry, one funnel, one loop."""

    def __init__(self, name: str = "harness", root: Any = ".", model: str = "model.scripted", *,
                 tools_dir: Any = None, skills_dir: Any = ("skills",), memory_path: Any = None,
                 budget_tokens: int = 4000, keep_last: int = 6, max_steps: int = 12,
                 allow_network: bool = False, dry_run: bool = False, extra_dirs: Any = (),
                 registry: Registry | None = None, env: dict | None = None, clock: Callable = time.time,
                 verbose: bool = False):
        self.name = name
        self.root = Path(str(root)).resolve()
        self.clock = clock
        # start from a copy of the process-wide registry so @tool plugins and
        # discover() results are adopted, then keep our own so agents cannot leak.
        self.registry = registry or Registry()
        self.registry.tools.update(_default_registry.tools)
        self.components: dict[str, Any] = {}

        self.policy = Policy(root=self.root, allow_network=allow_network, dry_run=dry_run,
                             max_steps=max_steps, extra_dirs=list(extra_dirs))
        self.hooks = Hooks(self.registry)
        self.memory = Memory(memory_path, clock=clock)
        self.context = Context(budget_tokens=budget_tokens, keep_last=keep_last)
        self.prompts = Prompts()
        self.skills = Skills(skills_dir if skills_dir is not None else (), registry=self.registry)
        self.tasks = Tasks(clock=clock)
        self.scheduler = Scheduler(clock=clock)
        self.models = Models(registry=self.registry, env=env)
        self.builtins = Builtins(self.root, clock=clock)
        for component in (self.policy, self.hooks, self.memory, self.context, self.prompts, self.skills,
                          self.tasks, self.scheduler, self.models, self.builtins):
            self.components.setdefault(type(component).__name__, component)
        self.model = model
        self.tool_defaults: dict[str, dict] = {}   # operator-set args merged into every call
        self.journal: list[Call] = []
        self.runs: list[Run] = []
        self.verbose = verbose
        self.tools_dir = tools_dir
        if tools_dir:
            self.plugins = discover(tools_dir, registry=self.registry)
        else:
            self.plugins = {"loaded": [], "modules": [], "errors": {}}
        self.register_loop_tools()

    # -- wiring -----------------------------------------------------------
    def register_loop_tools(self) -> None:
        reg = self.registry
        self.hooks.register(reg)
        self.memory.register(reg)
        self.context.register(reg)
        self.prompts.register(reg, self)
        self.skills.register(reg)
        self.tasks.register(reg)
        self.scheduler.register(reg)
        self.models.register(reg)
        self.builtins.register(reg)

        reg.add(Tool("tools.list", "every registered tool, by component kind",
                     args={"kind": (str, "tool|skill|memory|context|hook|policy|prompt|loop|task|schedule|model")},
                     handler=lambda kind="": {"count": len(self.registry), "kinds": {k: self.registry.names(k)
                                                                                    for k in {t.kind for t in self.registry.tools.values()}},
                                              "tools": self.registry.brief(kind or None)},
                     kind="loop"))
        reg.add(Tool("tools.describe", "one tool: its description, args and guardrail flags", args={"tool": str},
                    required=("tool",), handler=lambda tool: self.registry.get(tool).brief(), kind="loop"))
        reg.add(Tool("tools.call", "invoke any registered tool by name - the meta plug",
                     args={"tool": str, "args": dict}, required=("tool",), handler=self._t_tools_call,
                     kind="loop", mutating=True))
        reg.add(Tool("policy.get", "current guardrail settings and budget usage", handler=self.policy.summary, kind="policy"))
        reg.add(Tool("policy.set", "change guardrails: allow, deny, dry_run, allow_network, max_* ...",
                     args={"allow": list, "deny": list, "confirm_required": list, "extra_dirs": list,
                           "max_steps": int, "max_tool_calls": int, "max_writes": int, "max_deletes": int,
                           "max_seconds": float, "allow_network": bool, "dry_run": bool},
                     handler=self._t_policy_set, kind="policy", mutating=True))
        reg.add(Tool("policy.check", "ask whether the guardrails would allow a call",
                     args={"tool": str, "args": dict}, required=("tool",), handler=self._t_policy_check, kind="policy"))
        reg.add(Tool("confirm.grant", "operator approval for one confirm-required tool (one call unless repeats)",
                     args={"tool": str, "repeats": (bool, "keep approving until revoked")}, required=("tool",),
                     handler=self._t_grant, kind="policy", mutating=True))
        reg.add(Tool("confirm.revoke", "withdraw approval for a tool", args={"tool": str}, required=("tool",),
                     handler=lambda tool: {"revoked": self.policy.revoke(tool)}, kind="policy", mutating=True))
        reg.add(Tool("context.compact", "fold old turns into one summary, with a model if one is allowed",
                     args={"model": (str, "blank = extractive summary")}, handler=self.compact, kind="context",
                     mutating=True))
        reg.add(Tool("agent.status", "the live picture: loop, budgets, window, plugins, jobs, tasks",
                     handler=self.panel, kind="loop"))
        reg.add(Tool("agent.step", "run exactly one observe-act cycle and return it",
                     args={"goal": (str, "set or replace the goal")}, handler=self.step, kind="loop", mutating=True))
        reg.add(Tool("agent.run", "hand a goal to the loop and let it drive (denied by default)",
                     args={"goal": str, "max_steps": int}, required=("goal",), handler=self.run, kind="loop",
                     mutating=True))

    # -- the funnel -------------------------------------------------------
    def call(self, name: str, args: dict | None = None, source: str = "loop") -> Call:
        started = self.clock()
        spec = self.registry.tools.get(name)
        if spec is None:
            record = Call(tool=name, args=args or {}, source=source,
                          error=f"no tool {name!r}. {len(self.registry)} are registered; tools.list shows them",
                          ms=round(1000 * (self.clock() - started), 2))
            self.journal.append(record)
            self.hooks.emit("tool_error", {"tool": name, "args": args or {}, "error": record.error, "source": source})
            return record
        payload = {"tool": name, "args": {**(self.tool_defaults.get(name) or {}), **(args or {})},
                   "source": source, "kind": spec.kind}
        verdict = self.hooks.emit("pre_tool", payload)
        if verdict["block"]:
            record = Call(tool=name, kind=spec.kind, plugin=spec.plugin, args=payload["args"], blocked=True,
                          reason=verdict["reason"], source=source)
        else:
            allowed_args = verdict["args"]
            ok, reason = self.policy.check(spec, allowed_args, source=source)
            if not ok:
                record = Call(tool=name, kind=spec.kind, plugin=spec.plugin, args=allowed_args, blocked=True,
                              reason=reason, source=source)
            else:
                try:
                    value = spec.call(allowed_args)
                    record = Call(tool=name, kind=spec.kind, plugin=spec.plugin, args=allowed_args, ok=True,
                                  value=value, source=source)
                except ToolError as exc:
                    record = Call(tool=name, kind=spec.kind, plugin=spec.plugin, args=allowed_args,
                                  error=str(exc), source=source)
                except Exception as exc:  # noqa: BLE001 - a plugin crash is data, not a dead agent
                    record = Call(tool=name, kind=spec.kind, plugin=spec.plugin, args=allowed_args,
                                  error=f"{type(exc).__name__}: {error_text(exc)}", source=source)
        record.ms = round(1000 * (self.clock() - started), 2)
        if record.blocked:
            self._record_block(name, record.reason)
        self.journal.append(record)
        if record.error:
            self.hooks.emit("tool_error", {"tool": name, "args": record.args, "error": record.error, "source": source})
        elif record.ok:
            post = self.hooks.emit("post_tool", {"tool": name, "args": record.args, "result": record.value,
                                                 "source": source})
            if post["block"]:
                record.blocked, record.reason = True, f"post_tool veto: {post['reason']}"
                record.ok = False
        if self.verbose:
            print(f"  {name} {'BLOCK' if record.blocked else 'ERR' if record.error else 'ok'} "
                  f"{record.ms}ms {str(record.value or record.reason)[:90]}")
        return record

    def _record_block(self, name: str, reason: str) -> None:
        """Hook blocks land here; policy blocks already recorded themselves."""
        if not any(b["tool"] == name and b["reason"] == reason for b in self.policy.blocks[-3:]):
            self.policy.blocks.append({"tool": name, "reason": reason, "at": self.clock()})

    def tool_prompt(self) -> str:
        lines = []
        for spec in self.registry.tools.values():
            if any(fnmatch.fnmatchcase(spec.name, pattern) for pattern in self.policy.deny):
                continue
            flags = "!" if spec.danger else "+" if spec.mutating else "~" if spec.network else " "
            args = " ".join(f"<{k}>" if k in spec.required else f"[{k}]" for k in spec.args)
            lines.append(f"{flags} {spec.name} {args}: {spec.desc}")
        return "\n".join(sorted(lines))

    def visible_tools(self) -> list:
        return [n for n in self.registry.names()
                if not any(fnmatch.fnmatchcase(n, pattern) for pattern in self.policy.deny)]

    def think(self, step: int, goal: str) -> tuple[dict, Call]:
        messages = self.context.as_messages()
        call = self.call(self.model, {"messages": messages, "tools": self.visible_tools(), "goal": goal, "step": step},
                         source="model")
        if call.blocked or not call.ok:
            return {"thought": "", "final": ""}, call
        value = call.value if isinstance(call.value, dict) else {}
        return value, call

    def file_side_channels(self, action: dict) -> dict:
        """A model can hand back memory/task updates alongside its action."""
        notes = tasks = 0
        memory = action.get("memory")
        if isinstance(memory, dict) and memory.get("key"):
            self.memory.write(memory["key"], memory.get("value", ""), memory.get("tags", ""),
                              memory.get("kind", "note"))
            notes += 1
        task = action.get("task")
        if isinstance(task, dict) and task.get("subject"):
            self.tasks.create(task["subject"], task.get("description", ""), task.get("status", "pending"))
            tasks += 1
        return {"memory": notes, "tasks": tasks}

    # -- the loop ---------------------------------------------------------
    def run(self, goal: str, max_steps: int = 0, reset: bool = True, script: Any = None) -> Run:
        if script:
            self.models.script("default", script)
        run = Run(goal=goal, model=self.model, started=self.clock())
        self.policy.start_run()
        if reset:
            self.context.reset(self.prompts.render(self))
        self.context.push("user", goal, name="goal")
        steps = int(max_steps or self.policy.max_steps)
        self.hooks.emit("plan", {"tool": "goal", "args": {"goal": goal, "steps": steps}})
        model_failures = 0
        for index in range(steps):
            action, model_call = self.think(index, goal)
            if model_call.blocked or not model_call.ok:
                model_failures += 1
                self.context.push("tool", model_call.observation(), name=self.model)
                if model_failures >= 2:
                    run.status = "model-failed"
                    run.error = model_call.reason or model_call.error
                    break
                continue
            model_failures = 0
            thought = str(action.get("thought") or "")
            filed = self.file_side_channels(action)
            run.memory_writes += filed["memory"]
            run.tasks_created += filed["tasks"]
            turn = Turn(i=index, thought=thought, action=action)
            if action.get("final"):
                turn.final = str(action["final"])
                self.context.push("assistant", turn.final, name="final")
                run.turns.append(turn)
                run.status = "done"
                break
            spec = action.get("call") or {}
            name = str(spec.get("tool") or "")
            if not name:
                observation = {"error": f"turn {index} had neither call nor final: {json.dumps(action)[:200]}"}
                self.context.push("tool", observation, name="loop")
                run.turns.append(turn)
                continue
            record = self.call(name, spec.get("args") or {}, source="loop")
            turn.call = record
            run.turns.append(turn)
            self.context.push("assistant", thought or name, name="thought")
            self.context.push("tool", record.observation(), name=name)
            if self.context.needs_compaction():
                run.compactions += 1 if self.compact().get("compacted") else 0
            if self.scheduler.due():
                self.tick()
        if run.status == "running":
            run.status = "steps-exhausted"
        run.seconds = round(self.clock() - run.started, 3)
        self.hooks.emit("stop", {"tool": run.status, "args": {"goal": goal, "turns": len(run.turns)}})
        self.runs.append(run)
        return run

    def step(self, goal: str = "") -> dict:
        """One cycle for external control: a scheduler or a human can drive the loop."""
        if goal:
            self.context.push("user", goal, name="goal")
        active = next((r for r in reversed(self.runs) if r.status == "running"), None)
        target = goal or (active.goal if active else self.context.messages[0]["content"] if self.context.messages else "")
        index = len(active.turns) if active else 0
        action, model_call = self.think(index, target)
        if model_call.blocked or not model_call.ok:
            return {"ok": False, "reason": model_call.reason or model_call.error, "model": self.model}
        filed = self.file_side_channels(action)
        if action.get("final"):
            self.context.push("assistant", action["final"], name="final")
            return {"ok": True, "thought": action.get("thought", ""), "final": action["final"], "filed": filed}
        spec = action.get("call") or {}
        record = self.call(str(spec.get("tool") or ""), spec.get("args") or {}, source="step")
        self.context.push("assistant", action.get("thought") or "", name="thought")
        self.context.push("tool", record.observation(), name=record.tool)
        return {"ok": record.ok, "thought": action.get("thought", ""), "call": record.brief(), "filed": filed}

    def compact(self, model: str = "") -> dict:
        if not model or not self.registry.has(model):
            return self.context.compact()
        digest = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in self.context.messages)
        ask = self.call(model, {"messages": [{"role": "system",
                                              "content": "Summarise the transcript in <=3 sentences, keeping names, "
                                                         "paths, numbers and any blocked call."},
                                             {"role": "user", "content": digest}]}, source="compact")
        if ask.ok and isinstance(ask.value, dict):
            return self.context.compact(str(ask.value.get("final") or ask.value.get("thought") or "") or None)
        return self.context.compact()

    def dispatch(self, name: str, args: dict | None = None) -> dict:
        return self.call(name, args or {}, source="schedule").observation()

    def tick(self, now: float | None = None) -> list:
        return self.scheduler.tick(self.dispatch, now)

    def script(self, turns: Any, name: str = "default", append: bool = False) -> dict:
        return self.models.script(name, turns, append)

    def hook(self, event: str, pattern: str = "*", call: Callable | None = None, tool: str = "", name: str = "") -> Hook:
        return self.hooks.add(event, pattern, call=call, tool=tool, name=name)

    def grant(self, name: str, repeats: bool = False) -> dict:
        self.policy.grant(name, repeats)
        return {"granted": name, "repeats": repeats}

    def panel(self) -> dict:
        kinds = {}
        for spec in self.registry.tools.values():
            kinds.setdefault(spec.kind, []).append(spec.name)
        return {"agent": self.name, "root": str(self.root), "model": self.model,
                "tools": len(self.registry), "kinds": {k: len(v) for k, v in sorted(kinds.items())},
                "plugins": self.plugins, "policy": self.policy.summary(),
                "context": self.context.stats(), "memory": len(self.memory.entries),
                "skills": {"installed": [s["name"] for s in self.skills.list()], "active": self.skills.active},
                "tasks": self.tasks.summary(), "jobs": len(self.scheduler.jobs),
                "hooks": self.hooks.describe(), "hook_firings": len(self.hooks.fired),
                "blocks": self.policy.blocks[-5:], "journal": len(self.journal),
                "runs": [{"goal": r.goal[:60], "status": r.status, "turns": len(r.turns)} for r in self.runs]}

    # -- tool handlers that need the funnel -------------------------------
    def _t_tools_call(self, tool: str, args: dict | None = None) -> dict:
        return self.call(tool, args or {}, source="tools.call").observation()

    def _t_policy_set(self, **changes) -> dict:
        clean = {k: v for k, v in changes.items() if v not in (None, "")}
        return self.policy.set(**clean)

    def _t_policy_check(self, tool: str, args: dict | None = None) -> dict:
        spec = self.registry.get(tool)
        snapshot = (self.policy.calls, self.policy.writes, self.policy.deletes, len(self.policy.blocks))
        ok, reason = self.policy.check(spec, args or {})
        self.policy.calls, self.policy.writes, self.policy.deletes = snapshot[0], snapshot[1], snapshot[2]
        del self.policy.blocks[snapshot[3]:]
        return {"allowed": ok, "reason": reason, "tool": tool, "kind": spec.kind,
                "mutating": spec.mutating, "network": spec.network, "danger": spec.danger}

    def _t_grant(self, tool: str, repeats: bool = False) -> dict:
        self.registry.get(tool)
        return self.grant(tool, repeats)


# ---------------------------------------------------------------------------
# examples — every subsystem, run for real
# ---------------------------------------------------------------------------


def _scratch(name: str = "example", **kw) -> Agent:
    root = Path(tempfile.mkdtemp(prefix="agent-example-"))
    kw.setdefault("memory_path", root / ".agent" / "memory.json")
    return Agent(name=name, root=root, **kw)


def ex_tool_loop() -> list:
    """A scripted model drives the funnel: write a file, read it back, answer."""
    agent = _scratch()
    agent.grant("fs.write")
    agent.script([
        {"thought": "put the note down",
         "call": {"tool": "fs.write", "args": {"path": "notes.md", "content": "alpha beta gamma\n"}}},
        {"thought": "read it back", "call": {"tool": "fs.read", "args": {"path": "notes.md"}}},
        {"thought": "answer", "final": "the note holds one line of three words"},
    ])
    run = agent.run("write a note and prove you read it")
    return [f"status={run.status}  trail={run.trail}",
            f"answer={run.answer}",
            f"journal={[c.tool for c in agent.journal]}"]


def ex_guardrails() -> list:
    """Deny-first patterns, a path sandbox, approvals and a vetoing hook."""
    agent = _scratch()
    (agent.root / "secret.txt").write_text("do not read", encoding="utf-8")
    agent.hook("pre_tool", "fs.read", name="no-secrets",
               call=lambda payload: {"block": True, "reason": "private file"}
               if "secret" in str(payload["args"].get("path", "")) else None)
    agent.script([
        {"thought": "write without asking", "call": {"tool": "fs.write", "args": {"path": "a.txt", "content": "x"}}},
        {"thought": "read the secret", "call": {"tool": "fs.read", "args": {"path": "secret.txt"}}},
        {"thought": "escape the sandbox", "call": {"tool": "fs.read", "args": {"path": "../outside.txt"}}},
        {"thought": "delete it", "call": {"tool": "fs.delete", "args": {"path": "a.txt"}}},
        {"thought": "give up", "final": "four different guards said no"},
    ])
    run = agent.run("try to break the rules")
    return [f"turn {block['i']} {block['tool']}: {block['reason']}" for block in run.blocked] + [
        f"policy recorded {len(agent.policy.blocks)} blocks, spent {agent.policy.calls} calls"]


def ex_memory_context() -> list:
    """The model files its own notes; a long window folds into a summary."""
    agent = _scratch(budget_tokens=900, keep_last=3)
    agent.grant("fs.write")
    agent.script([
        {"thought": "write a long file", "memory": {"key": "notes", "value": "3 lines of prose", "kind": "fact"},
         "call": {"tool": "fs.write", "args": {"path": "notes.md", "content": ("alpha beta gamma delta " * 12 + "\n")}}},
        {"thought": "read it back", "call": {"tool": "fs.read", "args": {"path": "notes.md"}}},
        {"thought": "recall", "call": {"tool": "memory.search", "args": {"query": "notes"}}},
        {"thought": "answer", "final": "3 lines, and the window folded itself"},
    ])
    run = agent.run("keep a note about the notes")
    return [f"status={run.status}  compactions during run={run.compactions}  memory_writes={run.memory_writes}",
            f"window={json.dumps(agent.context.stats())}",
            f"memory={agent.memory.notes()}"]


def ex_tasks_scheduler() -> list:
    """A job is a tool call through the same funnel; the loop ticks whatever is due."""
    agent = _scratch()
    first = agent.call("task.create", {"subject": "read the note"}).value
    agent.call("task.create", {"subject": "summarise it", "blocks": str(first["id"])})
    agent.call("schedule.add", {"name": "heartbeat", "tool": "echo", "args": {"text": "beat"}, "at": 1.0})
    agent.script([
        {"thought": "finish the first task", "call": {"tool": "task.update",
                                                      "args": {"id": first["id"], "status": "completed"}}},
        {"thought": "what is next", "call": {"tool": "task.ready"}},
        {"thought": "answer", "final": "the queue moved on"},
    ])
    run = agent.run("work the queue")
    return [f"fired={[f['job'] for f in agent.scheduler.fired]} -> {agent.scheduler.fired[0]['result']}",
            f"tasks={json.dumps(agent.tasks.summary())}",
            f"status={run.status}"]


def ex_skills_prompts() -> list:
    """A markdown skill pack joins the system prompt while it is active."""
    directory = Path(tempfile.mkdtemp(prefix="agent-skills-"))
    (directory / "notekeeper.md").write_text(
        "---\nname: notekeeper\ndescription: keep short cited notes\n"
        "when_to_use: the user asks for notes\ntools: fs.write, nope.tool\n---\n"
        "Always cite the file you read.\n", encoding="utf-8")
    agent = _scratch(skills_dir=[directory])
    used = agent.call("skills.use", {"name": "notekeeper"})
    rendered = agent.call("prompt.get", {"rendered": True}).value
    return [f"installed={[s['name'] for s in agent.skills.list()]}",
            f"requires={used.value['requires']}  missing={used.value['missing_tools']}",
            f"prompt tokens={rendered['tokens']}  cites the skill={'cite the file' in rendered['rendered']}"]


def ex_plugin_pack() -> list:
    """Drop a .py into ./tools and it becomes a named, guardrailed, callable tool."""
    directory = Path(tempfile.mkdtemp(prefix="agent-tools-"))
    (directory / "wordcount.py").write_text(
        "from agent import tool\n\n"
        "@tool(\"words.count\", \"count the words in the text\", args={\"text\": str}, required=(\"text\",))\n"
        "def count(text):\n    return {\"words\": len(text.split())}\n", encoding="utf-8")
    agent = _scratch(tools_dir=directory)
    called = agent.call("words.count", {"text": "one two three"})
    denied = agent.call("policy.set", {"deny": ["words.*"]})
    blocked = agent.call("words.count", {"text": "one two three"})
    return [f"plugins={agent.plugins['loaded']}  call ok={called.ok} value={called.value}",
            f"after deny={json.dumps(denied.value['deny'][:3])}  blocked={blocked.reason}"]


def ex_nested_agents() -> list:
    """A worker agent is just another tool in the planner's registry."""
    worker = _scratch(name="worker")
    worker.script([{"call": {"tool": "echo", "args": {"text": "counted 3 rows"}}},
                   {"final": "3 rows counted"}])
    planner = _scratch(name="planner")
    planner.registry.add(Tool("worker.ask", "hand a goal to the worker agent", args={"goal": str},
                              required=("goal",), kind="tool", mutating=True,
                              handler=lambda goal: worker.run(goal).answer))
    planner.script([{"call": {"tool": "worker.ask", "args": {"goal": "count the rows"}}},
                    {"final": "the worker says 3 rows counted"}])
    run = planner.run("delegate the counting")
    return [f"planner trail={run.trail}", f"worker journal={[c.tool for c in worker.journal]}",
            f"answer={run.answer}"]


def ex_local_endpoints() -> list:
    """ollama, an OpenAI-compatible server and llama.cpp are ordinary tools. Set AGENT_LIVE=1 to answer."""
    agent = _scratch()
    lines = [f"{name}: network={agent.registry.get(name).network} "
             f"endpoint args={sorted(k for k in agent.registry.get(name).args if k not in ('messages', 'tools', 'goal', 'step'))}"
             for name in ("model.ollama", "model.openai", "model.llamacpp")]
    live = os.environ.get("AGENT_LIVE") == "1"
    probe = _scratch(model="model.ollama", allow_network=live)
    answer = probe.call("model.ollama", {"messages": [{"role": "user", "content": "reply with {\"final\": \"ping\"}"}],
                                         "tools": ["echo"]})
    lines.append(f"live={live} ok={answer.ok} detail="
                 f"{json.dumps(answer.value if answer.ok else (answer.reason or answer.error), default=str)[:220]}")
    return lines


def ex_llm_call() -> list:
    """One plain request to one model endpoint - no loop, no tools, just a reply and a log line."""
    agent = _scratch()
    agent.script([{"thought": "12 * 4 is 48", "final": "48"}])
    reply = agent.call("model.scripted", {"messages": [{"role": "system", "content": "reply with one JSON action"},
                                                       {"role": "user", "content": "what is 12 * 4?"}],
                                          "tools": ["echo"], "goal": "answer the arithmetic", "step": 0})
    probes = [agent.call("policy.check", {"tool": name, "args": {}}).value
              for name in ("model.scripted", "model.openai", "model.llamacpp")]
    return [f"action back: {json.dumps(reply.value)}",
            f"model.log: {json.dumps(agent.call('model.log', {'limit': 2}).value)}",
            " | ".join(f"{p['tool']} allowed={p['allowed']}"
                       f"{'' if p['allowed'] else ' - ' + p['reason']}" for p in probes),
            "agent.run() is this same call on a loop: think, act, observe, until 'final' or the step budget"]


def ex_rag() -> list:
    """Index chunks in memory, retrieve one, answer only with what came back."""
    corpus = {"chunk:funnel": "The harness funnel is registry -> hooks -> policy -> handler.",
              "chunk:budgets": "Budgets cap writes, deletes, tool calls and wall-clock seconds.",
              "chunk:models": "Model endpoints are ordinary network tools, off by default."}
    agent = _scratch()
    for key, value in corpus.items():
        agent.call("memory.write", {"key": key, "value": value, "kind": "chunk", "tags": "harness"})
    hit = agent.call("memory.search", {"query": "what caps writes", "kind": "chunk"}).value[0]
    agent.script([
        {"thought": "retrieve first", "call": {"tool": "memory.search",
                                               "args": {"query": "what caps writes", "kind": "chunk"}}},
        {"thought": "put the retrieved chunk in the window",
         "call": {"tool": "context.push", "args": {"role": "user", "content": hit["value"], "name": "rag"}}},
        {"thought": "answer from the chunk", "final": f"{hit['key']}: {hit['value']}"},
    ])
    run = agent.run("what caps writes? answer only from the indexed chunks")
    return [f"indexed {len(corpus)} chunks, top hit={hit['key']}",
            f"trail={run.trail}", f"status={run.status}", f"answer={run.answer}",
            f"window={json.dumps(agent.context.stats())}"]


def ex_agentic_rag() -> list:
    """The agent chooses its own retrieval queries and searches again when a hit comes back empty."""
    agent = _scratch()
    for key, value in {"chunk:scheduler": "A job is a tool call on a timer; the loop ticks whatever is due.",
                       "chunk:tasks": "Tasks block other tasks until they are marked completed."}.items():
        agent.call("memory.write", {"key": key, "value": value, "kind": "chunk", "tags": "harness"})
    agent.script([
        {"thought": "start with the words the goal used",
         "call": {"tool": "memory.search", "args": {"query": "vector embedding cosine", "kind": "chunk"}}},
        {"thought": "that came back empty; ask differently",
         "call": {"tool": "memory.search", "args": {"query": "timer every due", "kind": "chunk"}}},
        {"thought": "the second hit answers it", "final": "a job is a tool call on a timer"},
    ])
    run = agent.run("how does the scheduler work?")
    searches = [c for c in agent.journal if c.tool == "memory.search"]
    return [f"trail={run.trail}",
            f"query 1 -> {json.dumps([h['key'] for h in (searches[0].value or [])]) if searches else '-'}",
            f"query 2 -> {json.dumps([h['key'] for h in (searches[1].value or [])]) if len(searches) > 1 else '-'}",
            f"status={run.status} answer={run.answer}"]


def ex_structured_output() -> list:
    """A post_tool hook is the schema: a malformed record is vetoed, so the model refiles it as JSON."""
    agent = _scratch()

    def validate(payload: dict) -> dict | None:
        try:
            json.loads(str((payload.get("args") or {}).get("value")))
        except (ValueError, TypeError):
            return {"block": True, "reason": "structured-output: value must be one JSON object"}
        return None

    agent.hook("post_tool", "memory.write", name="schema", call=validate)
    agent.script([
        {"thought": "file the record", "call": {"tool": "memory.write",
                                                "args": {"key": "invoice", "value": "{ total: 42, currency: EUR }"}}},
        {"thought": "keys quoted and refiled", "call": {"tool": "memory.write",
                                                         "args": {"key": "invoice",
                                                                   "value": json.dumps({"total": 42, "currency": "EUR"})}}},
        {"thought": "done", "final": "the second write passed the schema hook"},
    ])
    run = agent.run("file the invoice as structured data")
    return [f"trail={run.trail}", f"vetoed={json.dumps([b['reason'] for b in run.blocked])}",
            f"record now={agent.memory.read('invoice')['value']}", f"status={run.status}"]


def ex_thinking() -> list:
    """No reasoning, no answer: a hook vetoes model actions that arrive without a thought."""
    agent = _scratch()
    agent.hook("post_tool", "model.scripted", name="show-your-work",
               call=lambda payload: {"block": True, "reason": "thinking: show the reasoning first"}
               if not str((payload.get("result") or {}).get("thought") or "") else None)
    agent.script([
        {"call": {"tool": "echo", "args": {"text": "42, guessed"}}},
        {"thought": "41 + 1: carry the one, so 42", "call": {"tool": "echo", "args": {"text": "42"}}},
        {"thought": "answered with the reasoning shown", "final": "42"},
    ])
    run = agent.run("answer with reasoning")
    vetoed = [c for c in agent.journal if c.tool == "model.scripted" and c.blocked]
    return [f"model calls vetoes={len(vetoed)} ({vetoed[0].reason if vetoed else 'none'})",
            f"turns that survived={[(t.i, t.thought[:26]) for t in run.turns]}",
            f"trail={run.trail}", f"status={run.status} answer={run.answer}"]


def ex_judge() -> list:
    """A judge tool scores the draft; the loop rewrites until the bar is cleared."""
    agent = _scratch()

    def judge(text: str) -> dict:
        words = len(str(text).split())
        score = min(5, words // 4)
        return {"score": score, "verdict": "accept" if score >= 3 else "revise", "why": f"{words} words, {score}/5"}

    agent.registry.add(Tool("judge.score", "score a draft 0-5; 3 is the bar", args={"text": str},
                            required=("text",), handler=judge, kind="tool"))
    draft = "the harness calls tools"
    better = ("one funnel, repeated: registry, pre_tool hooks, policy, handler, post_tool hooks, "
              "and the context window sees the observation")
    agent.script([
        {"thought": "score the first draft", "call": {"tool": "judge.score", "args": {"text": draft}}},
        {"thought": "too thin; write the rewrite down",
         "call": {"tool": "memory.write", "args": {"key": "draft", "value": better}}},
        {"thought": "score the rewrite", "call": {"tool": "judge.score", "args": {"text": better}}},
        {"thought": "accepted", "final": "the second draft cleared the judge"},
    ])
    run = agent.run("write something the judge accepts")
    scores = [c.value["score"] for c in agent.journal if c.tool == "judge.score" and c.ok]
    return [f"trail={run.trail}", f"scores={scores} (bar is 3)",
            f"status={run.status} answer={run.answer}",
            f"journal={json.dumps([c.tool for c in agent.journal])}"]


def ex_map_reduce() -> list:
    """Split a file into blocks, map a tool over each one, reduce the results to a single answer."""
    agent = _scratch()
    (agent.root / "rows.csv").write_text("a,b,c\n1,2,3\n4,5,6\n7,8,9\n", encoding="utf-8")

    def block_sum(block: str) -> dict:
        rows = [r for r in str(block).strip().splitlines() if r]
        return {"rows": len(rows), "sum": sum(int(number) for row in rows for number in row.split(","))}

    def combine(results: list) -> dict:
        return {"blocks": len(results), "rows": sum(r["rows"] for r in results),
                "total": sum(r["sum"] for r in results)}

    agent.registry.add(Tool("map.sum", "add up one block of csv rows", args={"block": str},
                            required=("block",), handler=block_sum, kind="tool"))
    agent.registry.add(Tool("reduce.combine", "fold the mapped results into one", args={"results": list},
                            required=("results",), handler=combine, kind="tool"))
    agent.script([
        {"thought": "read the file", "call": {"tool": "fs.read", "args": {"path": "rows.csv"}}},
        {"thought": "map block 1", "call": {"tool": "map.sum", "args": {"block": "1,2,3"}}},
        {"thought": "map block 2", "call": {"tool": "map.sum", "args": {"block": "4,5,6"}}},
        {"thought": "map block 3", "call": {"tool": "map.sum", "args": {"block": "7,8,9"}}},
        {"thought": "reduce the three maps", "call": {"tool": "reduce.combine",
                                                      "args": {"results": [{"rows": 1, "sum": 6},
                                                                           {"rows": 1, "sum": 15},
                                                                           {"rows": 1, "sum": 24}]}}},
        {"thought": "answered", "final": "3 blocks, 3 rows, total 45"},
    ])
    run = agent.run("add up rows.csv block by block")
    mapped = [c.value for c in agent.journal if c.tool == "map.sum" and c.ok]
    reduced = next((c.value for c in agent.journal if c.tool == "reduce.combine" and c.ok), {})
    return [f"trail={run.trail}", f"map -> {json.dumps(mapped)}", f"reduce -> {json.dumps(reduced)}",
            f"status={run.status} answer={run.answer}"]


def ex_chat_guardrail() -> list:
    """A hook in front of the model: on-topic goals get a system note, off-topic ones never reach it."""
    agent = _scratch()

    def guard(payload: dict) -> dict:
        goal = str((payload.get("args") or {}).get("goal") or "")
        if any(word in goal.lower() for word in ("password", "crypto")):
            return {"block": True, "reason": f"chat guardrail: {goal[:40]!r} is out of scope for this assistant"}
        return {"args": {"messages": [{"role": "system", "content": "Answer only about this codebase."}]
                         + list((payload.get("args") or {}).get("messages") or [])}}

    agent.hook("pre_tool", "model.*", name="guardrail", call=guard)
    agent.script([{"thought": "off topic", "final": "here is a password trick"}])
    off = agent.run("how do I reset someone's password")
    agent.script([{"thought": "on topic", "final": "one funnel: registry, hooks, policy, handler"}])
    on = agent.run("what is the funnel?")
    asked = [c for c in agent.journal if c.tool == "model.scripted" and not c.blocked]
    note = ((asked[-1].args.get("messages") or [{}])[0]).get("content", "") if asked else ""
    return [f"off-topic: status={off.status} error={off.error}",
            f"the model was never asked for it (model.log holds {len(agent.models.log)} calls)",
            f"on-topic: status={on.status} answer={on.answer}",
            f"first message the model did see: {note[:70]!r}"]


def ex_web_page() -> list:
    """browser.open cuts a page down to title, text and links - offline, over a data: URL."""
    page = ("<html><head><title>harness notes</title><style>body{color:red}</style></head>"
            "<body><script>tracking()</script><p>one funnel: registry, hooks, policy, handler</p>"
            "<a href=\"/hooks\">hooks</a><a href=\"/policy\">policy</a></body></html>")
    url = "data:text/html;charset=utf-8," + urllib.parse.quote(page)
    agent = _scratch()
    probe = agent.call("policy.check", {"tool": "browser.open", "args": {"url": url}}).value
    agent.policy.set(allow_network=True)
    seen = agent.call("browser.open", {"url": url}).value
    agent.script([
        {"thought": "open the page", "call": {"tool": "browser.open", "args": {"url": url}}},
        {"thought": "answer from the reduced text", "final": f"{seen['title']}, {len(seen['links'])} links"},
    ])
    run = agent.run("what does the page say?")
    return [f"guardrail first: allowed={probe['allowed']} ({probe['reason']})",
            f"network on -> title={seen['title']!r}", f"text={seen['text'][:58]!r}",
            f"links={json.dumps(seen['links'])}", f"trail={run.trail} status={run.status} answer={run.answer}"]


def ex_coding_agent() -> list:
    """Read the spec, write the module, run the tests, patch the failure, run them again."""
    agent = _scratch()
    (agent.root / "spec.md").write_text("# spec\nadd(a, b) returns the sum of two integers.\n", encoding="utf-8")
    (agent.root / "test_app.py").write_text(
        "from app import add\n\nassert add(2, 3) == 5, 'add must sum'\nprint('tests passed')\n", encoding="utf-8")
    # what a coding agent needs opened, and nothing more: the shell is allowed and approved, deletes still denied
    agent.policy.set(deny=["agent.run", "fs.delete"])
    agent.grant("fs.write", True)
    agent.grant("fs.edit", True)
    agent.grant("shell.run", True)
    # -B: no bytecode cache, or the second run would reuse the patched-away .pyc
    run_cmd = f'"{sys.executable}" -B test_app.py'
    agent.script([
        {"thought": "read the spec", "call": {"tool": "fs.read", "args": {"path": "spec.md"}}},
        {"thought": "write the module, with a slip in it",
         "call": {"tool": "fs.write", "args": {"path": "app.py", "content": "def add(a, b):\n    return a - b\n"}}},
        {"thought": "run the tests", "call": {"tool": "shell.run", "args": {"cmd": run_cmd}}},
        {"thought": "the failure says sum, not difference; patch that one line",
         "call": {"tool": "fs.edit", "args": {"path": "app.py", "find": "return a - b", "replace": "return a + b"}}},
        {"thought": "run them again", "call": {"tool": "shell.run", "args": {"cmd": run_cmd}}},
        {"thought": "green", "final": "tests pass: app.py now sums"},
    ])
    run = agent.run("implement spec.md and make the tests pass")
    tests = [c.value for c in agent.journal if c.tool == "shell.run" and c.ok]
    patched = next((c.value for c in agent.journal if c.tool == "fs.edit" and c.ok), {})
    last_line = lambda t: next((ln for ln in reversed((t["stdout"] or t["stderr"]).strip().splitlines()) if ln), "")
    return [f"guards for this job: deny={json.dumps(agent.policy.deny)} "
            f"granted={json.dumps(sorted(agent.policy.granted))}",
            f"trail={run.trail}",
            f"test runs={json.dumps([{'code': t['code'], 'last line': last_line(t)[:44]} for t in tests])}",
            f"patch={json.dumps({k: patched.get(k) for k in ('replacements', 'matches')})}",
            f"status={run.status} answer={run.answer}"]


EXAMPLES = {"tool-loop": ex_tool_loop, "guardrails": ex_guardrails, "memory-context": ex_memory_context,
            "tasks-scheduler": ex_tasks_scheduler, "skills-prompts": ex_skills_prompts,
            "plugin-pack": ex_plugin_pack, "nested-agents": ex_nested_agents,
            "local-endpoints": ex_local_endpoints, "llm-call": ex_llm_call, "rag": ex_rag,
            "agentic-rag": ex_agentic_rag, "structured-output": ex_structured_output, "thinking": ex_thinking,
            "judge": ex_judge, "map-reduce": ex_map_reduce, "chat-guardrail": ex_chat_guardrail,
            "web-page": ex_web_page, "coding-agent": ex_coding_agent}

def run_examples(names: list | None = None) -> int:
    wanted = names or list(EXAMPLES)
    unknown = [n for n in wanted if n not in EXAMPLES]
    if unknown:
        print(f"no example {', '.join(unknown)}. available: {', '.join(EXAMPLES)}")
        return 2
    for name in wanted:
        print(f"\n=== {name} - {EXAMPLES[name].__doc__.strip().splitlines()[0]}")
        for line in EXAMPLES[name]():
            print(f"  {line}")
    return 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def _demo_agent(root: Path) -> Agent:
    agent = Agent(name="demo", root=root, memory_path=root / ".agent" / "memory.json", budget_tokens=900)
    agent.hook("pre_tool", "fs.*", name="no-secrets",
               call=lambda payload: {"block": True, "reason": "the no-secrets hook refuses private files"}
               if "secret" in str(payload["args"].get("path", "")) else None)
    agent.grant("fs.write")
    agent.script([
        {"thought": "put the note down", "call": {"tool": "fs.write",
                                                  "args": {"path": "notes.md", "content": "# notes\nalpha beta gamma delta"}}},
        {"thought": "read it back", "call": {"tool": "fs.read", "args": {"path": "notes.md"}}},
        {"thought": "peek at something private", "call": {"tool": "fs.read", "args": {"path": "secret.txt"}}},
        {"thought": "try to delete it", "call": {"tool": "fs.delete", "args": {"path": "notes.md"}}},
        {"thought": "remember the count", "memory": {"key": "note-length", "value": "4 words", "kind": "fact"},
         "task": {"subject": "expand the notes"}},
        {"thought": "count the words", "call": {"tool": "memory.search", "args": {"query": "note"}}},
        {"thought": "answer", "final": "notes.md holds 4 words; secret read and delete were both blocked"},
    ])
    return agent


def _print_run(run: Run) -> None:
    print(f"\nstatus: {run.status}   model: {run.model}   {run.seconds}s")
    print(f"trail : {run.trail}")
    for turn in run.turns:
        record = turn.call
        flag = "BLOCK" if record and record.blocked else "ERR  " if record and record.error else "ok   "
        detail = (record.reason or record.error or json.dumps(record.value)[:70]) if record else ""
        print(f"  {turn.i} {flag} {turn.call.tool if record else '(no call)':<16} {turn.thought[:38]:<38} {str(detail)[:70]}")
    print(f"answer: {run.answer}")
    if run.blocked:
        print("blocked:")
        for block in run.blocked:
            print(f"  turn {block['i']} {block['tool']}: {block['reason']}")


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "demo"
    if command == "demo":
        root = Path(tempfile.mkdtemp(prefix="agent-demo-"))
        (root / "secret.txt").write_text("do not read", encoding="utf-8")
        agent = _demo_agent(root)
        print(f"workspace {agent.root}")
        print(f"panel    {json.dumps(agent.panel()['kinds'])} tools across "
              f"{agent.panel()['tools']} names; plugins={agent.plugins['loaded'] or 'none'}")
        run = agent.run("summarise the notes and prove the guardrails bite")
        _print_run(run)
        print(f"\ntasks   {json.dumps(agent.tasks.summary())}")
        print(f"memory  {agent.memory.notes()}")
        print(f"blocks  {json.dumps(agent.policy.blocks, default=str)}")
        print(f"context {json.dumps(agent.context.stats())}")
        print(f"hooks   {json.dumps(agent.hooks.fired[-3:], default=str)[:400]}")
        return 0
    if command == "tools":
        agent = Agent(root=Path.cwd)
        for spec in agent.registry.brief():
            marks = "".join(m for m, on in (("!", spec["danger"]), ("+", spec["mutating"]), ("~", spec["network"])) if on)
            print(f"{spec['kind']:<9}{spec['name']:<20}{marks:<4}{spec['desc'][:70]}")
        print(f"\n{len(agent.registry)} tools, kinds={json.dumps(agent.panel()['kinds'])}")
        return 0
    if command == "skills":
        print(json.dumps(Agent(root=Path.cwd).skills.list(), indent=2))
        return 0
    if command == "panel":
        print(json.dumps(Agent(root=Path.cwd).panel(), indent=2, default=str))
        return 0
    if command == "examples":
        return run_examples(argv[1:])
    if command == "run":
        goal = argv[1] if len(argv) > 1 else "hello"
        model = argv[argv.index("--model") + 1] if "--model" in argv else "model.scripted"
        agent = Agent(root=Path.cwd, model=model, allow_network=model != "model.scripted")
        if model == "model.scripted":
            agent.script([{"thought": "say hi", "call": {"tool": "echo", "args": {"text": goal}}},
                          {"thought": "done", "final": f"echoed {goal!r}"}])
        _print_run(agent.run(goal))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    # plugins do `from agent import tool`; as a script we are __main__, so alias ourselves
    sys.modules.setdefault("agent", sys.modules[__name__])
    raise SystemExit(main())




