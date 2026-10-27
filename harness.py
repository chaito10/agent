"""harness.py — a small, pluggable flow harness inspired by PocketFlow.

One file, zero dependencies. A *step* is a class with prep/exec/post; a *flow*
wires step **names** to step **names**, so all real behaviour comes from plugins
that register those names. Drop a .py file in ./plugins (or pip-install an entry
point in group ``harness.steps``) and the identical spec does something else.

    from harness import Flow, discover

    discover()                                  # ./plugins + installed entry points
    flow = Flow.from_spec({
        "name": "pipeline",
        "start": "fetch",
        "steps": {"fetch": "demo.fetch", "split": "text.split", "top": "text.longest"},
        "edges": {
            "fetch": {"ok": "split", "Fail": "fetch"},   # re-run on error, bounded by budget
            "split": {"ok": "top", "empty": "fetch"},
        },
    })
    report = flow.run({"url": "https://example.com/feed.json"})
    print(report.trail, report.status)          # fetch→fetch→fetch→split→top  ok
    print(report.memory["longest"])             # quick
    report.raise_for_status()

Routing roles (returned by ``post``):
    None or "Done"  stop, success               True  -> "Done",   False -> "Retry"
    "Retry"         run this step again         "*"   -> matches any wired role
    anything else   looked up in flow edges, then in the step's own ``success``

Return None from ``prep`` to end the run early (nothing to do). If ``exec``
raises, ``fail()`` supplies the role instead; a recovered failure keeps its
traceback on the step record and the run stays "ok", while an unwired failure
role ends the run with status "error:<role>" instead of silently continuing.

A run returns a Report instead of raising: status is "ok", "stalled:<role>"
(nothing wired for the role the step returned), "error:<role>" (a failure no
edge caught) or "budget-exhausted" (default 100 rounds). ``report.error`` holds
only the fatal exception; recovered ones live on their step records. Use
``report.raise_for_status()`` when you want the failure to be loud.

Subclasses must not override ``__init__`` — put setup in ``prep``.
"""

import importlib
import importlib.metadata
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

DONE = "Done"
RETRY = "Retry"
ANY = "*"
DEFAULT_PLUGINS_DIR = "plugins"
ENTRY_POINT_GROUP = "harness.steps"


class HarnessError(Exception):
    """Wiring problem: unknown step name, missing start, bad edge source."""


class StepError(Exception):
    """Raised by Report.raise_for_status(); the failure is kept as __cause__."""


# ---------------------------------------------------------------------------
# registry — the plug surface. Names are strings ("llm.summarize"), never classes.
# ---------------------------------------------------------------------------

_registry: dict[str, type["Step"]] = {}
_owner: dict[str, str] = {}  # step name -> what registered it, for idempotent discover()
_overrides: list[tuple[str, str, str]] = []


def register(name: str, cls: type["Step"], *, replace: bool = False) -> type["Step"]:
    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise HarnessError(f"{name!r} is not a Step subclass: {cls!r}")
    previous = _registry.get(name)
    if previous is not None and previous is not cls:
        if not replace:
            raise HarnessError(
                f"step {name!r} already registered by {previous.__module__}; "
                "pass replace=True to override it"
            )
        _overrides.append((name, previous.__module__, cls.__module__))
    cls.step_name = name
    _registry[name] = cls
    return cls


def unregister(name: str) -> None:
    _registry.pop(name, None)
    _owner.pop(name, None)


def get(name: str) -> type["Step"]:
    try:
        return _registry[name]
    except KeyError:
        known = ", ".join(sorted(_registry)) or "<none>"
        raise HarnessError(f"no step registered as {name!r}. Known: {known}") from None


def registry() -> dict[str, type["Step"]]:
    """Names to classes, for tooling that lists what is plugged in."""
    return dict(_registry)


def overrides() -> list[tuple[str, str, str]]:
    """(name, replaced-by, replaced-with) for plugins that shadowed a step."""
    return list(_overrides)


def step(name: str | None = None, *, replace: bool = False) -> Callable:
    """Class decorator that turns a plain class into a named, plugable step."""

    def wrap(cls: type[Step]) -> type[Step]:
        return register(name or cls.__name__, cls, replace=replace)

    return wrap


def _export(target: Any, *, source: str, replace: bool) -> list[str]:
    """Register what an entry point handed back: a class, a list, or a dict.

    A bare class is addressed by the entry point name — the distribution decides
    the plug name. A dict is addressed by its keys.
    """
    if isinstance(target, dict):
        items = list(target.items())
    elif isinstance(target, (list, tuple)):
        items = [(getattr(c, "step_name", None) or c.__name__, c) for c in target]
    else:
        items = [(source, target)]
    names = []
    for key, cls in items:
        register(key, cls, replace=replace)
        _owner[key] = source
        names.append(key)
    return names


def _claim(names: Iterable[str], source: str) -> list[str]:
    claimed = sorted(set(names))
    for name in claimed:
        _owner[name] = source
    return claimed


def discover(
    paths: str | Path | Iterable[str | Path] | None = None,
    *,
    group: str | None = ENTRY_POINT_GROUP,
    replace: bool = True,
) -> dict[str, Any]:
    """Load plugins from directories (flat ``*.py``) and installed entry points.

    Returns {"loaded": [step names], "modules": [plugin names], "errors": {}}.
    A plugin file registers by running the module, so ``@step`` decorators are the
    contract; the directory is put on sys.path first so a plugin can import its own
    siblings. One broken plugin is reported in ``errors`` and never kills the harness.
    """
    if paths is None:
        paths = [DEFAULT_PLUGINS_DIR]
    elif isinstance(paths, (str, Path)):
        paths = [paths]

    loaded: list[str] = []
    modules: list[str] = []
    errors: dict[str, BaseException] = {}

    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        directory = str(root.resolve())
        if directory not in sys.path:
            sys.path.append(directory)
        for file in sorted(root.glob("*.py")):
            if file.name.startswith("_"):
                continue
            before = set(_registry)
            try:
                importlib.import_module(file.stem)
            except BaseException as exc:  # noqa: BLE001 - isolate plugin failures
                errors[file.stem] = exc
                continue
            fresh = set(_registry) - before
            # an already-imported plugin registers nothing new, so fall back to
            # what it claimed last time to keep discover() idempotent.
            mine = _claim(fresh, file.stem) if fresh else [n for n, o in _owner.items() if o == file.stem]
            modules.append(file.stem)
            loaded += mine

    if group:
        for ep in importlib.metadata.entry_points(group=group):
            try:
                loaded += _export(ep.load(), source=ep.name, replace=replace)
                modules.append(ep.name)
            except BaseException as exc:  # noqa: BLE001
                errors[ep.name] = exc

    return {"loaded": sorted(set(loaded)), "modules": modules, "errors": errors}


# ---------------------------------------------------------------------------
# Step — one round is prep -> exec -> post, or prep -> exec(raises) -> fail
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StepRecord:
    name: str
    round: int
    role: str | None
    value: Any = None
    error: str | None = None
    seconds: float = 0.0


class Step:
    """A unit of work. Override prep/exec/post, plus fail for error routing."""

    step_name: str | None = None
    success: dict[str, Any] = {}  # local routing; flow edges win where both exist

    def __init__(self, memory: dict | None = None, name: str | None = None):
        self.memory = memory if memory is not None else {}
        self.name = name or self.step_name or type(self).__name__
        self.round = 0

    def prep(self, memory: dict) -> Any:
        """Return the work for exec(), or None to end the run."""
        return memory

    def exec(self, work: Any) -> Any:
        """The work itself. No memory access here — it keeps steps testable."""
        return work

    def post(self, memory: dict, work: Any, value: Any) -> Any:
        """Write results into memory, return the role that decides what is next."""
        return DONE

    def fail(self, memory: dict, work: Any, error: BaseException) -> Any:
        return "Fail"

    def _round(self) -> tuple[str | None, Any, BaseException | None]:
        work = self.prep(self.memory)
        if work is None:
            return None, None, None
        try:
            value = self.exec(work)
            return self._norm(self.post(self.memory, work, value)), value, None
        except BaseException as exc:  # noqa: BLE001 - capture, then route
            if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                raise
            return self._norm(self.fail(self.memory, work, exc)), None, exc

    @staticmethod
    def _norm(role: Any) -> str | None:
        if role is True:
            return DONE
        if role is False:
            return RETRY
        return None if role is None else str(role)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


# ---------------------------------------------------------------------------
# Flow — names resolved lazily at each hop, so edges may point at steps that
# are only plugged in later, and so flows nest inside flows for free.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Report:
    name: str
    status: str
    memory: dict = field(repr=False)
    steps: list[StepRecord] = field(default_factory=list)
    error: BaseException | None = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def trail(self) -> str:
        return "→".join(s.name for s in self.steps)

    @property
    def result(self) -> Any:
        """Value the last step produced — for a flow, the useful output."""
        return self.steps[-1].value if self.steps else None

    def seconds(self) -> float:
        return sum(s.seconds for s in self.steps)

    def __str__(self) -> str:
        return f"[{self.name}] {self.status} via {self.trail or '(no steps)'}"

    def raise_for_status(self) -> "Report":
        if not self.ok:
            raise StepError(str(self)) from self.error
        return self


class Flow(Step):
    """A named graph of steps. Being a Step itself is what makes flows composable."""

    def __init__(
        self,
        steps: dict[str, "Step | type[Step] | str"] | None = None,
        edges: dict[str, dict[str, "str | Step"]] | None = None,
        *,
        start: str | None = None,
        name: str | None = None,
        memory: dict | None = None,
        budget: int = 100,
    ):
        """`budget` caps rounds per run (default 100) so a Retry loop cannot spin forever."""
        super().__init__(memory=memory, name=name or "flow")
        self.steps: dict[str, Step] = {}
        self.edges: dict[str, dict[str, Any]] = {k: dict(v) for k, v in (edges or {}).items()}
        self.aliases: dict[str, str] = {}
        self.budget = budget
        self.start = start
        for key, value in (steps or {}).items():
            self.add(value, name=key, start=start is None)

    # -- construction ------------------------------------------------------
    def add(self, node: "Step | type[Step] | str", *, name: str | None = None, start: bool = False) -> Step:
        if isinstance(node, str):
            node = get(node)(memory=self.memory, name=name or node)
        elif isinstance(node, type) and issubclass(node, Step):
            node = node(memory=self.memory)
        if not isinstance(node, Step):
            raise HarnessError(f"expected a Step, Step class or step name, got {node!r}")
        key = name or node.name
        if key in self.steps:
            raise HarnessError(f"step {key!r} is already in flow {self.name!r}")
        node.name = key
        self.steps[key] = node
        if start:
            self.start = key
        return node

    def wire(self, source: str, role: str, target: "str | Step") -> "Flow":
        self.edges.setdefault(source, {})[role] = target
        return self

    def alias(self, name: str, key: str) -> "Flow":
        """Let an edge say a registry name for a step that lives under a spec key."""
        self.aliases[name] = key
        return self

    def get_step(self, name: str) -> Step | None:
        return self.steps.get(self.aliases.get(name, name))

    @classmethod
    def from_spec(cls, spec: dict, *, memory: dict | None = None) -> "Flow":
        """Build from a plain dict — the shape a plugin pack or a config file emits."""
        flow = cls(
            name=spec.get("name", "flow"),
            start=spec.get("start"),
            budget=spec.get("budget", 100),
            memory=memory if memory is not None else {},
        )
        if spec.get("plugins"):
            discover(spec["plugins"])
        for key, ref in (spec.get("steps") or {}).items():
            node = _build(ref, flow.memory, key)
            flow.add(node, name=key, start=flow.start is None)
            if isinstance(ref, str):
                flow.alias(ref, key)
        if spec.get("start"):
            flow.start = spec["start"]
        if flow.get_step(flow.start) is None:
            raise HarnessError(
                f"flow {flow.name!r} starts at {flow.start!r}, which is not one of its steps "
                f"({', '.join(flow.steps) or 'none declared'})"
            )
        for key, mapping in (spec.get("edges") or {}).items():
            if flow.get_step(key) is None:
                raise HarnessError(f"edge source {key!r} is not a step of flow {flow.name!r}")
            for role, target in mapping.items():
                if isinstance(target, (dict, type, Step)):
                    target = _build(target, flow.memory, f"{key}:{role}")
                flow.wire(key, role, target)
        return flow

    # -- routing -----------------------------------------------------------
    def successors(self, node: Step) -> dict[str, Any]:
        return {**node.success, **self.edges.get(node.name, {})}

    def _resolve(self, target: "str | Step") -> Step:
        if isinstance(target, Step):
            target.memory = self.memory
            return target
        key = self.aliases.get(target, target)
        node = self.steps.get(key)
        if node is None:
            node = get(target)(memory=self.memory, name=key)
            self.steps[key] = node
        return node

    # -- execution ---------------------------------------------------------
    def run(self, memory: dict | None = None, *, trace: Callable[[StepRecord], None] | None = None) -> Report:
        if memory is not None:
            self.memory = memory
        elif self.memory is None:
            self.memory = {}
        for node in self.steps.values():
            node.memory = self.memory
            node.round = 0  # rounds, and any step that branches on them, are per run

        report = Report(name=self.name, status="ok", memory=self.memory)
        if not self.start:
            raise HarnessError(f"flow {self.name!r} has no start step")
        node = self._resolve(self.start)

        while node is not None:
            if len(report.steps) >= self.budget:
                report.status = "budget-exhausted"
                break
            started = time.monotonic()
            role, value, error = node._round()
            record = StepRecord(
                name=node.name,
                round=node.round,
                role=role,
                value=value,
                error=None if error is None else "".join(traceback.format_exception(error)).strip(),
                seconds=time.monotonic() - started,
            )
            node.round += 1
            report.steps.append(record)
            if trace:
                trace(record)

            if role is None or role == DONE:
                break
            if role == RETRY:
                continue
            mapping = self.successors(node)
            target = mapping.get(role, mapping.get(ANY))
            if target is None:
                if error is not None:
                    report.error = error
                    report.status = f"error:{role}"
                else:
                    report.status = f"stalled:{role}"
                break
            node = self._resolve(target)
        return report

    # -- a flow used as a step --------------------------------------------
    def exec(self, work: Any) -> Any:
        return self.run(self.memory)

    def post(self, memory: dict, work: Any, value: Any) -> Any:
        return DONE if value.ok else value.status

    def __repr__(self) -> str:
        return f"<Flow {self.name} steps={list(self.steps)} start={self.start!r}>"


def _build(ref: Any, memory: dict, key: str) -> "Step | Flow":
    """Resolve a spec reference: registry name, class, instance, or sub-flow dict."""
    if isinstance(ref, Step):
        return ref
    if isinstance(ref, type) and issubclass(ref, Step):
        return ref(memory=memory)
    if isinstance(ref, dict):
        return Flow.from_spec(ref, memory=memory)
    return get(ref)(memory=memory, name=key)
