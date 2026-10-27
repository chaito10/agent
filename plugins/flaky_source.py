"""A flaky source plus a verification gate — demonstrates fallback and retry.

``demo.fetch`` fails its first two rounds on purpose so a flow can show routing
through ``Fail`` back to itself, bounded by the flow budget.
"""

from harness import Step, step


@step("demo.fetch")
class Fetch(Step):
    def prep(self, memory: dict):
        return memory.get("url")

    def exec(self, work: str):
        if self.round < 2:
            raise RuntimeError(f"connection reset by {work} (attempt {self.round + 1})")
        return "the quick brown fox jumps over the lazy dog"

    def post(self, memory: dict, work, value):
        memory["text"] = value
        return "ok"


@step("demo.gate")
class Gate(Step):
    """Check a condition; route to a repair step instead of trusting the run."""

    def prep(self, memory: dict):
        return memory.get("count")

    def exec(self, work):
        return work is not None and work > 0

    def post(self, memory: dict, work, value):
        return "ok" if value else "bad"


@step("demo.noop")
class Noop(Step):
    def exec(self, work):
        return "noop"

    def post(self, memory: dict, work, value):
        memory["degraded"] = True
        return "ok"
