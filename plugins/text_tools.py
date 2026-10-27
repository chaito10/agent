"""Text tools — the boring plug that a spec can wire without touching the core."""

from harness import Step, step


@step("text.split")
class Split(Step):
    def prep(self, memory: dict):
        return memory.get("text")

    def exec(self, work: str):
        return work.split()

    def post(self, memory: dict, work, value):
        memory["words"] = value
        return "ok" if value else "empty"


@step("text.count")
class Count(Step):
    def prep(self, memory: dict):
        return memory.get("words")

    def exec(self, work: list):
        return len(work)

    def post(self, memory: dict, work, value):
        memory["count"] = value
        return "ok"


@step("text.longest")
class Longest(Step):
    def prep(self, memory: dict):
        return memory.get("words")

    def exec(self, work: list):
        return max(work, key=len)

    def post(self, memory: dict, work, value):
        memory["longest"] = value
        return "Done"
