import dataclasses


@dataclasses.dataclass
class MyScriptExecutionResult:
    stdout: str | None
    stderr: str | None
    exit_code: int
    execution_time_ms: float
    success: bool
