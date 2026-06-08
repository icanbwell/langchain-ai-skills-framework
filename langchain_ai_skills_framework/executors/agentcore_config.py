import dataclasses


@dataclasses.dataclass(frozen=True)
class AgentCoreConfig:
    harness_id: str
    alias_id: str = "TSTALIASID"
    region: str = "us-east-1"
    timeout: int = 300
