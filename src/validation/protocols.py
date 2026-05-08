from dataclasses import dataclass, field
from typing import List, Literal


ProtocolName = Literal["smoke", "fixed_paired", "mirror", "benchmark"]
OpponentName = Literal["random", "random_no_switch", "heuristic"]


@dataclass(frozen=True)
class ValidationProtocol:
    """Definition for a validation protocol run."""

    name: ProtocolName
    episodes: int
    opponent: OpponentName
    requires_mlflow: bool = False
    opponents: List[OpponentName] = field(default_factory=list)
    episodes_per_opponent: int = 50


def get_protocol(name: ProtocolName, episodes: int | None = None) -> ValidationProtocol:
    """Return a protocol definition with conservative defaults."""
    if name == "smoke":
        return ValidationProtocol(
            name=name,
            episodes=episodes or 3,
            opponent="random",
            requires_mlflow=False,
        )

    if name == "fixed_paired":
        return ValidationProtocol(
            name=name,
            episodes=episodes or 40,
            opponent="random",
            requires_mlflow=True,
        )

    if name == "mirror":
        return ValidationProtocol(
            name=name,
            episodes=episodes or 40,
            opponent="heuristic",
            requires_mlflow=True,
        )

    if name == "benchmark":
        opponents: List[OpponentName] = ["random", "random_no_switch", "heuristic"]
        eps_per = episodes or 50
        return ValidationProtocol(
            name=name,
            episodes=len(opponents) * eps_per,
            opponent="random",
            requires_mlflow=True,
            opponents=opponents,
            episodes_per_opponent=eps_per,
        )

    raise ValueError(f"Unsupported validation protocol: {name}")
