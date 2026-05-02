from dataclasses import dataclass
from typing import Literal


ProtocolName = Literal["smoke", "fixed_paired", "mirror", "gauntlet_first_loss"]
OpponentName = Literal["random", "heuristic"]


@dataclass(frozen=True)
class ValidationProtocol:
    """Definition for a validation protocol run."""

    name: ProtocolName
    episodes: int
    opponent: OpponentName
    requires_mlflow: bool = False


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

    if name == "gauntlet_first_loss":
        return ValidationProtocol(
            name=name,
            episodes=episodes or 1,
            opponent="heuristic",
            requires_mlflow=True,
        )

    raise ValueError(f"Unsupported validation protocol: {name}")
