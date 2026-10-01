from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StepContext:
    v_ms: float
    soc_frac: float
    harvest_left_frac: float
    straight_mode: bool
    lap_frac: float
    straight_progress: float = 0.0


class EnergyPolicy(Protocol):
    name: str

    def command_w(self, ctx: StepContext) -> float:
        """Positive = deploy request, negative = harvest-from-engine request."""


@dataclass
class FlatOut:
    power_w: float = 350e3
    name: str = "flat_out"

    def command_w(self, ctx: StepContext) -> float:
        return self.power_w


@dataclass
class StraightsOnly:
    deploy_w: float = 350e3
    superclip_w: float = 120e3
    name: str = "straights_only"

    def command_w(self, ctx: StepContext) -> float:
        if ctx.straight_mode:
            return self.deploy_w
        return -self.superclip_w if ctx.harvest_left_frac > 0 else 0.0


@dataclass
class SocTarget:
    target: float = 0.6
    gain_w: float = 900e3
    reserve: float = 0.1
    name: str = "soc_target"

    def command_w(self, ctx: StepContext) -> float:
        if ctx.soc_frac < self.reserve:
            return -self.gain_w * (self.reserve - ctx.soc_frac) - 50e3
        cmd = self.gain_w * (ctx.soc_frac - self.target)
        if ctx.straight_mode:
            cmd += 150e3
        return cmd


@dataclass
class ClipEndOfStraight:
    """Deploy early on each straight, then harvest at full throttle before the braking point."""
    deploy_w: float = 350e3
    clip_from: float = 0.7
    clip_w: float = 250e3
    name: str = "clip_end_of_straight"

    def command_w(self, ctx: StepContext) -> float:
        if not ctx.straight_mode:
            return 0.0
        if ctx.straight_progress < self.clip_from or ctx.harvest_left_frac <= 0:
            return self.deploy_w
        return -self.clip_w


def random_policy(rng: np.random.Generator) -> EnergyPolicy:
    kind = rng.integers(4)
    if kind == 0:
        return FlatOut(power_w=rng.uniform(200e3, 350e3))
    if kind == 1:
        return StraightsOnly(deploy_w=rng.uniform(250e3, 350e3),
                             superclip_w=rng.uniform(0.0, 200e3))
    if kind == 2:
        return SocTarget(target=rng.uniform(0.3, 0.8), gain_w=rng.uniform(300e3, 1500e3),
                         reserve=rng.uniform(0.05, 0.2))
    return ClipEndOfStraight(deploy_w=rng.uniform(250e3, 350e3), clip_from=rng.uniform(0.5, 0.9),
                             clip_w=rng.uniform(100e3, 350e3))


@dataclass
class ProfilePolicy:
    """Open-loop command per track point (W; positive deploys, negative harvests)."""
    cmd_w: np.ndarray
    name: str = "profile"

    def command_w(self, ctx: StepContext) -> float:
        i = int(round(ctx.lap_frac * len(self.cmd_w))) % len(self.cmd_w)
        return float(self.cmd_w[i])
