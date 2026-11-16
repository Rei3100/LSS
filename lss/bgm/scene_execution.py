"""Bounded execution pacing for local scene models."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


REST_SECONDS = 60
PROFILES = {"legacy": 0, "scene_paced60": REST_SECONDS}


class ScenePacingError(ValueError):
    pass


def validate_profile(profile: str) -> str:
    if profile not in PROFILES:
        raise ScenePacingError(f"unsupported scene execution profile: {profile}")
    return profile


@dataclass
class SceneCadence:
    profile: str = "legacy"
    rest_seconds: int = field(init=False)
    events: list[dict] = field(default_factory=list)
    sleeper: object = time.sleep
    monotonic: object = time.monotonic

    def __post_init__(self):
        validate_profile(self.profile)
        self.rest_seconds = PROFILES[self.profile]

    def after_success(self, operation: str, synchronize, release_cache):
        if self.rest_seconds <= 0:
            return
        event = {"operation": operation, "rest_seconds": self.rest_seconds,
                 "rest_attempted": True, "synchronize_completed": False,
                 "cache_release_completed": False, "rest_started_monotonic": None,
                 "rest_completed_monotonic": None, "rest_elapsed_seconds": 0.0,
                 "rest_completed": False}
        self.events.append(event)
        try:
            synchronize()
            event["synchronize_completed"] = True
            release_cache()
            event["cache_release_completed"] = True
            started = self.monotonic()
            event["rest_started_monotonic"] = started
            deadline = started + self.rest_seconds
            while True:
                now = self.monotonic()
                event["rest_elapsed_seconds"] = max(0.0, now - started)
                remaining = deadline - now
                if remaining <= 0:
                    event["rest_completed_monotonic"] = now
                    event["rest_completed"] = event["rest_elapsed_seconds"] >= self.rest_seconds
                    break
                try:
                    self.sleeper(remaining)
                except InterruptedError:
                    continue
        finally:
            if event["rest_started_monotonic"] is not None and not event["rest_completed"]:
                event["rest_elapsed_seconds"] = max(
                    0.0, self.monotonic() - event["rest_started_monotonic"])
