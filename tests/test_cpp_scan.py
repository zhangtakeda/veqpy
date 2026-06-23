from __future__ import annotations

import json
from typing import Any

import pytest

from veqpy.cpp import (
    INITIAL_POLICY_COLD,
    INITIAL_POLICY_WARM_CLONE,
    payload_json_with_initial_policy,
    solve_payload_sequence,
)


class FakeSequenceSolver:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.calls = 0

    def set_case_json(self, payload: str) -> None:
        self.payloads.append(json.loads(payload))

    def solve_direct(self) -> tuple[float, bool, int, int, int, int, int, int, int, float, float]:
        self.calls += 1
        return (
            0.25 * self.calls,
            True,
            1,
            10 + self.calls,
            self.calls,
            20 + self.calls,
            30 + self.calls,
            40 + self.calls,
            50 + self.calls,
            1.0e-12 * self.calls,
            2.0e-12 * self.calls,
        )


def _payload(policy: int = INITIAL_POLICY_COLD, ip: float = 1.0) -> dict[str, Any]:
    return {
        "case_name": "pf-scan-test",
        "constraints": {"scaled_Ip": ip},
        "solver": {
            "method_code": 1,
            "initial_policy_code": policy,
        },
    }


def test_payload_json_with_initial_policy_rewrites_copy_only() -> None:
    payload = _payload(INITIAL_POLICY_COLD)

    rewritten = json.loads(payload_json_with_initial_policy(payload, "warm-clone"))

    assert payload["solver"]["initial_policy_code"] == INITIAL_POLICY_COLD
    assert rewritten["solver"]["initial_policy_code"] == INITIAL_POLICY_WARM_CLONE
    assert rewritten["constraints"]["scaled_Ip"] == 1.0


def test_solve_payload_sequence_uses_cold_then_warm_clone_by_default() -> None:
    fake = FakeSequenceSolver()
    payloads = [_payload(ip=1.0), _payload(ip=1.1), _payload(ip=1.2)]

    steps = solve_payload_sequence(fake, payloads)

    assert [item["solver"]["initial_policy_code"] for item in fake.payloads] == [
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_WARM_CLONE,
        INITIAL_POLICY_WARM_CLONE,
    ]
    assert [step.index for step in steps] == [0, 1, 2]
    assert [step.nfev for step in steps] == [11, 12, 13]
    assert [step.callbacks for step in steps] == [21, 22, 23]
    assert steps[-1].raw_norm == pytest.approx(3.0e-12)


def test_solve_payload_sequence_can_preserve_payload_policy() -> None:
    fake = FakeSequenceSolver()
    payloads = [_payload(INITIAL_POLICY_WARM_CLONE), _payload(INITIAL_POLICY_COLD)]

    steps = solve_payload_sequence(
        fake,
        payloads,
        first_policy=None,
        continuation_policy=None,
    )

    assert [step.initial_policy_code for step in steps] == [
        INITIAL_POLICY_WARM_CLONE,
        INITIAL_POLICY_COLD,
    ]
    assert [item["solver"]["initial_policy_code"] for item in fake.payloads] == [
        INITIAL_POLICY_WARM_CLONE,
        INITIAL_POLICY_COLD,
    ]


def test_payload_json_with_initial_policy_requires_solver_object() -> None:
    with pytest.raises(ValueError, match="solver object"):
        payload_json_with_initial_policy({"constraints": {"scaled_Ip": 1.0}}, "cold")
