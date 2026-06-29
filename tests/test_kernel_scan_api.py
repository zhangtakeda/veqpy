from __future__ import annotations

import json
from typing import Any

from veqlib.facade import (
    CONTINUE_POLICY_WARM,
    CONTINUE_POLICY_WARM_FIXED,
    INITIAL_POLICY_COLD,
    payload_json_with_continue_policy,
    payload_json_with_initial_policy,
    solve_payload_sequence,
)


def _payload(
    initial_policy_code: int = INITIAL_POLICY_COLD,
    continue_policy_code: int = CONTINUE_POLICY_WARM,
) -> dict[str, Any]:
    return {
        "case_name": "scan",
        "solver": {
            "initial_policy_code": initial_policy_code,
            "continue_policy_code": continue_policy_code,
            "method_code": 1,
        },
    }


class RecordingSolver:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.adopt_count = 0

    def set_case_json(self, payload: str) -> None:
        self.payloads.append(json.loads(payload))

    def solve_direct(self) -> tuple[object, ...]:
        index = len(self.payloads) - 1
        return (
            1.5 + index,
            True,
            0,
            10 + index,
            2,
            3,
            4,
            5,
            6,
            1.0e-3,
            2.0e-3,
            [],
            [],
            [],
            [],
        )


def test_payload_json_with_initial_policy_rewrites_deep_copy_only() -> None:
    payload = _payload(INITIAL_POLICY_COLD)

    rewritten = json.loads(payload_json_with_initial_policy(payload, "cold-geometric"))

    assert rewritten["solver"]["initial_policy_code"] == 2
    assert payload["solver"]["initial_policy_code"] == INITIAL_POLICY_COLD


def test_payload_json_with_continue_policy_rewrites_deep_copy_only() -> None:
    payload = _payload()

    rewritten = json.loads(payload_json_with_continue_policy(payload, "warm-fixed"))

    assert rewritten["solver"]["continue_policy_code"] == CONTINUE_POLICY_WARM_FIXED
    assert payload["solver"]["continue_policy_code"] == CONTINUE_POLICY_WARM


def test_solve_payload_sequence_uses_continue_policy_and_scalar_snapshots() -> None:
    solver = RecordingSolver()

    steps = solve_payload_sequence(
        solver,
        [_payload(), _payload(), _payload()],
        continuation_policy="warm-fixed",
    )

    assert [step.index for step in steps] == [0, 1, 2]
    assert [step.initial_policy_code for step in steps] == [
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_COLD,
    ]
    assert [step.continue_policy_code for step in steps] == [
        CONTINUE_POLICY_WARM,
        CONTINUE_POLICY_WARM_FIXED,
        CONTINUE_POLICY_WARM_FIXED,
    ]
    assert [step.elapsed_ms for step in steps] == [1.5, 2.5, 3.5]
    assert [step.nfev for step in steps] == [10, 11, 12]
    assert [payload["solver"]["initial_policy_code"] for payload in solver.payloads] == [
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_COLD,
    ]
    assert [payload["solver"]["continue_policy_code"] for payload in solver.payloads] == [
        CONTINUE_POLICY_WARM,
        CONTINUE_POLICY_WARM_FIXED,
        CONTINUE_POLICY_WARM_FIXED,
    ]
