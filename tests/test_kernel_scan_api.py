from __future__ import annotations

import json
from typing import Any

from veqlib.facade import (
    INITIAL_POLICY_COLD,
    INITIAL_POLICY_WARM_CLONE,
    payload_json_with_initial_policy,
    solve_payload_sequence,
)


def _payload(initial_policy_code: int = INITIAL_POLICY_COLD) -> dict[str, Any]:
    return {
        "case_name": "scan",
        "solver": {
            "initial_policy_code": initial_policy_code,
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

    def adopt_last_solution_as_initial(self) -> None:
        self.adopt_count += 1


def test_payload_json_with_initial_policy_rewrites_deep_copy_only() -> None:
    payload = _payload(INITIAL_POLICY_COLD)

    rewritten = json.loads(payload_json_with_initial_policy(payload, "warm-clone"))

    assert rewritten["solver"]["initial_policy_code"] == INITIAL_POLICY_WARM_CLONE
    assert payload["solver"]["initial_policy_code"] == INITIAL_POLICY_COLD


def test_solve_payload_sequence_uses_warm_clone_continuation_and_scalar_snapshots() -> None:
    solver = RecordingSolver()

    steps = solve_payload_sequence(solver, [_payload(), _payload(), _payload()])

    assert [step.index for step in steps] == [0, 1, 2]
    assert [step.initial_policy_code for step in steps] == [
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_WARM_CLONE,
        INITIAL_POLICY_WARM_CLONE,
    ]
    assert [step.elapsed_ms for step in steps] == [1.5, 2.5, 3.5]
    assert [step.nfev for step in steps] == [10, 11, 12]
    assert solver.adopt_count == 2
    assert [payload["solver"]["initial_policy_code"] for payload in solver.payloads] == [
        INITIAL_POLICY_COLD,
        INITIAL_POLICY_WARM_CLONE,
        INITIAL_POLICY_WARM_CLONE,
    ]
