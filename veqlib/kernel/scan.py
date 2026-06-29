from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol

from .options import INITIAL_POLICY_WARM_CLONE, initial_policy_code

PayloadLike = str | Mapping[str, Any]


class PayloadSequenceSolver(Protocol):
    def set_case_json(self, payload: str) -> None: ...

    def solve_direct(self) -> Any: ...

    def adopt_last_solution_as_initial(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PayloadSequenceStep:
    """Scalar result summary for one VEQlib payload-sequence solve.

    ``KernelSolver.solve_direct()`` returns NumPy views owned by the mutable C++
    solver workspace. A continuation scan overwrites that workspace at the next
    point, so the sequence helper records scalar copies only.
    """

    index: int
    initial_policy_code: int
    elapsed_ms: float
    success: bool
    info: int
    nfev: int
    njev: int
    callbacks: int
    jacobian_component_evaluations: int
    jvp_evaluations: int
    linear_iterations: int
    raw_norm: float
    scaled_norm: float


def payload_json_with_initial_policy(payload: PayloadLike, policy: str | int) -> str:
    """Return a compact JSON payload with ``solver.initial_policy_code`` rewritten.

    The input mapping is deep-copied before modification. This keeps scan setup
    side-effect-free while allowing callers to reuse a case payload template for
    cold and warm-clone runs.
    """

    data = _payload_object(payload)
    solver_config = data.get("solver")
    if not isinstance(solver_config, MutableMapping):
        raise ValueError("VEQlib case payload must contain a solver object")
    solver_config["initial_policy_code"] = initial_policy_code(policy)
    return _payload_json(data)


def solve_payload_sequence(
    solver: PayloadSequenceSolver,
    payloads: Iterable[PayloadLike],
    *,
    first_policy: str | int | None = "cold",
    continuation_policy: str | int | None = "warm-clone",
    adopt_solution_for_continuation: bool = True,
) -> list[PayloadSequenceStep]:
    """Solve an ordered same-topology payload sequence with one mutable solver.

    By default, the first payload is solved from the canonical cold policy and
    subsequent payloads use VEQlib's warm-clone policy, which copies the previous
    accepted solution into the next runtime case. The helper explicitly adopts
    each accepted result before a following warm-clone point so a cold first
    point can still seed the continuation path. Passing ``None`` for either
    policy preserves that payload's existing ``solver.initial_policy_code``.
    """

    payload_items = list(payloads)
    steps: list[PayloadSequenceStep] = []
    for index, payload in enumerate(payload_items):
        requested_policy = first_policy if index == 0 else continuation_policy
        payload_json, policy_code = _payload_json_and_policy(payload, requested_policy)
        solver.set_case_json(payload_json)
        step = _step_from_result(index, policy_code, solver.solve_direct())
        steps.append(step)
        if (
            adopt_solution_for_continuation
            and step.success
            and _next_payload_uses_warm_clone(
                payload_items,
                index + 1,
                first_policy=first_policy,
                continuation_policy=continuation_policy,
            )
        ):
            solver.adopt_last_solution_as_initial()
    return steps


def _payload_json_and_policy(
    payload: PayloadLike,
    policy: str | int | None,
) -> tuple[str, int]:
    if policy is not None:
        policy_code = initial_policy_code(policy)
        return payload_json_with_initial_policy(payload, policy_code), policy_code

    data = _payload_object(payload)
    solver_config = data.get("solver")
    if not isinstance(solver_config, Mapping):
        raise ValueError("VEQlib case payload must contain a solver object")
    try:
        policy_code = initial_policy_code(solver_config["initial_policy_code"])
    except KeyError as exc:
        raise ValueError("VEQlib solver payload must contain initial_policy_code") from exc
    return _payload_json(data), policy_code


def _next_payload_uses_warm_clone(
    payloads: list[PayloadLike],
    index: int,
    *,
    first_policy: str | int | None,
    continuation_policy: str | int | None,
) -> bool:
    if index >= len(payloads):
        return False
    requested_policy = first_policy if index == 0 else continuation_policy
    _payload_json, policy_code = _payload_json_and_policy(payloads[index], requested_policy)
    return policy_code == INITIAL_POLICY_WARM_CLONE


def _payload_object(payload: PayloadLike) -> dict[str, Any]:
    if isinstance(payload, str):
        data = json.loads(payload)
    elif isinstance(payload, Mapping):
        data = copy.deepcopy(dict(payload))
    else:
        raise TypeError("payload must be a JSON string or mapping")
    if not isinstance(data, dict):
        raise ValueError("VEQlib case payload must be a JSON object")
    return data


def _payload_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _step_from_result(index: int, policy_code: int, result: Any) -> PayloadSequenceStep:
    return PayloadSequenceStep(
        index=index,
        initial_policy_code=policy_code,
        elapsed_ms=float(result[0]),
        success=bool(result[1]),
        info=int(result[2]),
        nfev=int(result[3]),
        njev=int(result[4]),
        callbacks=int(result[5]),
        jacobian_component_evaluations=int(result[6]),
        jvp_evaluations=int(result[7]),
        linear_iterations=int(result[8]),
        raw_norm=float(result[9]),
        scaled_norm=float(result[10]),
    )
