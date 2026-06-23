from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Protocol

from .options import initial_policy_code

PayloadLike = str | Mapping[str, Any]


class PayloadSequenceSolver(Protocol):
    def set_case_json(self, payload: str) -> None: ...

    def solve_direct(self) -> Any: ...


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
) -> list[PayloadSequenceStep]:
    """Solve an ordered same-topology payload sequence with one mutable solver.

    By default, the first payload is solved from the canonical cold policy and
    subsequent payloads use VEQlib's warm-clone policy, which copies the previous
    accepted solution into the next runtime case. Passing ``None`` for either
    policy preserves that payload's existing ``solver.initial_policy_code``.
    """

    steps: list[PayloadSequenceStep] = []
    for index, payload in enumerate(payloads):
        requested_policy = first_policy if index == 0 else continuation_policy
        payload_json, policy_code = _payload_json_and_policy(payload, requested_policy)
        solver.set_case_json(payload_json)
        steps.append(_step_from_result(index, policy_code, solver.solve_direct()))
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
