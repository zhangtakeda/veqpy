"""
Module: operator.operator_case

Role:
- Provide the legacy OperatorCase name for code that still imports it from
  the operator package.

Public API:
- OperatorCase

Notes:
- The implementation lives in ``veqpy.model.problem.Problem``.
- New code should prefer ``Problem`` when naming user-facing solve inputs.
"""

from __future__ import annotations

from veqpy.model.problem import Problem

OperatorCase = Problem
