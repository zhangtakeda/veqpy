from __future__ import annotations

import os

import numpy as np
from _helpers import MU0, pf_reference_profiles
from numpy.testing import assert_allclose

from veqpy import (
    Kernel,
    KernelBoundary,
    KernelConfig,
    KernelRecipe,
    KernelSource,
    KernelTopology,
)

os.environ.setdefault("OPENMDAO_REPORTS", "0")


def _topology() -> KernelTopology:
    return KernelTopology(
        h_count=2,
        v_count=0,
        kappa_count=2,
        psin_count=3,
        F_count=0,
        c_counts=(),
        s_counts=(2,),
        Nr=8,
        Nt=8,
        route="PF",
        coordinate="psin",
        nodes="uniform",
        constraint="ip",
        sample_count=9,
    )


def _boundary() -> KernelBoundary:
    return KernelBoundary(
        a=0.5,
        R0=1.0,
        Z0=0.0,
        B0=3.0,
        ka=1.7,
        s_offsets=(float(np.arcsin(0.2)),),
    )


def _source() -> KernelSource:
    psin = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    FF_psin, scaled_P_psin = pf_reference_profiles(psin)
    return KernelSource(
        P_psin=scaled_P_psin / MU0,
        FF_psin=FF_psin,
        Ip=3.0e6,
    )


def _config() -> KernelConfig:
    return KernelConfig(
        method="powell",
        max_residual=1.0e-10,
        accepted_residual_floor=1.0e-9,
        continuation="warm",
        norm="none",
    )


def _kernel() -> Kernel:
    kernel = Kernel(
        topology=_topology(),
        recipe=KernelRecipe(backend="numba"),
        config=_config(),
    )
    kernel.prepare()
    return kernel


def _source_with_profiles(
    template: KernelSource,
    P_psin: np.ndarray,
    FF_psin: np.ndarray,
) -> KernelSource:
    return KernelSource(
        P_psin=P_psin,
        FF_psin=FF_psin,
        p0=template.p0,
        Ip=template.Ip,
        beta=template.beta,
        case_name=template.case_name,
    )


def _equilibrium_output(kernel: Kernel, result) -> np.ndarray:
    equilibrium = kernel.build_equilibrium(result.x)
    return np.concatenate(
        [
            equilibrium.psin,
            equilibrium.q,
            equilibrium.jtor,
            equilibrium.Phi_r,
            np.asarray([equilibrium.Ip]),
        ]
    )


def _directions() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.linspace(-1.0, 1.0, 9) * 1.0e4,
        np.linspace(1.0, -0.5, 9) * 1.0e-2,
    )


def _make_veq_component(om, kernel, boundary, source, output_size):
    class VEQComponent(om.ExplicitComponent):
        def setup(self) -> None:
            self.add_input("P_psin", val=source.P_psin)
            self.add_input("FF_psin", val=source.FF_psin)
            self.add_output("equilibrium", shape=output_size)
            self._base_result = None

        def setup_partials(self) -> None:
            self.declare_partials("*", "*")

        def compute(self, inputs, outputs) -> None:
            runtime_source = _source_with_profiles(
                source,
                np.asarray(inputs["P_psin"]),
                np.asarray(inputs["FF_psin"]),
            )
            self._base_result = kernel.solve(boundary, runtime_source)
            outputs["equilibrium"] = _equilibrium_output(kernel, self._base_result)

        def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode) -> None:
            if mode != "fwd":
                raise NotImplementedError("the initial VEQ solve-map contract is forward-only")
            runtime_source = _source_with_profiles(
                source,
                np.asarray(inputs["P_psin"]),
                np.asarray(inputs["FF_psin"]),
            )
            tangent = {}
            if "P_psin" in d_inputs:
                tangent["P_psin"] = np.asarray(d_inputs["P_psin"])
            if "FF_psin" in d_inputs:
                tangent["FF_psin"] = np.asarray(d_inputs["FF_psin"])
            d_outputs["equilibrium"] += kernel.solve_jvp(
                boundary,
                runtime_source,
                source_tangent=tangent,
                output=_equilibrium_output,
                base_result=self._base_result,
                relative_step=1.0e-5,
            )

    return VEQComponent


def _make_response_component(om, weights):
    class Response(om.ExplicitComponent):
        def setup(self) -> None:
            self.add_input("equilibrium", shape=weights.size)
            self.add_output("metric")

        def setup_partials(self) -> None:
            self.declare_partials("metric", "equilibrium", val=weights)

        def compute(self, inputs, outputs) -> None:
            outputs["metric"] = np.dot(weights, inputs["equilibrium"])

    return Response


def test_solve_jvp_covers_equilibrium_materialization_and_preserves_kernel_state() -> None:
    kernel = _kernel()
    boundary = _boundary()
    source = _source()
    base = kernel.solve(boundary, source)
    history = list(kernel.history)
    dp, dff = _directions()

    actual = kernel.solve_jvp(
        boundary,
        source,
        source_tangent={"P_psin": dp, "FF_psin": dff},
        output=_equilibrium_output,
        base_result=base,
        relative_step=1.0e-5,
    )

    relative_rate = max(
        np.max(np.abs(dp)) / max(1.0, np.max(np.abs(source.P_psin))),
        np.max(np.abs(dff)) / max(1.0, np.max(np.abs(source.FF_psin))),
    )
    reference_step = 0.5e-5 / relative_rate
    plus_source = _source_with_profiles(
        source,
        source.P_psin + reference_step * dp,
        source.FF_psin + reference_step * dff,
    )
    minus_source = _source_with_profiles(
        source,
        source.P_psin - reference_step * dp,
        source.FF_psin - reference_step * dff,
    )
    plus = kernel.solve(boundary, plus_source, x0=base.x)
    plus_output = _equilibrium_output(kernel, plus)
    minus = kernel.solve(boundary, minus_source, x0=base.x)
    minus_output = _equilibrium_output(kernel, minus)
    expected = (plus_output - minus_output) / (2.0 * reference_step)

    assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)
    assert kernel.result is minus
    assert kernel.history[:-2] == history

    # Re-establish the original public state and check the JVP itself did not
    # append derivative solves to history or replace the cached base result.
    kernel.clear()
    base = kernel.solve(boundary, source)
    size = len(kernel.history)
    kernel.solve_jvp(
        boundary,
        source,
        source_tangent={"P_psin": dp, "FF_psin": dff},
        base_result=base,
    )
    assert len(kernel.history) == size
    assert kernel.result is base
    zero = kernel.solve_jvp(
        boundary,
        source,
        source_tangent={"P_psin": np.zeros_like(source.P_psin)},
        output=_equilibrium_output,
        base_result=base,
    )
    assert zero.shape == (4 * _topology().Nr + 1,)
    assert_allclose(zero, 0.0)
    kernel.close()


def test_solve_jvp_supports_boundary_directions_and_rejects_inactive_fields() -> None:
    kernel = _kernel()
    boundary = _boundary()
    source = _source()
    base = kernel.solve(boundary, source)
    direction = {"a": 0.1, "B0": -0.2}

    actual = kernel.solve_jvp(
        boundary,
        source,
        boundary_tangent=direction,
        base_result=base,
        relative_step=1.0e-5,
    )
    step = 2.0e-5
    plus_boundary = KernelBoundary(
        a=boundary.a + step * direction["a"],
        R0=boundary.R0,
        Z0=boundary.Z0,
        B0=boundary.B0 + step * direction["B0"],
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )
    minus_boundary = KernelBoundary(
        a=boundary.a - step * direction["a"],
        R0=boundary.R0,
        Z0=boundary.Z0,
        B0=boundary.B0 - step * direction["B0"],
        ka=boundary.ka,
        c_offsets=boundary.c_offsets,
        s_offsets=boundary.s_offsets,
    )
    plus = kernel.solve(plus_boundary, source, x0=base.x)
    minus = kernel.solve(minus_boundary, source, x0=base.x)
    expected = (plus.x - minus.x) / (2.0 * step)

    assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-7)
    with __import__("pytest").raises(KeyError, match="inactive"):
        kernel.solve_jvp(
            boundary,
            source,
            source_tangent={"q": np.ones_like(source.P_psin)},
            base_result=base,
        )
    kernel.close()


def test_solve_jvp_rejects_cxx_until_native_continuation_is_transactional() -> None:
    kernel = Kernel(topology=_topology(), recipe=KernelRecipe(backend="cxx"))

    with __import__("pytest").raises(NotImplementedError, match="transactionally restored"):
        kernel.solve_jvp(
            _boundary(),
            _source(),
            source_tangent={"P_psin": np.ones(9)},
        )
    kernel.close()


def test_openmdao_propagates_veq_solve_jvp_through_a_downstream_response() -> None:
    om = __import__("pytest").importorskip("openmdao.api")

    boundary = _boundary()
    source = _source()
    kernel = _kernel()
    output_size = 4 * _topology().Nr + 1
    weights = np.linspace(0.5, 1.5, output_size)

    VEQComponent = _make_veq_component(om, kernel, boundary, source, output_size)
    Response = _make_response_component(om, weights)

    problem = om.Problem(reports=False)
    independent = om.IndepVarComp()
    independent.add_output("P_psin", val=source.P_psin)
    independent.add_output("FF_psin", val=source.FF_psin)
    problem.model.add_subsystem("input", independent, promotes_outputs=["*"])
    problem.model.add_subsystem("veq", VEQComponent(), promotes_inputs=["P_psin", "FF_psin"])
    problem.model.add_subsystem("response", Response())
    problem.model.connect("veq.equilibrium", "response.equilibrium")
    problem.setup(mode="fwd")
    problem.run_model()

    dp, dff = _directions()
    propagated = problem.compute_jacvec_product(
        of=["response.metric"],
        wrt=["P_psin", "FF_psin"],
        mode="fwd",
        seed={"P_psin": dp, "FF_psin": dff},
        linearize=True,
    )["response.metric"]
    expected_equilibrium = kernel.solve_jvp(
        boundary,
        source,
        source_tangent={"P_psin": dp, "FF_psin": dff},
        output=_equilibrium_output,
        relative_step=1.0e-5,
    )
    totals = problem.compute_totals(
        of=["response.metric"],
        wrt=["P_psin", "FF_psin"],
        return_format="array",
    )

    assert_allclose(propagated, np.dot(weights, expected_equilibrium), rtol=2.0e-6)
    assert_allclose(
        totals @ np.concatenate([dp, dff]),
        propagated,
        rtol=2.0e-6,
    )
    kernel.close()


def test_openmdao_solves_coupled_total_jvp_with_veq_inside_a_cycle() -> None:
    om = __import__("pytest").importorskip("openmdao.api")

    boundary = _boundary()
    source = _source()
    kernel = _kernel()
    output_size = 4 * _topology().Nr + 1
    weights = np.linspace(0.5, 1.5, output_size)
    feedback_shape = np.linspace(0.0, 1.0, source.P_psin.size) * 1.0e4
    VEQComponent = _make_veq_component(om, kernel, boundary, source, output_size)
    Response = _make_response_component(om, weights)

    class ComposePressure(om.ExplicitComponent):
        def setup(self) -> None:
            self.add_input("pbase", val=source.P_psin)
            self.add_input("feedback", val=0.0)
            self.add_output("P_psin", val=source.P_psin)

        def setup_partials(self) -> None:
            indices = np.arange(source.P_psin.size)
            self.declare_partials("P_psin", "pbase", rows=indices, cols=indices, val=1.0)
            self.declare_partials("P_psin", "feedback", val=feedback_shape[:, None])

        def compute(self, inputs, outputs) -> None:
            outputs["P_psin"] = inputs["pbase"] + inputs["feedback"] * feedback_shape

    class EquilibriumFeedback(om.ExplicitComponent):
        def setup(self) -> None:
            self.add_input("equilibrium", shape=output_size)
            self.add_output("feedback")

        def setup_partials(self) -> None:
            self.declare_partials(
                "feedback",
                "equilibrium",
                rows=np.asarray([0]),
                cols=np.asarray([_topology().Nr]),
                val=2.0e-2,
            )

        def compute(self, inputs, outputs) -> None:
            outputs["feedback"] = 2.0e-2 * inputs["equilibrium"][_topology().Nr]

    problem = om.Problem(reports=False)
    independent = om.IndepVarComp()
    independent.add_output("pbase", val=source.P_psin)
    independent.add_output("FF_psin", val=source.FF_psin)
    problem.model.add_subsystem("input", independent, promotes_outputs=["*"])
    problem.model.add_subsystem("compose", ComposePressure(), promotes_inputs=["pbase"])
    problem.model.add_subsystem("veq", VEQComponent(), promotes_inputs=["FF_psin"])
    problem.model.add_subsystem("feedback", EquilibriumFeedback())
    problem.model.add_subsystem("response", Response())
    problem.model.connect("compose.P_psin", "veq.P_psin")
    problem.model.connect("veq.equilibrium", "feedback.equilibrium")
    problem.model.connect("feedback.feedback", "compose.feedback")
    problem.model.connect("veq.equilibrium", "response.equilibrium")
    problem.model.nonlinear_solver = om.NonlinearBlockGS(
        maxiter=20,
        atol=1.0e-11,
        rtol=1.0e-11,
        iprint=0,
    )
    problem.model.linear_solver = om.LinearBlockGS(
        maxiter=50,
        atol=1.0e-11,
        rtol=1.0e-11,
        iprint=0,
    )
    problem.setup(mode="fwd")
    problem.run_model()

    dp, dff = _directions()
    propagated = problem.compute_jacvec_product(
        of=["response.metric"],
        wrt=["pbase", "FF_psin"],
        mode="fwd",
        seed={"pbase": dp, "FF_psin": dff},
        linearize=True,
    )["response.metric"]

    reference_step = 4.0e-4
    problem.set_val("pbase", source.P_psin + reference_step * dp)
    problem.set_val("FF_psin", source.FF_psin + reference_step * dff)
    problem.run_model()
    plus = float(problem.get_val("response.metric")[0])
    problem.set_val("pbase", source.P_psin - reference_step * dp)
    problem.set_val("FF_psin", source.FF_psin - reference_step * dff)
    problem.run_model()
    minus = float(problem.get_val("response.metric")[0])
    expected = (plus - minus) / (2.0 * reference_step)

    assert_allclose(propagated, expected, rtol=3.0e-4, atol=1.0e-3)
    kernel.close()
