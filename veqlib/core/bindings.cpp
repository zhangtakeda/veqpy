#include "python_kernel_solver.h"

NB_MODULE(veqlib_ext, module)
{
    module.doc() = "Single-thread nanobind bridge for the production VEQlib kernel solver.";

    nb::class_<veqlib_python::KernelSolver>(module, "KernelSolver")
        .def(nb::init<int>(),
             nb::arg("solver_code") = static_cast<int>(veqlib_kernel_api::SolverMethodPowell))
        .def("metadata", &veqlib_python::KernelSolver::metadata)
        .def("metadata_json", &veqlib_python::KernelSolver::metadata_json)
        .def("set_case_json", &veqlib_python::KernelSolver::set_case_json, nb::arg("payload"))
        .def("set_kernel_runtime",
             &veqlib_python::KernelSolver::set_kernel_runtime,
             nb::arg("case_name"),
             nb::arg("a"),
             nb::arg("R0"),
             nb::arg("Z0"),
             nb::arg("B0"),
             nb::arg("ka"),
             nb::arg("c_offsets"),
             nb::arg("s_offsets"),
             nb::arg("scaled_heat"),
             nb::arg("scaled_current"),
             nb::arg("scaled_Ip"),
             nb::arg("beta"),
             nb::arg("fix_rho"),
             nb::arg("method_code"),
             nb::arg("max_residual"),
             nb::arg("max_evaluations"),
             nb::arg("accepted_residual_factor"),
             nb::arg("accepted_residual_floor"),
             nb::arg("initial_policy_code"),
             nb::arg("continue_policy_code"),
             nb::arg("residual_normalization_code"),
             nb::arg("residual_normalization_floor"),
             nb::arg("residual_normalization_max_ratio"),
             nb::arg("residual_normalization_huber_tau"),
             nb::arg("residual_normalization_probe_count"),
             nb::arg("residual_normalization_probe_step"),
             nb::arg("residual_normalization_sensitivity_lambda"),
             "Set the full runtime case and solve policy without JSON serialization.")
        .def("warmup", &veqlib_python::KernelSolver::warmup, nb::arg("count"))
        .def("solve_json", &veqlib_python::KernelSolver::solve_json)
        .def("solve_direct",
             &veqlib_python::KernelSolver::solve_direct,
             "Run one solve and return scalars plus read-only NumPy views without JSON serialization.")
        .def("adopt_last_solution_as_initial",
             &veqlib_python::KernelSolver::adopt_last_solution_as_initial,
             "Use the last accepted solve result as the current initial state.")
        .def("residual_var_into",
             &veqlib_python::KernelSolver::residual_var_into,
             nb::arg("x"),
             nb::arg("out"),
             "Evaluate the raw variational residual into a caller-owned packed output array.")
        .def("jvp_into",
             &veqlib_python::KernelSolver::jvp_into,
             nb::arg("x"),
             nb::arg("v"),
             nb::arg("out"),
             "Evaluate a raw-residual Jacobian-vector product into a caller-owned packed output array.")
        .def("jacobian_into",
             &veqlib_python::KernelSolver::jacobian_into,
             nb::arg("x"),
             nb::arg("out"),
             "Evaluate the dense raw-residual Jacobian into a caller-owned row-major matrix.")
        .def_prop_ro("last_elapsed_ms", &veqlib_python::KernelSolver::last_elapsed_ms);
}
