#include "python_kernel_solver.h"

NB_MODULE(veqlib_ext, module)
{
    module.doc() = "Single-thread nanobind bridge for the production VEQlib kernel solver.";

    nb::class_<veqlib_python::KernelSolver>(module, "KernelSolver")
        .def(nb::init<int, int>(),
             nb::arg("solver_code")  = static_cast<int>(veqlib_kernel_api::SolverMethodPowell),
             nb::arg("enzyme_width") = 1)
        .def("metadata", &veqlib_python::KernelSolver::metadata)
        .def("metadata_json", &veqlib_python::KernelSolver::metadata_json)
        .def("set_case_json", &veqlib_python::KernelSolver::set_case_json, nb::arg("payload"))
        .def("warmup", &veqlib_python::KernelSolver::warmup, nb::arg("count"))
        .def("solve_json", &veqlib_python::KernelSolver::solve_json)
        .def("solve_direct",
             &veqlib_python::KernelSolver::solve_direct,
             "Run one solve and return scalars plus read-only NumPy views without JSON serialization.")
        .def("residual_var_into",
             &veqlib_python::KernelSolver::residual_var_into,
             nb::arg("x"),
             nb::arg("out"),
             "Evaluate the raw variational residual into a caller-owned packed output array.")
        .def_prop_ro("last_elapsed_ms", &veqlib_python::KernelSolver::last_elapsed_ms);
}
