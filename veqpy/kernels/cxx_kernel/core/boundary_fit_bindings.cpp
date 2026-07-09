#include "boundary_fit.h"

NB_MODULE(veqpy_boundary_fit_ext, module)
{
    module.doc() = "Standalone native boundary scatter-to-coefficient fitters.";
    module.def("fit_boundary_qr",
               &cxx_python::fit_boundary_qr,
               nb::arg("R_boundary"),
               nb::arg("Z_boundary"),
               nb::arg("c_order"),
               nb::arg("s_order"),
               "Fit R/Z boundary points to Kernel boundary coefficients with native weighted QR.");
    module.def("fit_boundary_weighted_gnqr",
               &cxx_python::fit_boundary_weighted_gnqr,
               nb::arg("R_boundary"),
               nb::arg("Z_boundary"),
               nb::arg("c_order"),
               nb::arg("s_order"),
               "Fit R/Z boundary points with weighted phase QR followed by fixed-geometry GNQR.");
    module.def("fit_boundary_least_square",
               &cxx_python::fit_boundary_least_square,
               nb::arg("R_boundary"),
               nb::arg("Z_boundary"),
               nb::arg("c_order"),
               nb::arg("s_order"),
               "Fit R/Z boundary points with bounded full least-square optimization.");
}
