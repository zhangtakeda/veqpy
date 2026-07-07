#include "boundary_fit.h"

NB_MODULE(veqpy_boundary_fit_ext, module)
{
    module.doc() = "Standalone native boundary scatter-to-coefficient phase QR fitter.";
    module.def("fit_boundary_qr",
               &cxx_python::fit_boundary_qr,
               nb::arg("R_boundary"),
               nb::arg("Z_boundary"),
               nb::arg("c_order"),
               nb::arg("s_order"),
               "Fit R/Z boundary points to Kernel boundary coefficients with native phase QR.");
}
