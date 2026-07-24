#include "nonlinear.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>

namespace
{
    struct AffineResidual
    {
        static constexpr std::size_t equations = 2;
        static constexpr std::size_t variables = 2;

        void operator()(const double* x, double* f) const noexcept
        {
            f[0] = x[0] - 1.0;
            f[1] = x[1] + 2.0;
        }
    };

    struct AffineResidualWithJacobian : AffineResidual
    {
        void jacobian(const double*, double* jacobian) const noexcept
        {
            jacobian[0] = 1.0;
            jacobian[1] = 0.0;
            jacobian[2] = 0.0;
            jacobian[3] = 1.0;
        }
    };

    template <typename Policy, typename Functor>
    bool run_case(std::string_view name, const Functor& functor)
    {
        nonlinear::Workspace<Functor::variables> workspace;
        auto solver = nonlinear::make_solver<Policy>(functor, workspace);
        solver.context.tolerance       = 1.0e-12;
        solver.context.max_evaluations = 1000;

        std::array<double, Functor::variables> x{-4.0, 9.0};
        solver.context.optimize_inplace(x.data());

        std::array<double, Functor::equations> residual{};
        functor(x.data(), residual.data());
        const double norm = std::hypot(residual[0], residual[1]);
        std::cout << name << ' ' << solver.context.info << ' ' << solver.context.evaluations << ' '
                  << solver.context.jacobian_evaluations << ' ' << norm << '\n';
        return solver.context.info > 0 && norm <= 1.0e-10;
    }
} // namespace

static_assert(nonlinear::detail::cminpack_fallback_min_dimension == 2);
static_assert(nonlinear::detail::uses_standard_cminpack_v<nonlinear::Powell, AffineResidual>);
static_assert(nonlinear::detail::uses_standard_cminpack_v<nonlinear::Powell, AffineResidualWithJacobian>);
static_assert(nonlinear::detail::uses_standard_cminpack_v<nonlinear::LevenbergMarquardt, AffineResidual>);
static_assert(
    nonlinear::detail::uses_standard_cminpack_v<nonlinear::LevenbergMarquardt, AffineResidualWithJacobian>);

int main()
{
    const bool passed =
        run_case<nonlinear::Powell>("powell_fd", AffineResidual{}) &&
        run_case<nonlinear::Powell>("powell_jac", AffineResidualWithJacobian{}) &&
        run_case<nonlinear::LevenbergMarquardt>("lm_fd", AffineResidual{}) &&
        run_case<nonlinear::LevenbergMarquardt>("lm_jac", AffineResidualWithJacobian{});
    return passed ? 0 : 1;
}
