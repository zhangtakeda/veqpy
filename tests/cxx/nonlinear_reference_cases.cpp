#include "nonlinear.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace
{
    struct RosenbrockResidual
    {
        static constexpr std::size_t equations = 2;
        static constexpr std::size_t variables = 2;

        void operator()(const double* x, double* f) const noexcept
        {
            f[0] = 10.0 * (x[1] - x[0] * x[0]);
            f[1] = 1.0 - x[0];
        }
    };

    struct RosenbrockResidualWithJacobian : RosenbrockResidual
    {
        void jacobian(const double* x, double* jacobian) const noexcept
        {
            jacobian[0] = -20.0 * x[0];
            jacobian[1] = 10.0;
            jacobian[2] = -1.0;
            jacobian[3] = 0.0;
        }
    };

    struct ScaledResidual
    {
        static constexpr std::size_t equations = 2;
        static constexpr std::size_t variables = 2;

        void operator()(const double* x, double* f) const noexcept
        {
            f[0] = 1.0e150 * (x[0] - 1.0);
            f[1] = 1.0e-150 * (x[1] - 2.0);
        }
    };

    struct RankDeficientResidualWithJacobian
    {
        static constexpr std::size_t equations = 2;
        static constexpr std::size_t variables = 2;

        void operator()(const double* x, double* f) const noexcept
        {
            f[0] = x[0] + x[1] - 2.0;
            f[1] = 2.0 * f[0];
        }

        void jacobian(const double*, double* jacobian) const noexcept
        {
            jacobian[0] = 1.0;
            jacobian[1] = 1.0;
            jacobian[2] = 2.0;
            jacobian[3] = 2.0;
        }
    };

    struct RemoteCminpackResidual
    {
        static constexpr std::size_t equations = 8096;
        static constexpr std::size_t variables = 8096;
    };

    template <typename Policy, typename Functor>
    void run_case(std::string_view                         name,
                  const Functor&                           functor,
                  std::array<double, Functor::variables>   x,
                  double                                   tolerance,
                  int                                      max_evaluations)
    {
        nonlinear::Workspace<Functor::variables> workspace;
        auto solver = nonlinear::make_solver<Policy>(functor, workspace);
        solver.context.tolerance       = tolerance;
        solver.context.max_evaluations = max_evaluations;
        solver.context.optimize_inplace(x.data());

        std::array<double, Functor::equations> residual{};
        functor(x.data(), residual.data());
        double norm = 0.0;
        for (double value : residual)
            norm = std::hypot(norm, value);

        std::cout << name << ' ' << solver.context.info << ' ' << solver.context.evaluations << ' '
                  << solver.context.jacobian_evaluations;
        for (double value : x)
            std::cout << ' ' << value;
        std::cout << ' ' << norm << '\n';
    }
} // namespace

static_assert(nonlinear::detail::cminpack_fallback_min_dimension == 8096);
static_assert(!nonlinear::detail::uses_standard_cminpack_v<nonlinear::Powell, RosenbrockResidual>);
static_assert(!nonlinear::detail::uses_standard_cminpack_v<nonlinear::LevenbergMarquardt, RosenbrockResidual>);
static_assert(nonlinear::detail::uses_standard_cminpack_v<nonlinear::Powell, RemoteCminpackResidual>);
static_assert(
    nonlinear::detail::uses_standard_cminpack_v<nonlinear::LevenbergMarquardt, RemoteCminpackResidual>);

int main()
{
    std::cout << std::setprecision(17);

    run_case<nonlinear::Powell>("powell_fd_rosenbrock", RosenbrockResidual{}, {-1.2, 1.0}, 1.0e-10, 1000);
    run_case<nonlinear::Powell>(
        "powell_jac_rosenbrock", RosenbrockResidualWithJacobian{}, {-1.2, 1.0}, 1.0e-10, 1000);
    run_case<nonlinear::LevenbergMarquardt>(
        "lm_fd_rosenbrock", RosenbrockResidual{}, {-1.2, 1.0}, 1.0e-10, 1000);
    run_case<nonlinear::LevenbergMarquardt>(
        "lm_jac_rosenbrock", RosenbrockResidualWithJacobian{}, {-1.2, 1.0}, 1.0e-10, 1000);

    run_case<nonlinear::Powell>("powell_immediate", RosenbrockResidual{}, {1.0, 1.0}, 1.0e-10, 1000);
    run_case<nonlinear::LevenbergMarquardt>(
        "lm_immediate", RosenbrockResidualWithJacobian{}, {1.0, 1.0}, 1.0e-10, 1000);
    run_case<nonlinear::Powell>("powell_budget", RosenbrockResidual{}, {-1.2, 1.0}, 1.0e-10, 1);
    run_case<nonlinear::LevenbergMarquardt>(
        "lm_budget", RosenbrockResidual{}, {-1.2, 1.0}, 1.0e-10, 1);

    run_case<nonlinear::Powell>("powell_scaled", ScaledResidual{}, {0.0, 0.0}, 1.0e-10, 1000);
    run_case<nonlinear::LevenbergMarquardt>(
        "lm_rank_deficient", RankDeficientResidualWithJacobian{}, {0.0, 0.0}, 1.0e-10, 1000);
}
