#pragma once

#include "linalg.h"
#include "tensor.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cminpack.h>
#include <cstddef>

namespace nonlinear::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    struct LevenbergMarquardt
    {
        template <typename Functor>
        struct Context;
    };

    struct NewtonKrylov
    {
        template <typename Functor>
        struct Context;
    };

    struct Newton
    {
        template <typename Functor>
        struct Context;
    };

    struct NewtonRaphson
    {
        template <typename Functor>
        struct Context;
    };

    struct Powell
    {
        template <typename Functor>
        struct Context;
    };

    template <typename Functor>
    inline constexpr bool has_jacobian_v =
        requires(Functor& functor, const double* x, double* jacobian) { functor.jacobian(x, jacobian); };

    template <typename Functor>
    inline constexpr bool has_jvp_v =
        requires(Functor& functor, const double* x, const double* v, double* jv) { functor.jvp(x, v, jv); };

    template <size_t N>
    double norm2(const double* values) noexcept
    {
        double total = 0.0;
        for (size_t i = 0; i < N; ++i)
            total += values[i] * values[i];
        return std::sqrt(total);
    }

    template <size_t N>
    void scaled_add(double* out, const double* x, double scale, const double* step) noexcept
    {
        for (size_t i = 0; i < N; ++i)
            out[i] = x[i] + scale * step[i];
    }

    template <typename Functor, size_t M, size_t N>
    void finite_difference_jacobian(Functor& functor, const double* x, double* jacobian)
    {
        std::array<double, N> x_plus;
        std::array<double, M> f_base;
        std::array<double, M> f_plus;
        std::copy(x, x + N, x_plus.begin());
        functor(x, f_base.data());
        for (size_t col = 0; col < N; ++col)
        {
            const double saved = x_plus[col];
            const double step  = 1.0e-7 * std::max(1.0, std::abs(saved));
            x_plus[col]        = saved + step;
            functor(x_plus.data(), f_plus.data());
            x_plus[col] = saved;
            for (size_t row = 0; row < M; ++row)
                jacobian[row * N + col] = (f_plus[row] - f_base[row]) / step;
        }
    }

    template <typename Functor, size_t M, size_t N>
    void evaluate_jacobian(Functor& functor, const double* x, double* jacobian)
    {
        if constexpr (has_jacobian_v<Functor>)
            functor.jacobian(x, jacobian);
        else
            finite_difference_jacobian<Functor, M, N>(functor, x, jacobian);
    }

    template <typename Functor, size_t N>
    void evaluate_jvp(Functor& functor, const double* x, const double* v, const double* f_base, double* jv)
    {
        if constexpr (has_jvp_v<Functor>)
        {
            functor.jvp(x, v, jv);
        }
        else
        {
            std::array<double, N> x_plus;
            std::array<double, N> f_plus;
            std::copy(x, x + N, x_plus.begin());
            double v_norm = norm2<N>(v);
            if (v_norm <= 0.0)
            {
                std::fill(jv, jv + N, 0.0);
                return;
            }
            const double x_norm = norm2<N>(x);
            const double eps    = std::sqrt(1.0e-12) * (1.0 + x_norm) / v_norm;
            for (size_t i = 0; i < N; ++i)
                x_plus[i] += eps * v[i];
            functor(x_plus.data(), f_plus.data());
            for (size_t i = 0; i < N; ++i)
                jv[i] = (f_plus[i] - f_base[i]) / eps;
        }
    }

    template <size_t N>
    struct DenseNewtonWork
    {
        Matrix<double, N, N>  jacobian{uninitialized};
        Matrix<double, N, 1>  rhs{uninitialized};
        Matrix<double, N, 1>  step{uninitialized};
        std::array<double, N> trial_x{};
        std::array<double, N> trial_f{};
    };

    template <typename Functor>
    struct Newton::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "Newton requires a square residual");

        Functor functor;
        double  tolerance            = 1.0e-8;
        int     max_iterations       = 50;
        int     evaluations          = 0;
        int     jacobian_evaluations = 0;
        int     info                 = 0;

        explicit Context(const Functor& value) : functor(value) {}

        void optimize_inplace(double* x)
        {
            constexpr size_t      n = variables;
            Matrix<double, n, n>  jacobian{uninitialized};
            Matrix<double, n, 1>  rhs{uninitialized};
            Matrix<double, n, 1>  step{uninitialized};
            std::array<double, n> f{};
            evaluations          = 0;
            jacobian_evaluations = 0;
            info                 = 5;

            for (int iteration = 0; iteration < max_iterations; ++iteration)
            {
                functor(x, f.data());
                ++evaluations;
                if (norm2<n>(f.data()) <= tolerance)
                {
                    info = 1;
                    return;
                }

                evaluate_jacobian<Functor, n, n>(functor, x, jacobian.data());
                ++jacobian_evaluations;
                for (size_t i = 0; i < n; ++i)
                    rhs[i] = -f[i];
                linalg::solve_into(step, jacobian, rhs);

                if (norm2<n>(step.data()) <= tolerance)
                {
                    info = 2;
                    return;
                }
                for (size_t i = 0; i < n; ++i)
                    x[i] += step[i];
            }
        }
    };

    template <typename Functor>
    struct NewtonRaphson::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "NewtonRaphson requires a square residual");

        Functor functor;
        double  tolerance            = 1.0e-8;
        int     max_iterations       = 50;
        int     evaluations          = 0;
        int     jacobian_evaluations = 0;
        int     info                 = 0;

        explicit Context(const Functor& value) : functor(value) {}

        void optimize_inplace(double* x)
        {
            constexpr size_t      n = variables;
            DenseNewtonWork<n>    work{};
            std::array<double, n> f{};
            evaluations          = 0;
            jacobian_evaluations = 0;
            info                 = 5;

            functor(x, f.data());
            ++evaluations;
            double current_norm = norm2<n>(f.data());
            if (current_norm <= tolerance)
            {
                info = 1;
                return;
            }

            for (int iteration = 0; iteration < max_iterations; ++iteration)
            {
                evaluate_jacobian<Functor, n, n>(functor, x, work.jacobian.data());
                ++jacobian_evaluations;
                for (size_t i = 0; i < n; ++i)
                    work.rhs[i] = -f[i];
                linalg::solve_into(work.step, work.jacobian, work.rhs);

                double step_scale = 1.0;
                bool   accepted   = false;
                for (int trial = 0; trial < 16; ++trial)
                {
                    scaled_add<n>(work.trial_x.data(), x, step_scale, work.step.data());
                    functor(work.trial_x.data(), work.trial_f.data());
                    ++evaluations;
                    const double trial_norm = norm2<n>(work.trial_f.data());
                    if (trial_norm < current_norm)
                    {
                        std::copy(work.trial_x.begin(), work.trial_x.end(), x);
                        f            = work.trial_f;
                        current_norm = trial_norm;
                        accepted     = true;
                        break;
                    }
                    step_scale *= 0.5;
                }

                if (current_norm <= tolerance)
                {
                    info = 1;
                    return;
                }
                if (!accepted)
                {
                    info = 4;
                    return;
                }
            }
        }
    };

    template <size_t N>
    struct GmresWork
    {
        std::array<double, (N + 1) * N> basis{};
        std::array<double, (N + 1) * N> hessenberg{};
        std::array<double, N>           givens_cos{};
        std::array<double, N>           givens_sin{};
        std::array<double, N + 1>       residual_axis{};
        std::array<double, N>           arnoldi{};
        std::array<double, N>           y{};
    };

    template <size_t N>
    double& hessenberg_at(GmresWork<N>& work, size_t row, size_t col) noexcept
    {
        return work.hessenberg[row * N + col];
    }

    template <size_t N>
    double* basis_vector(GmresWork<N>& work, size_t col) noexcept
    {
        return work.basis.data() + col * N;
    }

    template <typename Functor, size_t N>
    int gmres_solve(Functor&      functor,
                    const double* x,
                    const double* f_base,
                    const double* rhs,
                    double*       solution,
                    int           max_dimension,
                    double        tolerance,
                    int&          jvp_evaluations)
    {
        GmresWork<N> work{};
        std::fill(solution, solution + N, 0.0);

        const double beta = norm2<N>(rhs);
        if (beta <= tolerance)
            return 0;

        const size_t krylov_dimension = std::min(N, static_cast<size_t>(max_dimension));
        double*      first_basis      = basis_vector(work, 0);
        for (size_t i = 0; i < N; ++i)
            first_basis[i] = rhs[i] / beta;
        work.residual_axis[0] = beta;

        size_t used_dimension = 0;
        for (size_t col = 0; col < krylov_dimension; ++col)
        {
            evaluate_jvp<Functor, N>(functor, x, basis_vector(work, col), f_base, work.arnoldi.data());
            ++jvp_evaluations;

            for (size_t row = 0; row <= col; ++row)
            {
                const double* basis = basis_vector(work, row);
                double        dot   = 0.0;
                for (size_t i = 0; i < N; ++i)
                    dot += work.arnoldi[i] * basis[i];
                hessenberg_at(work, row, col) = dot;
                for (size_t i = 0; i < N; ++i)
                    work.arnoldi[i] -= dot * basis[i];
            }

            const double next_norm            = norm2<N>(work.arnoldi.data());
            hessenberg_at(work, col + 1, col) = next_norm;
            if (next_norm > 0.0 && col + 1 < N + 1)
            {
                double* next_basis = basis_vector(work, col + 1);
                for (size_t i = 0; i < N; ++i)
                    next_basis[i] = work.arnoldi[i] / next_norm;
            }

            for (size_t row = 0; row < col; ++row)
            {
                const double h0                   = hessenberg_at(work, row, col);
                const double h1                   = hessenberg_at(work, row + 1, col);
                hessenberg_at(work, row, col)     = work.givens_cos[row] * h0 + work.givens_sin[row] * h1;
                hessenberg_at(work, row + 1, col) = -work.givens_sin[row] * h0 + work.givens_cos[row] * h1;
            }

            const double h0                   = hessenberg_at(work, col, col);
            const double h1                   = hessenberg_at(work, col + 1, col);
            const double denom                = std::hypot(h0, h1);
            work.givens_cos[col]              = denom == 0.0 ? 1.0 : h0 / denom;
            work.givens_sin[col]              = denom == 0.0 ? 0.0 : h1 / denom;
            hessenberg_at(work, col, col)     = work.givens_cos[col] * h0 + work.givens_sin[col] * h1;
            hessenberg_at(work, col + 1, col) = 0.0;

            const double axis0          = work.residual_axis[col];
            work.residual_axis[col]     = work.givens_cos[col] * axis0;
            work.residual_axis[col + 1] = -work.givens_sin[col] * axis0;
            used_dimension              = col + 1;
            if (std::abs(work.residual_axis[col + 1]) <= tolerance)
                break;
        }

        for (size_t rr = used_dimension; rr > 0; --rr)
        {
            const size_t row = rr - 1;
            double       sum = work.residual_axis[row];
            for (size_t col = row + 1; col < used_dimension; ++col)
                sum -= hessenberg_at(work, row, col) * work.y[col];
            const double pivot = hessenberg_at(work, row, row);
            work.y[row]        = std::abs(pivot) <= 1.0e-14 ? 0.0 : sum / pivot;
        }

        for (size_t col = 0; col < used_dimension; ++col)
        {
            const double* basis = basis_vector(work, col);
            for (size_t i = 0; i < N; ++i)
                solution[i] += basis[i] * work.y[col];
        }
        return static_cast<int>(used_dimension);
    }

    template <typename Functor>
    struct NewtonKrylov::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "NewtonKrylov requires a square residual");

        Functor functor;
        double  tolerance         = 1.0e-8;
        int     max_iterations    = 50;
        int     max_dimension     = static_cast<int>(variables);
        int     evaluations       = 0;
        int     jvp_evaluations   = 0;
        int     linear_iterations = 0;
        int     info              = 0;

        explicit Context(const Functor& value) : functor(value) {}

        void optimize_inplace(double* x)
        {
            constexpr size_t      n = variables;
            std::array<double, n> f{};
            std::array<double, n> rhs{};
            std::array<double, n> step{};
            std::array<double, n> trial_x{};
            std::array<double, n> trial_f{};
            evaluations       = 0;
            jvp_evaluations   = 0;
            linear_iterations = 0;
            info              = 5;

            functor(x, f.data());
            ++evaluations;
            double current_norm = norm2<n>(f.data());
            if (current_norm <= tolerance)
            {
                info = 1;
                return;
            }

            for (int iteration = 0; iteration < max_iterations; ++iteration)
            {
                for (size_t i = 0; i < n; ++i)
                    rhs[i] = -f[i];
                linear_iterations += gmres_solve<Functor, n>(
                    functor, x, f.data(), rhs.data(), step.data(), max_dimension, tolerance * 0.1, jvp_evaluations);

                double step_scale = 1.0;
                bool   accepted   = false;
                for (int trial = 0; trial < 16; ++trial)
                {
                    scaled_add<n>(trial_x.data(), x, step_scale, step.data());
                    functor(trial_x.data(), trial_f.data());
                    ++evaluations;
                    const double trial_norm = norm2<n>(trial_f.data());
                    if (trial_norm < current_norm)
                    {
                        std::copy(trial_x.begin(), trial_x.end(), x);
                        f            = trial_f;
                        current_norm = trial_norm;
                        accepted     = true;
                        break;
                    }
                    step_scale *= 0.5;
                }

                if (current_norm <= tolerance)
                {
                    info = 1;
                    return;
                }
                if (!accepted)
                {
                    info = 4;
                    return;
                }
            }
        }
    };

    template <typename Functor>
    struct Powell::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "Powell hybrid requires a square residual");
        static constexpr size_t work_size = variables * (3 * variables + 13) / 2;

        Functor                               functor;
        Vector<double, equations>             fvec{uninitialized};
        Vector<double, work_size>             work{uninitialized};
        Vector<double, equations * variables> fjac{uninitialized};
        double                                tolerance            = 1.0e-8;
        int                                   max_evaluations      = 1000;
        int                                   evaluations          = 0;
        int                                   jacobian_evaluations = 0;
        int                                   info                 = 0;

        explicit Context(const Functor& value) : functor(value) {}

        void optimize_inplace(double* x)
        {
            evaluations          = 0;
            jacobian_evaluations = 0;
            if constexpr (has_jacobian_v<Functor>)
            {
                info = hybrj1(callback_with_jacobian,
                              this,
                              static_cast<int>(variables),
                              x,
                              fvec.data(),
                              fjac.data(),
                              static_cast<int>(equations),
                              tolerance,
                              work.data(),
                              static_cast<int>(work_size));
            }
            else
            {
                info = hybrd1(callback,
                              this,
                              static_cast<int>(variables),
                              x,
                              fvec.data(),
                              tolerance,
                              work.data(),
                              static_cast<int>(work_size));
            }
        }

        static int callback(void* data, int, const double* x, double* fvec, int iflag)
        {
            auto& self = *static_cast<Context*>(data);
            if (iflag > 0)
            {
                if (self.max_evaluations > 0 && self.evaluations >= self.max_evaluations)
                    return -1;
                self.functor(x, fvec);
                ++self.evaluations;
            }
            return 0;
        }

        static int
        callback_with_jacobian(void* data, int, const double* x, double* fvec, double* fjac, int ldfjac, int iflag)
        {
            auto& self = *static_cast<Context*>(data);
            if (iflag == 1)
            {
                if (self.max_evaluations > 0 && self.evaluations >= self.max_evaluations)
                    return -1;
                self.functor(x, fvec);
                ++self.evaluations;
            }
            else if (iflag == 2)
            {
                evaluate_jacobian<Functor, equations, variables>(self.functor, x, self.fjac.data());
                ++self.jacobian_evaluations;
                for (size_t row = 0; row < equations; ++row)
                    for (size_t col = 0; col < variables; ++col)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = self.fjac[row * variables + col];
            }
            return 0;
        }
    };

    template <typename Functor>
    struct LevenbergMarquardt::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static constexpr size_t work_size = equations * variables + 5 * variables + equations;

        Functor                               functor;
        Vector<double, equations>             fvec{uninitialized};
        Vector<double, work_size>             work{uninitialized};
        Vector<double, equations * variables> fjac{uninitialized};
        Vector<int, variables>                ipvt{uninitialized};
        double                                tolerance            = 1.0e-8;
        int                                   max_evaluations      = 1000;
        int                                   evaluations          = 0;
        int                                   jacobian_evaluations = 0;
        int                                   info                 = 0;

        explicit Context(const Functor& value) : functor(value) {}

        void optimize_inplace(double* x)
        {
            evaluations          = 0;
            jacobian_evaluations = 0;
            if constexpr (has_jacobian_v<Functor>)
            {
                info = lmder1(callback_with_jacobian,
                              this,
                              static_cast<int>(equations),
                              static_cast<int>(variables),
                              x,
                              fvec.data(),
                              fjac.data(),
                              static_cast<int>(equations),
                              tolerance,
                              ipvt.data(),
                              work.data(),
                              static_cast<int>(work_size));
            }
            else
            {
                info = lmdif1(callback,
                              this,
                              static_cast<int>(equations),
                              static_cast<int>(variables),
                              x,
                              fvec.data(),
                              tolerance,
                              ipvt.data(),
                              work.data(),
                              static_cast<int>(work_size));
            }
        }

        static int callback(void* data, int, int, const double* x, double* fvec, int iflag)
        {
            auto& self = *static_cast<Context*>(data);
            if (iflag > 0)
            {
                if (self.max_evaluations > 0 && self.evaluations >= self.max_evaluations)
                    return -1;
                self.functor(x, fvec);
                ++self.evaluations;
            }
            return 0;
        }

        static int
        callback_with_jacobian(void* data, int, int, const double* x, double* fvec, double* fjac, int ldfjac, int iflag)
        {
            auto& self = *static_cast<Context*>(data);
            if (iflag == 1)
            {
                if (self.max_evaluations > 0 && self.evaluations >= self.max_evaluations)
                    return -1;
                self.functor(x, fvec);
                ++self.evaluations;
            }
            else if (iflag == 2)
            {
                evaluate_jacobian<Functor, equations, variables>(self.functor, x, self.fjac.data());
                ++self.jacobian_evaluations;
                for (size_t row = 0; row < equations; ++row)
                    for (size_t col = 0; col < variables; ++col)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = self.fjac[row * variables + col];
            }
            return 0;
        }
    };

    template <typename Policy, typename Functor>
    struct Solver
    {
        typename Policy::template Context<Functor> context;

        explicit Solver(const Functor& functor) : context(functor) {}

        template <size_t N>
        void optimize_inplace(Vector<double, N>& x)
        {
            static_assert(N == Functor::variables);
            context.optimize_inplace(x.data());
        }
    };

    template <typename Policy, typename Functor>
    Solver<Policy, Functor> make_solver(const Functor& functor)
    {
        return Solver<Policy, Functor>{functor};
    }
} // namespace nonlinear::detail

namespace nonlinear
{
    using detail::LevenbergMarquardt;
    using detail::Newton;
    using detail::NewtonKrylov;
    using detail::NewtonRaphson;
    using detail::Powell;
    using detail::Solver;
    using detail::make_solver;
} // namespace nonlinear
