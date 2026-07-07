#pragma once

// Nonlinear solver adapters and finite-difference helpers for generated Cxx Kernel artifacts.

#include "linalg.h"
#include "tensor.h"
#include <algorithm>
#include <array>
#include <cmath>
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

    using FuncNN    = int (*)(void*, int, const double*, double*, int);
    using FuncDerNN = int (*)(void*, int, const double*, double*, double*, int, int);
    using FuncMN    = int (*)(void*, int, int, const double*, double*, int);
    using FuncDerMN = int (*)(void*, int, int, const double*, double*, double*, int, int);

    namespace cminpack
    {
        int hybrd(FuncNN  callback,
                  void*   data,
                  int     n,
                  double* x,
                  double* fvec,
                  double  xtol,
                  int     max_evaluations,
                  int     ml,
                  int     mu,
                  double  epsfcn,
                  double* diag,
                  int     mode,
                  double  factor,
                  int     nprint,
                  int*    nfev,
                  double* fjac,
                  int     ldfjac,
                  double* r,
                  int     lr,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);

        int hybrj(FuncDerNN callback,
                  void*     data,
                  int       n,
                  double*   x,
                  double*   fvec,
                  double*   fjac,
                  int       ldfjac,
                  double    xtol,
                  int       max_evaluations,
                  double*   diag,
                  int       mode,
                  double    factor,
                  int       nprint,
                  int*      nfev,
                  int*      njev,
                  double*   r,
                  int       lr,
                  double*   qtf,
                  double*   wa1,
                  double*   wa2,
                  double*   wa3,
                  double*   wa4);

        int lmdif(FuncMN  callback,
                  void*   data,
                  int     m,
                  int     n,
                  double* x,
                  double* fvec,
                  double  ftol,
                  double  xtol,
                  double  gtol,
                  int     max_evaluations,
                  double  epsfcn,
                  double* diag,
                  int     mode,
                  double  factor,
                  int     nprint,
                  int*    nfev,
                  double* fjac,
                  int     ldfjac,
                  int*    ipvt,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);

        int lmder(FuncDerMN callback,
                  void*     data,
                  int       m,
                  int       n,
                  double*   x,
                  double*   fvec,
                  double*   fjac,
                  int       ldfjac,
                  double    ftol,
                  double    xtol,
                  double    gtol,
                  int       max_evaluations,
                  double*   diag,
                  int       mode,
                  double    factor,
                  int       nprint,
                  int*      nfev,
                  int*      njev,
                  int*      ipvt,
                  double*   qtf,
                  double*   wa1,
                  double*   wa2,
                  double*   wa3,
                  double*   wa4);
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
        static constexpr int    cminpack_size = static_cast<int>(variables);
        static constexpr int    lr_size       = cminpack_size * (cminpack_size + 1) / 2;

        Functor                               functor;
        Vector<double, equations>             fvec{uninitialized};
        Vector<double, variables>             diag{uninitialized};
        Vector<double, equations * variables> fjac{uninitialized};
        Vector<double, equations * variables> jacobian{uninitialized};
        Vector<double, static_cast<size_t>(lr_size)> r{uninitialized};
        Vector<double, variables>             qtf{uninitialized};
        Vector<double, variables>             wa1{uninitialized};
        Vector<double, variables>             wa2{uninitialized};
        Vector<double, variables>             wa3{uninitialized};
        Vector<double, variables>             wa4{uninitialized};
        double                                tolerance            = 1.0e-8;
        double                                finite_difference_step = 1.0e-6;
        double                                initial_step_bound    = 1.0;
        int                                   max_evaluations      = 1000;
        int                                   lower_bandwidth      = cminpack_size - 1;
        int                                   upper_bandwidth      = cminpack_size - 1;
        int                                   scale_mode           = 1;
        int                                   print_interval       = 0;
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
                run_jacobian_backend(x);
            }
            else
            {
                run_finite_difference_backend(x);
            }
        }

        void run_jacobian_backend(double* x)
        {
            diag.fill(1.0);
            int nfev = 0;
            int njev = 0;
            info     = cminpack::hybrj(callback_with_jacobian,
                                   this,
                                   cminpack_size,
                                   x,
                                   fvec.data(),
                                   fjac.data(),
                                   cminpack_size,
                                   tolerance,
                                   max_evaluations,
                                   diag.data(),
                                   scale_mode,
                                   initial_step_bound,
                                   print_interval,
                                   &nfev,
                                   &njev,
                                   r.data(),
                                   lr_size,
                                   qtf.data(),
                                   wa1.data(),
                                   wa2.data(),
                                   wa3.data(),
                                   wa4.data());
            evaluations          = nfev;
            jacobian_evaluations = njev;
        }

        void run_finite_difference_backend(double* x)
        {
            diag.fill(1.0);
            int nfev = 0;
            info     = cminpack::hybrd(callback,
                                   this,
                                   cminpack_size,
                                   x,
                                   fvec.data(),
                                   tolerance,
                                   max_evaluations,
                                   lower_bandwidth,
                                   upper_bandwidth,
                                   finite_difference_step,
                                   diag.data(),
                                   scale_mode,
                                   initial_step_bound,
                                   print_interval,
                                   &nfev,
                                   fjac.data(),
                                   cminpack_size,
                                   r.data(),
                                   lr_size,
                                   qtf.data(),
                                   wa1.data(),
                                   wa2.data(),
                                   wa3.data(),
                                   wa4.data());
            evaluations = nfev;
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
                evaluate_jacobian<Functor, equations, variables>(self.functor, x, self.jacobian.data());
                ++self.jacobian_evaluations;
                for (size_t row = 0; row < equations; ++row)
                    for (size_t col = 0; col < variables; ++col)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = self.jacobian[row * variables + col];
            }
            return 0;
        }
    };

    template <typename Functor>
    struct LevenbergMarquardt::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static constexpr int    cminpack_equations = static_cast<int>(equations);
        static constexpr int    cminpack_variables = static_cast<int>(variables);

        Functor                               functor;
        Vector<double, equations>             fvec{uninitialized};
        Vector<double, variables>             diag{uninitialized};
        Vector<double, equations * variables> fjac{uninitialized};
        Vector<double, equations * variables> jacobian{uninitialized};
        Vector<int, variables>                ipvt{uninitialized};
        Vector<double, variables>             qtf{uninitialized};
        Vector<double, variables>             wa1{uninitialized};
        Vector<double, variables>             wa2{uninitialized};
        Vector<double, variables>             wa3{uninitialized};
        Vector<double, equations>             wa4{uninitialized};
        double                                tolerance            = 1.0e-8;
        double                                finite_difference_step = 0.0;
        double                                initial_step_bound    = 100.0;
        int                                   max_evaluations      = 1000;
        int                                   scale_mode           = 2;
        int                                   print_interval       = 0;
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
                run_jacobian_backend(x);
            }
            else
            {
                run_finite_difference_backend(x);
            }
        }

        void run_jacobian_backend(double* x)
        {
            diag.fill(1.0);
            int nfev = 0;
            int njev = 0;
            info     = cminpack::lmder(callback_with_jacobian,
                                   this,
                                   cminpack_equations,
                                   cminpack_variables,
                                   x,
                                   fvec.data(),
                                   fjac.data(),
                                   cminpack_equations,
                                   tolerance,
                                   tolerance,
                                   tolerance,
                                   max_evaluations,
                                   diag.data(),
                                   scale_mode,
                                   initial_step_bound,
                                   print_interval,
                                   &nfev,
                                   &njev,
                                   ipvt.data(),
                                   qtf.data(),
                                   wa1.data(),
                                   wa2.data(),
                                   wa3.data(),
                                   wa4.data());
            evaluations          = nfev;
            jacobian_evaluations = njev;
        }

        void run_finite_difference_backend(double* x)
        {
            diag.fill(1.0);
            int nfev = 0;
            info     = cminpack::lmdif(callback,
                                   this,
                                   cminpack_equations,
                                   cminpack_variables,
                                   x,
                                   fvec.data(),
                                   tolerance,
                                   tolerance,
                                   tolerance,
                                   max_evaluations,
                                   finite_difference_step,
                                   diag.data(),
                                   scale_mode,
                                   initial_step_bound,
                                   print_interval,
                                   &nfev,
                                   fjac.data(),
                                   cminpack_equations,
                                   ipvt.data(),
                                   qtf.data(),
                                   wa1.data(),
                                   wa2.data(),
                                   wa3.data(),
                                   wa4.data());
            evaluations = nfev;
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
                evaluate_jacobian<Functor, equations, variables>(self.functor, x, self.jacobian.data());
                ++self.jacobian_evaluations;
                for (size_t row = 0; row < equations; ++row)
                    for (size_t col = 0; col < variables; ++col)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = self.jacobian[row * variables + col];
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
    using detail::NewtonKrylov;
    using detail::NewtonRaphson;
    using detail::Powell;
    using detail::Solver;
    using detail::make_solver;
} // namespace nonlinear
