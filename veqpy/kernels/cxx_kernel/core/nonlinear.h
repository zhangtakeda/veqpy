#pragma once

// Nonlinear solver adapters and finite-difference helpers for generated Cxx Kernel artifacts.

#include "linalg.h"
#include "minpack/inline_powell.h"
#include "tensor.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <new>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace nonlinear::detail
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    template <size_t N>
    class Workspace
    {
        static constexpr size_t alignment = 64;
        // Policies are mutually exclusive, so one arena only needs to cover the
        // largest live layout.  Three N-by-N double buffers plus linear terms
        // cover Powell, LM, GMRES, and Newton/LU with per-slice alignment slack.
        static constexpr size_t capacity_bytes =
            (3 * N * N + 32 * N + 16) * sizeof(double) + N * sizeof(int) + 12 * alignment;

        struct AlignedDelete
        {
            void operator()(std::byte* pointer) const noexcept
            {
                ::operator delete(pointer, std::align_val_t{alignment});
            }
        };

        std::unique_ptr<std::byte, AlignedDelete> storage_{
            static_cast<std::byte*>(::operator new(capacity_bytes, std::align_val_t{alignment}))};
        size_t cursor_ = 0;

    public:
        class Scope
        {
            Workspace* workspace_;
            size_t     marker_;

        public:
            explicit Scope(Workspace& workspace) noexcept
                : workspace_(&workspace), marker_(workspace.cursor_)
            {
            }

            Scope(const Scope&)            = delete;
            Scope& operator=(const Scope&) = delete;

            ~Scope() { workspace_->cursor_ = marker_; }
        };

        Workspace() = default;

        Workspace(const Workspace&)            = delete;
        Workspace& operator=(const Workspace&) = delete;

        void reset() noexcept { cursor_ = 0; }

        template <typename T>
        std::span<T> take(size_t count)
        {
            const size_t aligned = (cursor_ + alignment - 1) & ~(alignment - 1);
            const size_t bytes   = count * sizeof(T);
            if (aligned > capacity_bytes || bytes > capacity_bytes - aligned)
                throw std::runtime_error("nonlinear workspace capacity exceeded");
            auto* result = reinterpret_cast<T*>(storage_.get() + aligned);
            cursor_      = aligned + bytes;
            return {result, count};
        }

        template <typename T>
        T& take_object()
        {
            static_assert(std::is_trivially_destructible_v<T>);
            auto storage = take<std::byte>(sizeof(T));
            return *::new (static_cast<void*>(storage.data())) T{};
        }
    };

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

    // Generated VEQ artifacts stay on the fixed-size, VEQ-specialized path.
    // This deliberately remote handoff keeps the standard CMINPACK driver
    // available without putting a runtime size test in a generated kernel.
#ifndef VEQPY_CXX_CMINPACK_FALLBACK_MIN_DIMENSION
#define VEQPY_CXX_CMINPACK_FALLBACK_MIN_DIMENSION (1024 * 128)
#endif
    inline constexpr size_t cminpack_fallback_min_dimension =
        static_cast<size_t>(VEQPY_CXX_CMINPACK_FALLBACK_MIN_DIMENSION);

    template <typename Policy, typename Functor>
    inline constexpr bool uses_standard_cminpack_v =
        (std::is_same_v<Policy, Powell> || std::is_same_v<Policy, LevenbergMarquardt>) &&
        Functor::variables >= cminpack_fallback_min_dimension;

    // Keep the system CMINPACK header out of this header: the specialized
    // primitives intentionally retain a private, prefixed CMINPACK subset.
    // nonlinear.cpp includes the system header and forwards these calls.
    namespace standard_cminpack
    {
        using FuncNN = int (*)(void*, int, const double*, double*, int);
        using FuncDerNN = int (*)(void*, int, const double*, double*, double*, int, int);
        using FuncMN = int (*)(void*, int, int, const double*, double*, int);
        using FuncDerMN = int (*)(void*, int, int, const double*, double*, double*, int, int);

        int hybrd(FuncNN callback,
                  void* data,
                  int n,
                  double* x,
                  double* fvec,
                  double xtol,
                  int max_evaluations,
                  int ml,
                  int mu,
                  double epsfcn,
                  double* diag,
                  int mode,
                  double factor,
                  int nprint,
                  int* nfev,
                  double* fjac,
                  int ldfjac,
                  double* r,
                  int lr,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);

        int hybrj(FuncDerNN callback,
                  void* data,
                  int n,
                  double* x,
                  double* fvec,
                  double* fjac,
                  int ldfjac,
                  double xtol,
                  int max_evaluations,
                  double* diag,
                  int mode,
                  double factor,
                  int nprint,
                  int* nfev,
                  int* njev,
                  double* r,
                  int lr,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);

        int lmdif(FuncMN callback,
                  void* data,
                  int m,
                  int n,
                  double* x,
                  double* fvec,
                  double ftol,
                  double xtol,
                  double gtol,
                  int max_evaluations,
                  double epsfcn,
                  double* diag,
                  int mode,
                  double factor,
                  int nprint,
                  int* nfev,
                  double* fjac,
                  int ldfjac,
                  int* ipvt,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);

        int lmder(FuncDerMN callback,
                  void* data,
                  int m,
                  int n,
                  double* x,
                  double* fvec,
                  double* fjac,
                  int ldfjac,
                  double ftol,
                  double xtol,
                  double gtol,
                  int max_evaluations,
                  double* diag,
                  int mode,
                  double factor,
                  int nprint,
                  int* nfev,
                  int* njev,
                  int* ipvt,
                  double* qtf,
                  double* wa1,
                  double* wa2,
                  double* wa3,
                  double* wa4);
    } // namespace standard_cminpack

    template <typename Functor>
    inline constexpr bool has_jacobian_v =
        requires(Functor& functor, const double* x, double* jacobian) { functor.jacobian(x, jacobian); };

    template <typename Functor>
    inline constexpr bool has_column_major_jacobian_v =
        requires(Functor& functor, const double* x, double* jacobian, int leading_dimension) {
            functor.jacobian_column_major(x, jacobian, leading_dimension);
        };

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
    struct NewtonRaphson::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "NewtonRaphson requires a square residual");

        Functor               functor;
        Workspace<variables>* workspace;
        double  tolerance            = 1.0e-8;
        int     max_iterations       = 50;
        int     evaluations          = 0;
        int     jacobian_evaluations = 0;
        int     info                 = 0;

        Context(const Functor& value, Workspace<variables>& value_workspace)
            : functor(value), workspace(&value_workspace)
        {
        }

        void optimize_inplace(double* x)
        {
            constexpr size_t n = variables;
            typename Workspace<n>::Scope scope{*workspace};
            auto& work = workspace->template take_object<DenseNewtonWork<n>>();
            auto& factorization =
                workspace->template take_object<linalg::Context<linalg::Doolittle, n, n>>();
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
                std::copy(work.rhs.begin(), work.rhs.end(), work.step.begin());
                linalg::factorize_into(factorization, work.jacobian);
                factorization.template substitute_inplace<1>(work.step.data());

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
        std::span<double> basis;
        std::span<double> hessenberg;
        std::span<double> givens_cos;
        std::span<double> givens_sin;
        std::span<double> residual_axis;
        std::span<double> arnoldi;
        std::span<double> y;

        explicit GmresWork(Workspace<N>& workspace)
            : basis(workspace.template take<double>((N + 1) * N)),
              hessenberg(workspace.template take<double>((N + 1) * N)),
              givens_cos(workspace.template take<double>(N)),
              givens_sin(workspace.template take<double>(N)),
              residual_axis(workspace.template take<double>(N + 1)),
              arnoldi(workspace.template take<double>(N)),
              y(workspace.template take<double>(N))
        {
            std::fill(basis.begin(), basis.end(), 0.0);
            std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
            std::fill(givens_cos.begin(), givens_cos.end(), 0.0);
            std::fill(givens_sin.begin(), givens_sin.end(), 0.0);
            std::fill(residual_axis.begin(), residual_axis.end(), 0.0);
            std::fill(arnoldi.begin(), arnoldi.end(), 0.0);
            std::fill(y.begin(), y.end(), 0.0);
        }
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
                    Workspace<N>& workspace,
                    const double* x,
                    const double* f_base,
                    const double* rhs,
                    double*       solution,
                    int           max_dimension,
                    double        tolerance,
                    int&          jvp_evaluations)
    {
        typename Workspace<N>::Scope scope{workspace};
        GmresWork<N>                  work{workspace};
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

        Functor               functor;
        Workspace<variables>* workspace;
        double  tolerance         = 1.0e-8;
        int     max_iterations    = 50;
        int     max_dimension     = static_cast<int>(variables);
        int     evaluations       = 0;
        int     jvp_evaluations   = 0;
        int     linear_iterations = 0;
        int     info              = 0;

        Context(const Functor& value, Workspace<variables>& value_workspace)
            : functor(value), workspace(&value_workspace)
        {
        }

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
                linear_iterations += gmres_solve<Functor, n>(functor,
                                                             *workspace,
                                                             x,
                                                             f.data(),
                                                             rhs.data(),
                                                             step.data(),
                                                             max_dimension,
                                                             tolerance * 0.1,
                                                             jvp_evaluations);

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
        static constexpr size_t lr_size = variables * (variables + 1) / 2;
        static constexpr bool cminpack_dimensions_supported =
            variables <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            lr_size <= static_cast<size_t>(std::numeric_limits<int>::max());

        Functor           functor;
        std::span<double> fvec;
        std::span<double> diag;
        std::span<double> fjac;
        std::span<double> jacobian;
        std::span<double> r;
        std::span<double> qtf;
        std::span<double> wa1;
        std::span<double> wa2;
        std::span<double> wa3;
        std::span<double> wa4;
        double                                tolerance            = 1.0e-8;
        double                                finite_difference_step = 1.0e-6;
        double                                initial_step_bound    = 1.0;
        int                                   max_evaluations      = 1000;
        int                                   evaluations          = 0;
        int                                   jacobian_evaluations = 0;
        int                                   info                 = 0;

        Context(const Functor& value, Workspace<variables>& workspace)
            : functor(value),
              fvec(workspace.template take<double>(equations)),
              diag(workspace.template take<double>(variables)),
              fjac(workspace.template take<double>(equations * variables)),
              jacobian(take_jacobian_workspace(workspace)),
              r(workspace.template take<double>(lr_size)),
              qtf(workspace.template take<double>(variables)),
              wa1(workspace.template take<double>(variables)),
              wa2(workspace.template take<double>(variables)),
              wa3(workspace.template take<double>(variables)),
              wa4(workspace.template take<double>(variables))
        {
        }

        void optimize_inplace(double* x)
        {
            evaluations          = 0;
            jacobian_evaluations = 0;
            if constexpr (uses_standard_cminpack_v<Powell, Functor>)
            {
                static_assert(cminpack_dimensions_supported,
                              "standard CMINPACK cannot represent this Powell workspace");
                if constexpr (has_jacobian_v<Functor>)
                    run_standard_jacobian_backend(x);
                else
                    run_standard_finite_difference_backend(x);
            }
            else
            {
                if constexpr (has_jacobian_v<Functor>)
                    run_jacobian_backend(x);
                else
                    run_finite_difference_backend(x);
            }
        }

        static constexpr int cminpack_dimension() noexcept { return static_cast<int>(variables); }

        static int standard_finite_difference_callback(void* data,
                                                       int n,
                                                       const double* values,
                                                       double* residual,
                                                       int flag)
        {
            auto& context = *static_cast<Context*>(data);
            if (n != cminpack_dimension())
                return -1;
            if (flag > 0)
            {
                if (context.max_evaluations > 0 && context.evaluations >= context.max_evaluations)
                    return -1;
                context.functor(values, residual);
                ++context.evaluations;
            }
            return 0;
        }

        static int standard_jacobian_callback(void* data,
                                              int n,
                                              const double* values,
                                              double* residual,
                                              double* jacobian_values,
                                              int leading_dimension,
                                              int flag)
        {
            auto& context = *static_cast<Context*>(data);
            if (n != cminpack_dimension() || leading_dimension < n)
                return -1;
            if (flag == 1)
            {
                if (context.max_evaluations > 0 && context.evaluations >= context.max_evaluations)
                    return -1;
                context.functor(values, residual);
                ++context.evaluations;
            }
            else if (flag == 2)
            {
                if constexpr (has_column_major_jacobian_v<Functor>)
                {
                    context.functor.jacobian_column_major(values, jacobian_values, leading_dimension);
                }
                else
                {
                    evaluate_jacobian<Functor, equations, variables>(
                        context.functor, values, context.jacobian.data());
                    for (size_t row = 0; row < equations; ++row)
                        for (size_t col = 0; col < variables; ++col)
                            jacobian_values[row + static_cast<size_t>(leading_dimension) * col] =
                                context.jacobian[row * variables + col];
                }
                ++context.jacobian_evaluations;
            }
            return 0;
        }

        void run_standard_jacobian_backend(double* x)
        {
            int nfev = 0;
            int njev = 0;
            info = standard_cminpack::hybrj(&Context::standard_jacobian_callback,
                                             this,
                                             cminpack_dimension(),
                                             x,
                                             fvec.data(),
                                             fjac.data(),
                                             cminpack_dimension(),
                                             tolerance,
                                             max_evaluations,
                                             diag.data(),
                                             1,
                                             initial_step_bound,
                                             0,
                                             &nfev,
                                             &njev,
                                             r.data(),
                                             static_cast<int>(lr_size),
                                             qtf.data(),
                                             wa1.data(),
                                             wa2.data(),
                                             wa3.data(),
                                             wa4.data());
            evaluations          = nfev;
            jacobian_evaluations = njev;
        }

        void run_standard_finite_difference_backend(double* x)
        {
            int nfev = 0;
            info = standard_cminpack::hybrd(&Context::standard_finite_difference_callback,
                                             this,
                                             cminpack_dimension(),
                                             x,
                                             fvec.data(),
                                             tolerance,
                                             max_evaluations,
                                             cminpack_dimension() - 1,
                                             cminpack_dimension() - 1,
                                             finite_difference_step,
                                             diag.data(),
                                             1,
                                             initial_step_bound,
                                             0,
                                             &nfev,
                                             fjac.data(),
                                             cminpack_dimension(),
                                             r.data(),
                                             static_cast<int>(lr_size),
                                             qtf.data(),
                                             wa1.data(),
                                             wa2.data(),
                                             wa3.data(),
                                             wa4.data());
            evaluations = nfev;
        }

        void run_jacobian_backend(double* x)
        {
            int nfev = 0;
            int njev = 0;
            auto evaluate = [this](const double* values,
                                   double* residual,
                                   double* jacobian_values,
                                   int     leading_dimension,
                                   int     flag) {
                if (flag == 1)
                {
                    if (max_evaluations > 0 && evaluations >= max_evaluations)
                        return -1;
                    functor(values, residual);
                    ++evaluations;
                }
                else if (flag == 2)
                {
                    if constexpr (has_column_major_jacobian_v<Functor>)
                    {
                        functor.jacobian_column_major(values, jacobian_values, leading_dimension);
                    }
                    else
                    {
                        evaluate_jacobian<Functor, equations, variables>(functor, values, jacobian.data());
                        for (size_t row = 0; row < equations; ++row)
                            for (size_t col = 0; col < variables; ++col)
                                jacobian_values[row + static_cast<size_t>(leading_dimension) * col] =
                                    jacobian[row * variables + col];
                    }
                    ++jacobian_evaluations;
                }
                return 0;
            };
            info = minpack_inline::powell_with_jacobian<variables>(evaluate,
                                                                   x,
                                                                   fvec.data(),
                                                                   fjac.data(),
                                                                   tolerance,
                                                                   max_evaluations,
                                                                   diag.data(),
                                                                   initial_step_bound,
                                                                   &nfev,
                                                                   &njev,
                                                                   r.data(),
                                                                   qtf.data(),
                                                                   wa1.data(),
                                                                   wa2.data(),
                                                                   wa3.data(),
                                                                   wa4.data());
            evaluations          = nfev;
            jacobian_evaluations = njev;
        }

        static std::span<double> take_jacobian_workspace(Workspace<variables>& workspace)
        {
            if constexpr (!has_jacobian_v<Functor> || has_column_major_jacobian_v<Functor>)
                return {};
            else
                return workspace.template take<double>(equations * variables);
        }

        void run_finite_difference_backend(double* x)
        {
            int nfev = 0;
            auto evaluate = [this](const double* values, double* residual, int flag) {
                if (flag > 0)
                {
                    if (max_evaluations > 0 && evaluations >= max_evaluations)
                        return -1;
                    functor(values, residual);
                    ++evaluations;
                }
                return 0;
            };
            info = minpack_inline::powell_finite_difference<variables>(evaluate,
                                                                       x,
                                                                       fvec.data(),
                                                                       tolerance,
                                                                       max_evaluations,
                                                                       finite_difference_step,
                                                                       diag.data(),
                                                                       initial_step_bound,
                                                                       &nfev,
                                                                       fjac.data(),
                                                                       r.data(),
                                                                       qtf.data(),
                                                                       wa1.data(),
                                                                       wa2.data(),
                                                                       wa3.data(),
                                                                       wa4.data());
            evaluations = nfev;
        }

    };

    template <typename Functor>
    struct LevenbergMarquardt::Context
    {
        static constexpr size_t equations = Functor::equations;
        static constexpr size_t variables = Functor::variables;
        static_assert(equations == variables, "VEQPy LM specialization requires a square residual");
        static constexpr bool cminpack_dimensions_supported =
            equations <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            variables <= static_cast<size_t>(std::numeric_limits<int>::max());

        Functor           functor;
        std::span<double> fvec;
        std::span<double> fjac;
        std::span<double> jacobian;
        std::span<double> diag;
        std::span<int>    ipvt;
        std::span<double> qtf;
        std::span<double> wa1;
        std::span<double> wa2;
        std::span<double> wa3;
        std::span<double> wa4;
        double                                tolerance            = 1.0e-8;
        double                                finite_difference_step = 0.0;
        double                                initial_step_bound    = 100.0;
        int                                   max_evaluations      = 1000;
        int                                   evaluations          = 0;
        int                                   jacobian_evaluations = 0;
        int                                   info                 = 0;

        Context(const Functor& value, Workspace<variables>& workspace)
            : functor(value),
              fvec(workspace.template take<double>(equations)),
              fjac(workspace.template take<double>(equations * variables)),
              jacobian(take_jacobian_workspace(workspace)),
              diag(workspace.template take<double>(variables)),
              ipvt(workspace.template take<int>(variables)),
              qtf(workspace.template take<double>(variables)),
              wa1(workspace.template take<double>(variables)),
              wa2(workspace.template take<double>(variables)),
              wa3(workspace.template take<double>(variables)),
              wa4(workspace.template take<double>(equations))
        {
        }

        void optimize_inplace(double* x)
        {
            evaluations          = 0;
            jacobian_evaluations = 0;
            if constexpr (uses_standard_cminpack_v<LevenbergMarquardt, Functor>)
            {
                static_assert(cminpack_dimensions_supported,
                              "standard CMINPACK cannot represent this LM workspace");
                if constexpr (has_jacobian_v<Functor>)
                    run_standard_jacobian_backend(x);
                else
                    run_standard_finite_difference_backend(x);
            }
            else
            {
                if constexpr (has_jacobian_v<Functor>)
                    run_jacobian_backend(x);
                else
                    run_finite_difference_backend(x);
            }
        }

        static constexpr int cminpack_dimension() noexcept { return static_cast<int>(variables); }

        static int standard_finite_difference_callback(void* data,
                                                       int m,
                                                       int n,
                                                       const double* values,
                                                       double* residual,
                                                       int flag)
        {
            auto& context = *static_cast<Context*>(data);
            if (m != cminpack_dimension() || n != cminpack_dimension())
                return -1;
            if (flag > 0)
            {
                if (context.max_evaluations > 0 && context.evaluations >= context.max_evaluations)
                    return -1;
                context.functor(values, residual);
                ++context.evaluations;
            }
            return 0;
        }

        static int standard_jacobian_callback(void* data,
                                              int m,
                                              int n,
                                              const double* values,
                                              double* residual,
                                              double* jacobian_values,
                                              int leading_dimension,
                                              int flag)
        {
            auto& context = *static_cast<Context*>(data);
            if (m != cminpack_dimension() || n != cminpack_dimension() || leading_dimension < m)
                return -1;
            if (flag == 1)
            {
                if (context.max_evaluations > 0 && context.evaluations >= context.max_evaluations)
                    return -1;
                context.functor(values, residual);
                ++context.evaluations;
            }
            else if (flag == 2)
            {
                if constexpr (has_column_major_jacobian_v<Functor>)
                {
                    context.functor.jacobian_column_major(values, jacobian_values, leading_dimension);
                }
                else
                {
                    evaluate_jacobian<Functor, equations, variables>(
                        context.functor, values, context.jacobian.data());
                    for (size_t row = 0; row < equations; ++row)
                        for (size_t col = 0; col < variables; ++col)
                            jacobian_values[row + static_cast<size_t>(leading_dimension) * col] =
                                context.jacobian[row * variables + col];
                }
                ++context.jacobian_evaluations;
            }
            return 0;
        }

        void run_standard_jacobian_backend(double* x)
        {
            std::fill(diag.begin(), diag.end(), 1.0);
            int nfev = 0;
            int njev = 0;
            info = standard_cminpack::lmder(&Context::standard_jacobian_callback,
                                             this,
                                             cminpack_dimension(),
                                             cminpack_dimension(),
                                             x,
                                             fvec.data(),
                                             fjac.data(),
                                             cminpack_dimension(),
                                             tolerance,
                                             tolerance,
                                             tolerance,
                                             max_evaluations,
                                             diag.data(),
                                             2,
                                             initial_step_bound,
                                             0,
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

        void run_standard_finite_difference_backend(double* x)
        {
            std::fill(diag.begin(), diag.end(), 1.0);
            int nfev = 0;
            info = standard_cminpack::lmdif(&Context::standard_finite_difference_callback,
                                             this,
                                             cminpack_dimension(),
                                             cminpack_dimension(),
                                             x,
                                             fvec.data(),
                                             tolerance,
                                             tolerance,
                                             tolerance,
                                             max_evaluations,
                                             finite_difference_step,
                                             diag.data(),
                                             2,
                                             initial_step_bound,
                                             0,
                                             &nfev,
                                             fjac.data(),
                                             cminpack_dimension(),
                                             ipvt.data(),
                                             qtf.data(),
                                             wa1.data(),
                                             wa2.data(),
                                             wa3.data(),
                                             wa4.data());
            evaluations = nfev;
        }

        void run_jacobian_backend(double* x)
        {
            int nfev = 0;
            int njev = 0;
            auto evaluate = [this](const double* values,
                                   double* residual,
                                   double* jacobian_values,
                                   int     leading_dimension,
                                   int     flag) {
                if (flag == 1)
                {
                    if (max_evaluations > 0 && evaluations >= max_evaluations)
                        return -1;
                    functor(values, residual);
                    ++evaluations;
                }
                else if (flag == 2)
                {
                    if constexpr (has_column_major_jacobian_v<Functor>)
                    {
                        functor.jacobian_column_major(values, jacobian_values, leading_dimension);
                    }
                    else
                    {
                        evaluate_jacobian<Functor, equations, variables>(functor, values, jacobian.data());
                        for (size_t row = 0; row < equations; ++row)
                            for (size_t col = 0; col < variables; ++col)
                                jacobian_values[row + static_cast<size_t>(leading_dimension) * col] =
                                    jacobian[row * variables + col];
                    }
                    ++jacobian_evaluations;
                }
                return 0;
            };
            info = minpack_inline::lm_with_jacobian<variables>(evaluate,
                                                               x,
                                                               fvec.data(),
                                                               fjac.data(),
                                                               tolerance,
                                                               tolerance,
                                                               tolerance,
                                                               max_evaluations,
                                                               initial_step_bound,
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

        static std::span<double> take_jacobian_workspace(Workspace<variables>& workspace)
        {
            if constexpr (!has_jacobian_v<Functor> || has_column_major_jacobian_v<Functor>)
                return {};
            else
                return workspace.template take<double>(equations * variables);
        }

        void run_finite_difference_backend(double* x)
        {
            int nfev = 0;
            auto evaluate = [this](const double* values, double* residual, int flag) {
                if (flag > 0)
                {
                    if (max_evaluations > 0 && evaluations >= max_evaluations)
                        return -1;
                    functor(values, residual);
                    ++evaluations;
                }
                return 0;
            };
            info = minpack_inline::lm_finite_difference<variables>(evaluate,
                                                                   x,
                                                                   fvec.data(),
                                                                   tolerance,
                                                                   tolerance,
                                                                   tolerance,
                                                                   max_evaluations,
                                                                   finite_difference_step,
                                                                   initial_step_bound,
                                                                   &nfev,
                                                                   fjac.data(),
                                                                   ipvt.data(),
                                                                   qtf.data(),
                                                                   wa1.data(),
                                                                   wa2.data(),
                                                                   wa3.data(),
                                                                   wa4.data());
            evaluations = nfev;
        }

    };

    template <typename Policy, typename Functor>
    struct Solver
    {
        typename Policy::template Context<Functor> context;

        Solver(const Functor& functor, Workspace<Functor::variables>& workspace) : context(functor, workspace) {}

        template <size_t N>
        void optimize_inplace(Vector<double, N>& x)
        {
            static_assert(N == Functor::variables);
            context.optimize_inplace(x.data());
        }
    };

    template <typename Policy, typename Functor>
    Solver<Policy, Functor> make_solver(const Functor& functor, Workspace<Functor::variables>& workspace)
    {
        workspace.reset();
        return Solver<Policy, Functor>{functor, workspace};
    }
} // namespace nonlinear::detail

namespace nonlinear
{
    using detail::LevenbergMarquardt;
    using detail::NewtonKrylov;
    using detail::NewtonRaphson;
    using detail::Powell;
    using detail::Solver;
    using detail::Workspace;
    using detail::make_solver;
} // namespace nonlinear
