#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cminpack.h>
#include <kinsol/kinsol.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunmatrix/sunmatrix_dense.h>
#ifdef ENABLE_ENZYME
#include <enzyme/enzyme>
extern int enzyme_dupv;
extern int enzyme_width;
#endif
#include <nlohmann/json.hpp>

#include "grid.h"
#include "math.h"
#include "nonlinear.h"
#include "pf_psin_uniform_operator.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_pf_psin_uniform_benchmark_cli
{

namespace
{
    using grid::Legendre;
    using grid::Spectral;
    using operator_pf::PfPsinUniformOperator;
    using std::size_t;
    using tensor::uninitialized;

    constexpr size_t max_size(size_t lhs, size_t rhs) noexcept { return lhs < rhs ? rhs : lhs; }

    template <auto Counts>
    consteval size_t max_profile_count() noexcept
    {
        size_t value = 0;
        for (size_t count : Counts)
            value = max_size(value, count);
        return value;
    }

    template <auto CFamilyCounts, auto SFamilyCounts>
    consteval size_t inferred_M_max() noexcept
    {
        constexpr size_t c_max = CFamilyCounts.size() == 0 ? 0 : CFamilyCounts.size() - 1;
        constexpr size_t s_max = SFamilyCounts.size();
        const size_t     value = max_size(c_max, s_max);
        return value > 1 ? value : 1;
    }

    template <size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts>
    consteval size_t inferred_L_max() noexcept
    {
        size_t count = HCount;
        count        = max_size(count, VCount);
        count        = max_size(count, KappaCount);
        count        = max_size(count, PsinCount);
        count        = max_size(count, FCount);
        count        = max_size(count, max_profile_count<CFamilyCounts>());
        count        = max_size(count, max_profile_count<SFamilyCounts>());
        return count > 1 ? count - 1 : 1;
    }

    template <size_t Mmax>
    consteval size_t inferred_K_max() noexcept
    {
        return Mmax > 2 ? Mmax : 2;
    }

    template <size_t Nr,
              size_t Nt,
              size_t SourceSamples,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              typename QuadratureScheme,
              typename CalculusScheme>
    struct PfPsinUniformIpTopology
    {
        static constexpr size_t L_max = inferred_L_max<
            HCount,
            VCount,
            KappaCount,
            PsinCount,
            FCount,
            CFamilyCounts,
            SFamilyCounts>();
        static constexpr size_t M_max = inferred_M_max<CFamilyCounts, SFamilyCounts>();
        static constexpr size_t K_max = inferred_K_max<M_max>();

        using Shape = profiles::OptimizedProfileShapeFromCountsT<
            L_max,
            K_max,
            HCount,
            VCount,
            KappaCount,
            PsinCount,
            FCount,
            CFamilyCounts,
            SFamilyCounts>;
        using Grid = grid::Grid<
            Nr,
            Nt,
            Shape::L_max,
            Shape::M_max,
            Shape::K_max,
            QuadratureScheme,
            CalculusScheme>;
        using Source   = source::UniformSourceShape<SourceSamples>;
        using Operator = PfPsinUniformOperator<Shape, Grid, Source>;
    };

    constexpr auto bench_c_counts = std::array<size_t, 1>{0};
    constexpr auto bench_s_counts = std::array<size_t, 1>{3};

    using BenchTopology = PfPsinUniformIpTopology<
        32,
        16,
        51,
        3,
        0,
        6,
        6,
        0,
        bench_c_counts,
        bench_s_counts,
        Legendre,
        Spectral>;
    using BenchShape    = BenchTopology::Shape;
    using BenchGrid     = BenchTopology::Grid;
    using BenchSource   = BenchTopology::Source;
    using BenchOperator = BenchTopology::Operator;
    using PackedVector  = BenchOperator::PackedVector;

    static_assert(BenchShape::x_size == 18);
    static_assert(BenchShape::L_max == 5);
    static_assert(BenchShape::M_max == 1);
    static_assert(BenchShape::K_max == 2);

    constexpr double veqpy_max_residual              = 1.0e-6;
    constexpr int    veqpy_requested_max_evaluations = 1000;
    constexpr int    veqpy_maxfev =
        veqpy_requested_max_evaluations > 500 ? veqpy_requested_max_evaluations : 500;
    constexpr double veqpy_hybr_eps                 = 1.0e-6;
    constexpr double veqpy_hybr_factor              = 1.0;
    constexpr int    veqpy_hybr_mode                = 1;
    constexpr int    veqpy_hybr_nprint              = 0;
    constexpr double veqpy_accepted_residual_factor = 10.0;
    constexpr double veqpy_accepted_residual_floor  = 1.0e-5;
    constexpr double veqpy_x_scale_floor            = 1.0e-2;
    constexpr double veqpy_core_profile_prior       = 1.5e-1;
    constexpr double veqpy_fourier_profile_prior    = 5.0e-2;
    constexpr double veqpy_F_profile_prior          = 2.5e-1;
    constexpr double veqpy_kappa_profile_prior      = 1.0;

    enum class SolverKind
    {
        ResidualOnly,
        EnzymeJacobian,
        LevenbergMarquardt,
        Newton,
        NewtonKrylov,
        NewtonRaphson,
        Powell,
        SundialsNewtonKrylov,
        SundialsNewtonRaphson,
    };

    struct CaseInput
    {
        std::string case_name = "PF_psin_uniform_Ip";
        std::array<double, BenchSource::sample_count> heat{};
        std::array<double, BenchSource::sample_count> current{};
        std::array<double, BenchShape::x_size>        x0{};
        std::array<double, BenchShape::x_size>        x_scale{};
        std::array<double, BenchShape::x_size>        residual_scale{};
        double a         = 1.05 / 1.85;
        double R0        = 1.05;
        double Z0        = 0.0;
        double B0        = 3.0;
        double ka        = 2.2;
        double c0_offset = 0.0;
        double s1_offset = 0.52359877559829887308;
        double Ip        = 3.7699111867885415;
        double fix_rho   = 0.05;
        int    repeat    = 10;
        int    warmup    = 1;
        int    enzyme_width = 1;
        SolverKind solver = SolverKind::ResidualOnly;
    };

    struct SolveResult
    {
        std::array<double, BenchShape::x_size> x{};
        PackedVector                          raw{};
        PackedVector                          scaled{};
        std::array<double, 2>                 alpha{};
        double                                raw_norm    = 0.0;
        double                                scaled_norm = 0.0;
        int                                   info        = 0;
        int                                   nfev        = 0;
        int                                   njev        = 0;
        int                                   callbacks   = 0;
        int                                   jacobian_component_evaluations = 0;
        int                                   jvp_evaluations = 0;
        int                                   linear_iterations = 0;
        double                                residual_callback_ms = 0.0;
        double                                residual_kernel_ms = 0.0;
        double                                residual_scale_ms = 0.0;
        double                                final_residual_ms = 0.0;
        double                                jacobian_callback_ms = 0.0;
        double                                jvp_callback_ms = 0.0;
        double                                linear_solve_ms = 0.0;
        bool                                  accepted    = false;
    };

    struct JacobianCheck
    {
        double enzyme_norm = 0.0;
        double finite_difference_norm = 0.0;
        double max_abs_diff = 0.0;
        double max_rel_diff = 0.0;
    };

    double norm2(std::span<const double, BenchShape::x_size> values) noexcept
    {
        double total = 0.0;
        for (double value : values)
            total += value * value;
        return std::sqrt(total);
    }

    double elapsed_ms_since(std::chrono::steady_clock::time_point started) noexcept
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    }

    constexpr double veqpy_acceptance_threshold() noexcept
    {
        constexpr double scaled = veqpy_max_residual * veqpy_accepted_residual_factor;
        return scaled > veqpy_accepted_residual_floor ? scaled : veqpy_accepted_residual_floor;
    }

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_heat = {
        -0.789683058574694,   -0.7925936329632908,  -0.7953979059157582,
        -0.7981175242684836,  -0.8007734699426484,  -0.8033829643453037,
        -0.8059602413311435,  -0.8085160883171674,  -0.811058402798229,
        -0.8135924782601793,  -0.8161210171627857,  -0.8186443668543919,
        -0.8211602449193173,  -0.82366353909573,    -0.8261466407410907,
        -0.8286005031116496,  -0.8310129351634512,  -0.8333638631960294,
        -0.8356348482673257,  -0.8378088362996647,  -0.8398452317867598,
        -0.8417128477427658,  -0.8433930625296288,  -0.8448046915616387,
        -0.8459047888622718,  -0.8467054195394251,  -0.8468550092753417,
        -0.8466915344330476,  -0.8459189587408882,  -0.844353928057723,
        -0.8418811874023397,  -0.8384860115442367,  -0.8339038151163928,
        -0.8279318154386046,  -0.8204511956034657,  -0.8111301194374045,
        -0.7996783681429246,  -0.7858549460849861,  -0.7692183653145488,
        -0.7492833542853738,  -0.7256038086449172,  -0.6975411195823384,
        -0.6643030775956854,  -0.6249342427240242,  -0.5782460674488447,
        -0.5227081311582821,  -0.4562798131359923,  -0.3761603550417907,
        -0.2784903156513912,  -0.15751845630174122, -0.004428769494182179,
    };

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_current = {
        -0.2884247371510828,   -0.28903009704030685,  -0.28957011555052764,
        -0.29005075790401863,  -0.2904781886031161,   -0.2908575435658439,
        -0.2911919683745232,   -0.2914838716684553,   -0.2917344967330943,
        -0.2919437907627883,   -0.29211086667346203,  -0.2922346427093772,
        -0.2923145136340828,   -0.29232232828599186,  -0.2923070658458569,
        -0.2922075036101631,   -0.29204221633819877,  -0.2918026604476458,
        -0.2914746207679046,   -0.29104564954616613,  -0.2905098257293566,
        -0.28984851066007944,  -0.2890431993822433,   -0.2880834160319397,
        -0.2869422749878048,   -0.28559761849429993,  -0.28403023480001177,
        -0.2822015304104101,   -0.2800887239343856,   -0.27765576104313033,
        -0.274853815959196,    -0.27165492157456717,  -0.26800248440388824,
        -0.2638385916755866,   -0.25912135836327627,  -0.25377334278513203,
        -0.2477205935589761,   -0.2409002037441097,   -0.23321180576616163,
        -0.22454617787318162,  -0.2148028033443947,   -0.20384443014756062,
        -0.191504616045878,    -0.17758595047988296,  -0.1618458187382766,
        -0.143974507066849,    -0.12356213030456985,  -0.10004932271682981,
        -0.07266570051589127,  -0.040268147498330784, -0.0011074929612556953,
    };

    template <typename Shape>
    constexpr bool is_c_profile_id(size_t profile_id) noexcept
    {
        return profile_id >= Shape::c0_profile_id && profile_id <= Shape::c0_profile_id + Shape::M_max;
    }

    template <typename Shape>
    constexpr bool is_s_profile_id(size_t profile_id) noexcept
    {
        return profile_id >= Shape::c0_profile_id + Shape::M_max + 1 &&
               profile_id <= Shape::c0_profile_id + 2 * Shape::M_max;
    }

    template <typename Shape>
    constexpr double x_scale_profile_prior(size_t profile_id) noexcept
    {
        if (profile_id == Shape::h_profile_id || profile_id == Shape::v_profile_id ||
            profile_id == Shape::psin_profile_id)
            return veqpy_core_profile_prior;
        if (profile_id == Shape::kappa_profile_id)
            return veqpy_kappa_profile_prior;
        if (is_c_profile_id<Shape>(profile_id) || is_s_profile_id<Shape>(profile_id))
            return veqpy_fourier_profile_prior;
        if (profile_id == Shape::F_profile_id)
            return veqpy_F_profile_prior;
        return veqpy_F_profile_prior;
    }

    template <typename Shape>
    constexpr bool x_scale_offsetless(size_t profile_id) noexcept
    {
        return profile_id == Shape::h_profile_id || profile_id == Shape::v_profile_id ||
               profile_id == Shape::psin_profile_id;
    }

    profiles::ProfileRuntimeParams<BenchShape> profile_params_for_case(const CaseInput& input) noexcept
    {
        profiles::ProfileRuntimeParams<BenchShape> params{};
        params.offsets[BenchShape::kappa_profile_id] = input.ka;
        params.offsets[BenchShape::c_profile_id<0>()] = input.c0_offset;
        params.offsets[BenchShape::s_profile_id<1>()] = input.s1_offset;
        return params;
    }

    template <typename Shape>
    std::array<double, Shape::x_size> build_x_block_scale_vector(
        const std::array<double, Shape::x_size>&             x_guess,
        const profiles::ProfileRuntimeParams<Shape>& profile_params
    ) noexcept
    {
        std::array<double, Shape::x_size> scale{};
        scale.fill(1.0);
        for (size_t active_slot = 0; active_slot < Shape::active_count; ++active_slot)
        {
            const size_t profile_id = Shape::active_profile_ids[active_slot];
            const size_t length     = Shape::active_lengths[active_slot];
            if (length == 0)
                continue;

            double guess_norm2 = 0.0;
            for (size_t degree = 0; degree < length; ++degree)
            {
                const auto x_index = static_cast<size_t>(Shape::coeff_index[profile_id][degree]);
                guess_norm2 += x_guess[x_index] * x_guess[x_index];
            }

            const double guess_rms = std::sqrt(guess_norm2 / static_cast<double>(length));
            const double prior     = x_scale_profile_prior<Shape>(profile_id);
            const double offset_scale =
                x_scale_offsetless<Shape>(profile_id) ? 0.0 : std::abs(profile_params.offsets[profile_id]);
            double profile_scale = std::abs(profile_params.scales[profile_id]);
            if (std::abs(profile_scale - 1.0) <= 1.0e-12)
                profile_scale = prior;
            const double block_scale =
                std::max({offset_scale, profile_scale, prior, guess_rms, veqpy_x_scale_floor});

            for (size_t degree = 0; degree < length; ++degree)
            {
                const auto x_index = static_cast<size_t>(Shape::coeff_index[profile_id][degree]);
                scale[x_index]     = block_scale;
            }
        }
        return scale;
    }

    std::array<double, BenchShape::x_size> build_block_rms_residual_scale(const PackedVector& residual) noexcept
    {
        std::array<double, BenchShape::x_size> scale;
        size_t                                 offset = 0;
        for (size_t block = 0; block < BenchShape::active_count; ++block)
        {
            const size_t length = BenchShape::active_lengths[block];
            double       total  = 0.0;
            for (size_t i = 0; i < length; ++i)
                total += residual[offset + i] * residual[offset + i];
            const double rms = std::sqrt(total / static_cast<double>(length));
            const double block_scale = rms > 1.0 ? rms : 1.0;
            for (size_t i = 0; i < length; ++i)
                scale[offset + i] = block_scale;
            offset += length;
        }
        return scale;
    }

    int parse_nonnegative_int(const char* flag, const char* value)
    {
        char*      end    = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0 || parsed > 1000000000L)
            throw std::runtime_error(std::string{"invalid "} + flag + " value: " + value);
        return static_cast<int>(parsed);
    }

    SolverKind parse_solver_kind(const char* value)
    {
        const std::string solver{value};
        if (solver == "residual" || solver == "residual-only" || solver == "hybrd")
            return SolverKind::ResidualOnly;
        if (solver == "enzyme" || solver == "enzyme-jacobian" || solver == "hybrj")
        {
#ifdef ENABLE_ENZYME
            return SolverKind::EnzymeJacobian;
#else
            throw std::runtime_error("--solver enzyme requires the clang-enzyme-release preset");
#endif
        }
        if (solver == "lm" || solver == "levenberg-marquardt")
            return SolverKind::LevenbergMarquardt;
        if (solver == "newton" || solver == "pure-newton")
            return SolverKind::Newton;
        if (solver == "nk" || solver == "newton-krylov")
            return SolverKind::NewtonKrylov;
        if (solver == "nr" || solver == "newton-raphson")
            return SolverKind::NewtonRaphson;
        if (solver == "powell" || solver == "hybrid")
            return SolverKind::Powell;
        if (solver == "sundials-nk" || solver == "kinsol-nk")
            return SolverKind::SundialsNewtonKrylov;
        if (solver == "sundials-nr" || solver == "kinsol-nr")
            return SolverKind::SundialsNewtonRaphson;
        throw std::runtime_error("invalid --solver value: " + solver);
    }

    constexpr const char* solver_entrypoint(SolverKind solver) noexcept
    {
        switch (solver)
        {
        case SolverKind::ResidualOnly:
            return "cminpack::hybrd";
        case SolverKind::EnzymeJacobian:
            return "cminpack::hybrj";
        case SolverKind::LevenbergMarquardt:
            return "nonlinear::LevenbergMarquardt";
        case SolverKind::Newton:
            return "nonlinear::Newton";
        case SolverKind::NewtonKrylov:
            return "nonlinear::NewtonKrylov";
        case SolverKind::NewtonRaphson:
            return "nonlinear::NewtonRaphson";
        case SolverKind::Powell:
            return "nonlinear::Powell";
        case SolverKind::SundialsNewtonKrylov:
            return "SUNDIALS KINSOL + SUNLinSol_SPGMR";
        case SolverKind::SundialsNewtonRaphson:
            return "SUNDIALS KINSOL + SUNLinSol_Dense";
        }
        return "unknown";
    }

    constexpr const char* solver_method(SolverKind solver) noexcept
    {
        switch (solver)
        {
        case SolverKind::ResidualOnly:
            return "hybrd";
        case SolverKind::EnzymeJacobian:
            return "hybrj";
        case SolverKind::LevenbergMarquardt:
            return "levenberg_marquardt";
        case SolverKind::Newton:
            return "pure_newton";
        case SolverKind::NewtonKrylov:
            return "newton_krylov";
        case SolverKind::NewtonRaphson:
            return "newton_raphson";
        case SolverKind::Powell:
            return "powell_hybrid";
        case SolverKind::SundialsNewtonKrylov:
            return "sundials_newton_krylov";
        case SolverKind::SundialsNewtonRaphson:
            return "sundials_newton_raphson";
        }
        return "unknown";
    }

    constexpr bool solver_info_succeeded(SolverKind solver, int info) noexcept
    {
        switch (solver)
        {
        case SolverKind::SundialsNewtonKrylov:
        case SolverKind::SundialsNewtonRaphson:
            return info >= 0;
        case SolverKind::ResidualOnly:
        case SolverKind::EnzymeJacobian:
        case SolverKind::LevenbergMarquardt:
        case SolverKind::Newton:
        case SolverKind::NewtonKrylov:
        case SolverKind::NewtonRaphson:
        case SolverKind::Powell:
            return info > 0;
        }
        return false;
    }

    constexpr bool supported_enzyme_width(int width) noexcept
    {
        return width == 1 || width == 2 || width == 3 || width == 4 || width == 5 || width == 6 ||
               width == 8 || width == 9 || width == 10 || width == 12 || width == 18;
    }

    constexpr const char* solver_jacobian(const CaseInput& input) noexcept
    {
        switch (input.solver)
        {
        case SolverKind::ResidualOnly:
            return "cminpack forward difference";
        case SolverKind::EnzymeJacobian:
            switch (input.enzyme_width)
            {
            case 1:
                return "Enzyme scalar forward-mode dense Jacobian";
            case 2:
                return "Enzyme vector forward-mode dense Jacobian (width=2)";
            case 3:
                return "Enzyme vector forward-mode dense Jacobian (width=3)";
            case 4:
                return "Enzyme vector forward-mode dense Jacobian (width=4)";
            case 5:
                return "Enzyme vector forward-mode dense Jacobian (width=5)";
            case 6:
                return "Enzyme vector forward-mode dense Jacobian (width=6)";
            case 8:
                return "Enzyme vector forward-mode dense Jacobian (width=8)";
            case 9:
                return "Enzyme vector forward-mode dense Jacobian (width=9)";
            case 10:
                return "Enzyme vector forward-mode dense Jacobian (width=10)";
            case 12:
                return "Enzyme vector forward-mode dense Jacobian (width=12)";
            case 18:
                return "Enzyme vector forward-mode dense Jacobian (width=18)";
            default:
                return "Enzyme vector forward-mode dense Jacobian";
            }
        case SolverKind::LevenbergMarquardt:
            return "cminpack forward difference";
        case SolverKind::Newton:
#ifdef ENABLE_ENZYME
            return "Enzyme dense Jacobian through full-step Newton";
#else
            return "finite-difference dense Jacobian through full-step Newton";
#endif
        case SolverKind::NewtonKrylov:
#ifdef ENABLE_ENZYME
            return "Enzyme Jacobian-vector product through GMRES";
#else
            return "finite-difference Jacobian-vector product through GMRES";
#endif
        case SolverKind::NewtonRaphson:
#ifdef ENABLE_ENZYME
            return "Enzyme dense Jacobian through dense Newton";
#else
            return "finite-difference dense Jacobian through dense Newton";
#endif
        case SolverKind::Powell:
            return "cminpack forward difference";
        case SolverKind::SundialsNewtonKrylov:
#ifdef ENABLE_ENZYME
            return "KINSOL SPGMR with Enzyme Jacobian-vector product";
#else
            return "KINSOL SPGMR with finite-difference Jacobian-vector product";
#endif
        case SolverKind::SundialsNewtonRaphson:
#ifdef ENABLE_ENZYME
            return "KINSOL dense linear solver with Enzyme dense Jacobian";
#else
            return "KINSOL dense linear solver with finite-difference dense Jacobian";
#endif
        }
        return "unknown";
    }

    CaseInput build_inline_case(int repeat, int warmup, SolverKind solver, int enzyme_jacobian_width)
    {
        CaseInput input{};
        input.heat           = benchmark_scaled_heat;
        input.current        = benchmark_scaled_current;
        input.repeat         = repeat;
        input.warmup         = warmup;
        input.solver         = solver;
        input.enzyme_width   = enzyme_jacobian_width;
        input.x_scale        = build_x_block_scale_vector<BenchShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);
        return input;
    }

    template <typename Values>
    nlohmann::json json_array(const Values& values)
    {
        nlohmann::json out = nlohmann::json::array();
        for (double value : values)
            out.push_back(value);
        return out;
    }

    std::array<double, BenchShape::x_size> decode_z_to_x(
        std::span<const double, BenchShape::x_size> z,
        const std::array<double, BenchShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, BenchShape::x_size> x;
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            x[i] = z[i] * x_scale[i];
        return x;
    }

    std::array<double, BenchShape::x_size> encode_x_to_z(
        const std::array<double, BenchShape::x_size>& x,
        const std::array<double, BenchShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, BenchShape::x_size> z;
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            z[i] = x[i] / x_scale[i];
        return z;
    }

    void configure_operator_for_case(BenchOperator& op, const CaseInput& input) noexcept
    {
        BenchOperator::RuntimeParams params{};
        params.a              = input.a;
        params.R0             = input.R0;
        params.Z0             = input.Z0;
        params.B0             = input.B0;
        params.Ip             = input.Ip;
        params.fix_rho        = input.fix_rho;
        params.profile_params = profile_params_for_case(input);
        op.set_runtime_params(params);
        op.set_uniform_sources(
            std::span<const double, BenchSource::sample_count>{input.heat.data(), input.heat.size()},
            std::span<const double, BenchSource::sample_count>{input.current.data(), input.current.size()}
        );
    }

    struct SolveContext
    {
        BenchOperator op{};
        CaseInput     input{};
        int           evaluations = 0;
        int           jacobian_component_evaluations = 0;
        double        residual_callback_ms = 0.0;
        double        residual_kernel_ms   = 0.0;
        double        residual_scale_ms    = 0.0;
        double        final_residual_ms    = 0.0;
        double        jacobian_callback_ms = 0.0;
        double        jvp_callback_ms      = 0.0;
        double        linear_solve_ms      = 0.0;

        explicit SolveContext(const CaseInput& case_input) : input(case_input)
        {
            configure_operator_for_case(op, input);
        }

        void reset_solve_counters() noexcept
        {
            evaluations                      = 0;
            jacobian_component_evaluations   = 0;
            residual_callback_ms             = 0.0;
            residual_kernel_ms               = 0.0;
            residual_scale_ms                = 0.0;
            final_residual_ms                = 0.0;
            jacobian_callback_ms             = 0.0;
            jvp_callback_ms                  = 0.0;
            linear_solve_ms                  = 0.0;
        }

        void raw_residual(
            std::span<const double, BenchShape::x_size> x,
            std::span<double, BenchShape::x_size>       residual
        ) noexcept
        {
            PackedVector raw{uninitialized};
            op.evaluate(x, raw);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                residual[i] = raw[i];
        }
    };

    void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept;

#ifdef ENABLE_ENZYME
    struct EnzymeResidualContext
    {
        BenchOperator                                op{};
        std::array<double, BenchShape::x_size>       x_scale{};
        std::array<double, BenchShape::x_size>       residual_scale{};
    };

    EnzymeResidualContext enzyme_context_for_input(const CaseInput& input) noexcept
    {
        EnzymeResidualContext context{};
        configure_operator_for_case(context.op, input);
        context.x_scale        = input.x_scale;
        context.residual_scale = input.residual_scale;
        return context;
    }

    double scaled_residual_vector_for_enzyme(
        double*       z,
        double*       fvec,
        void*         context_value
    ) noexcept
    {
        auto& context = *static_cast<EnzymeResidualContext*>(context_value);
        const auto x = decode_z_to_x(
            std::span<const double, BenchShape::x_size>{z, BenchShape::x_size},
            context.x_scale
        );

        PackedVector raw{uninitialized};
        context.op.evaluate(
            std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size},
            raw
        );
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            fvec[i] = raw[i] / context.residual_scale[i];
        return 0.0;
    }
#endif

    int pf_residual_z(void* data, int n, const double* z, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != static_cast<int>(BenchShape::x_size))
            return 0;

        auto& context = *static_cast<SolveContext*>(data);
        ++context.evaluations;
        scaled_residual_z_no_count(context, z, fvec);
        return 0;
    }

    void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept
    {
        const auto callback_started = std::chrono::steady_clock::now();
        const auto x = decode_z_to_x(
            std::span<const double, BenchShape::x_size>{z, BenchShape::x_size},
            context.input.x_scale
        );
        PackedVector raw{uninitialized};
        const auto kernel_started = std::chrono::steady_clock::now();
        context.raw_residual(
            std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{raw.data(), BenchShape::x_size}
        );
        context.residual_kernel_ms += elapsed_ms_since(kernel_started);
        const auto scale_started = std::chrono::steady_clock::now();
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            fvec[i] = raw[i] / context.input.residual_scale[i];
        context.residual_scale_ms += elapsed_ms_since(scale_started);
        context.residual_callback_ms += elapsed_ms_since(callback_started);
    }

#ifdef ENABLE_ENZYME
    void fill_enzyme_jvp_z(SolveContext& context, const double* z, const double* v, double* jv)
    {
        std::array<double, BenchShape::x_size> z_primal;
        std::array<double, BenchShape::x_size> z_dot;
        std::array<double, BenchShape::x_size> f_primal{};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
        {
            z_primal[i] = z[i];
            z_dot[i]    = v[i];
        }

        EnzymeResidualContext jvp_context = enzyme_context_for_input(context.input);
        EnzymeResidualContext jvp_context_dot{};
        std::memset(&jvp_context_dot, 0, sizeof(jvp_context_dot));
        (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
            scaled_residual_vector_for_enzyme,
            enzyme::Duplicated<double*>{z_primal.data(), z_dot.data()},
            enzyme::Duplicated<double*>{f_primal.data(), jv},
            enzyme::Duplicated<void*>{static_cast<void*>(&jvp_context), static_cast<void*>(&jvp_context_dot)}
        );
    }

    template <size_t Width>
    void fill_enzyme_jacobian_z_vector(SolveContext& context,
                                       const double* z,
                                       double*       fjac,
                                       int           ldfjac)
    {
        static_assert(Width > 0);
        constexpr size_t n = BenchShape::x_size;
        constexpr size_t lane_stride_bytes = n * sizeof(double);

        std::array<double, n> z_primal;
        for (size_t i = 0; i < n; ++i)
            z_primal[i] = z[i];

        for (size_t first_col = 0; first_col < n; first_col += Width)
        {
            std::array<double, Width * n> z_dot{};
            std::array<double, n>         f_primal{};
            std::array<double, Width * n> f_dot{};
            EnzymeResidualContext         chunk_context = enzyme_context_for_input(context.input);
            std::array<EnzymeResidualContext, Width> chunk_context_dot{};
            std::memset(chunk_context_dot.data(), 0, sizeof(chunk_context_dot));

            const size_t lane_count = std::min(Width, n - first_col);
            for (size_t lane = 0; lane < lane_count; ++lane)
                z_dot[lane * n + first_col + lane] = 1.0;

            __enzyme_fwddiff<void>(
                reinterpret_cast<void*>(scaled_residual_vector_for_enzyme),
                enzyme_width,
                static_cast<int>(Width),
                enzyme_dupv,
                static_cast<int>(lane_stride_bytes),
                z_primal.data(),
                z_dot.data(),
                enzyme_dupv,
                static_cast<int>(lane_stride_bytes),
                f_primal.data(),
                f_dot.data(),
                enzyme_dupv,
                static_cast<int>(sizeof(EnzymeResidualContext)),
                static_cast<void*>(&chunk_context),
                static_cast<void*>(chunk_context_dot.data())
            );

            for (size_t lane = 0; lane < lane_count; ++lane)
            {
                const size_t col = first_col + lane;
                for (size_t row = 0; row < n; ++row)
                    fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[lane * n + row];
            }
        }
        context.jacobian_component_evaluations += static_cast<int>(n);
    }

    void fill_enzyme_jacobian_z_scalar(SolveContext& context,
                                       const double* z,
                                       double*       fjac,
                                       int           ldfjac)
    {
        std::array<double, BenchShape::x_size> z_primal;
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            z_primal[i] = z[i];

        for (size_t col = 0; col < BenchShape::x_size; ++col)
        {
            std::array<double, BenchShape::x_size> z_dot{};
            std::array<double, BenchShape::x_size> f_primal{};
            std::array<double, BenchShape::x_size> f_dot{};
            EnzymeResidualContext                  column_context = enzyme_context_for_input(context.input);
            EnzymeResidualContext                  column_context_dot{};
            std::memset(&column_context_dot, 0, sizeof(column_context_dot));
            z_dot[col] = 1.0;
            (void)enzyme::autodiff<enzyme::Forward, enzyme::Const<double>>(
                scaled_residual_vector_for_enzyme,
                enzyme::Duplicated<double*>{z_primal.data(), z_dot.data()},
                enzyme::Duplicated<double*>{f_primal.data(), f_dot.data()},
                enzyme::Duplicated<void*>{static_cast<void*>(&column_context),
                                          static_cast<void*>(&column_context_dot)}
            );
            for (size_t row = 0; row < BenchShape::x_size; ++row)
                fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[row];
        }
        context.jacobian_component_evaluations += static_cast<int>(BenchShape::x_size);
    }

    void fill_enzyme_jacobian_z(SolveContext& context, const double* z, double* fjac, int ldfjac)
    {
        switch (context.input.enzyme_width)
        {
        case 1:
            fill_enzyme_jacobian_z_scalar(context, z, fjac, ldfjac);
            break;
        case 2:
            fill_enzyme_jacobian_z_vector<2>(context, z, fjac, ldfjac);
            break;
        case 3:
            fill_enzyme_jacobian_z_vector<3>(context, z, fjac, ldfjac);
            break;
        case 4:
            fill_enzyme_jacobian_z_vector<4>(context, z, fjac, ldfjac);
            break;
        case 5:
            fill_enzyme_jacobian_z_vector<5>(context, z, fjac, ldfjac);
            break;
        case 6:
            fill_enzyme_jacobian_z_vector<6>(context, z, fjac, ldfjac);
            break;
        case 8:
            fill_enzyme_jacobian_z_vector<8>(context, z, fjac, ldfjac);
            break;
        case 9:
            fill_enzyme_jacobian_z_vector<9>(context, z, fjac, ldfjac);
            break;
        case 10:
            fill_enzyme_jacobian_z_vector<10>(context, z, fjac, ldfjac);
            break;
        case 12:
            fill_enzyme_jacobian_z_vector<12>(context, z, fjac, ldfjac);
            break;
        case 18:
            fill_enzyme_jacobian_z_vector<18>(context, z, fjac, ldfjac);
            break;
        default:
            throw std::runtime_error("unsupported Enzyme vector width");
        }
    }

    void scaled_residual_z_array_no_count(
        SolveContext&                                      context,
        const std::array<double, BenchShape::x_size>&      z,
        std::array<double, BenchShape::x_size>&            fvec
    ) noexcept
    {
        scaled_residual_z_no_count(context, z.data(), fvec.data());
    }

    JacobianCheck check_enzyme_jacobian_at_initial(SolveContext& context)
    {
        constexpr size_t n = BenchShape::x_size;
        auto             z = encode_x_to_z(context.input.x0, context.input.x_scale);

        std::array<double, n * n> enzyme_jac{};
        SolveContext                 enzyme_context{context.input};
        fill_enzyme_jacobian_z(enzyme_context, z.data(), enzyme_jac.data(), static_cast<int>(n));

        std::array<double, n * n> finite_difference_jac{};
        for (size_t col = 0; col < n; ++col)
        {
            auto z_plus  = z;
            auto z_minus = z;
            const double step = 1.0e-6 * std::max(1.0, std::abs(z[col]));
            z_plus[col] += step;
            z_minus[col] -= step;

            std::array<double, n> f_plus{};
            std::array<double, n> f_minus{};
            (void)scaled_residual_z_array_no_count(context, z_plus, f_plus);
            (void)scaled_residual_z_array_no_count(context, z_minus, f_minus);
            for (size_t row = 0; row < n; ++row)
                finite_difference_jac[row + n * col] = (f_plus[row] - f_minus[row]) / (2.0 * step);
        }

        JacobianCheck check{};
        for (size_t i = 0; i < n * n; ++i)
        {
            const double enzyme_value = enzyme_jac[i];
            const double fd_value     = finite_difference_jac[i];
            const double diff         = std::abs(enzyme_value - fd_value);
            check.enzyme_norm += enzyme_value * enzyme_value;
            check.finite_difference_norm += fd_value * fd_value;
            if (diff > check.max_abs_diff)
                check.max_abs_diff = diff;
            const double scale = std::max({1.0, std::abs(enzyme_value), std::abs(fd_value)});
            const double rel   = diff / scale;
            if (rel > check.max_rel_diff)
                check.max_rel_diff = rel;
        }
        check.enzyme_norm            = std::sqrt(check.enzyme_norm);
        check.finite_difference_norm = std::sqrt(check.finite_difference_norm);
        return check;
    }

    int pf_residual_jacobian_z(
        void*         data,
        int           n,
        const double* z,
        double*       fvec,
        double*       fjac,
        int           ldfjac,
        int           iflag
    )
    {
        if (n != static_cast<int>(BenchShape::x_size))
            return 0;
        if (iflag == 1)
            return pf_residual_z(data, n, z, fvec, iflag);
        if (iflag == 2)
        {
            auto& context = *static_cast<SolveContext*>(data);
            const auto started = std::chrono::steady_clock::now();
            fill_enzyme_jacobian_z(context, z, fjac, ldfjac);
            context.jacobian_callback_ms += elapsed_ms_since(started);
        }
        return 0;
    }
#endif

    struct ScaledResidualProblem
    {
        static constexpr size_t equations = BenchShape::x_size;
        static constexpr size_t variables = BenchShape::x_size;

        SolveContext* context = nullptr;

        void operator()(const double* z, double* fvec) const noexcept
        {
            (void)scaled_residual_z_no_count(*context, z, fvec);
        }

#ifdef ENABLE_ENZYME
        void jacobian(const double* z, double* jacobian) const
        {
            constexpr size_t n = BenchShape::x_size;
            std::array<double, n * n> column_major{};
            fill_enzyme_jacobian_z(*context, z, column_major.data(), static_cast<int>(n));
            for (size_t row = 0; row < n; ++row)
                for (size_t col = 0; col < n; ++col)
                    jacobian[row * n + col] = column_major[row + n * col];
        }

        void jvp(const double* z, const double* v, double* jv) const
        {
            fill_enzyme_jvp_z(*context, z, v, jv);
        }
#endif
    };

    struct ScaledResidualOnlyProblem
    {
        static constexpr size_t equations = BenchShape::x_size;
        static constexpr size_t variables = BenchShape::x_size;

        SolveContext* context = nullptr;

        void operator()(const double* z, double* fvec) const noexcept
        {
            (void)scaled_residual_z_no_count(*context, z, fvec);
        }
    };

    struct SundialsSolveData
    {
        SolveContext* context = nullptr;
        int evaluations = 0;
        int jacobian_evaluations = 0;
        int jvp_evaluations = 0;
    };

    int sundials_residual_callback(N_Vector u, N_Vector fval, void* user_data)
    {
        auto&  data   = *static_cast<SundialsSolveData*>(user_data);
        auto*  z      = N_VGetArrayPointer(u);
        auto*  f_data = N_VGetArrayPointer(fval);
        ++data.evaluations;
        scaled_residual_z_no_count(*data.context, z, f_data);
        return 0;
    }

    void fill_finite_difference_jacobian_z(SolveContext& context, const double* z, double* jacobian)
    {
        constexpr size_t n = BenchShape::x_size;
        std::array<double, n> z_plus;
        std::array<double, n> f_base;
        std::array<double, n> f_plus;
        std::copy(z, z + n, z_plus.begin());
        (void)scaled_residual_z_no_count(context, z, f_base.data());
        for (size_t col = 0; col < n; ++col)
        {
            const double saved = z_plus[col];
            const double step  = 1.0e-7 * std::max(1.0, std::abs(saved));
            z_plus[col]        = saved + step;
            (void)scaled_residual_z_no_count(context, z_plus.data(), f_plus.data());
            z_plus[col] = saved;
            for (size_t row = 0; row < n; ++row)
                jacobian[row * n + col] = (f_plus[row] - f_base[row]) / step;
        }
    }

    void fill_finite_difference_jvp_z(SolveContext& context, const double* z, const double* v, double* jv)
    {
        constexpr size_t n = BenchShape::x_size;
        std::array<double, n> z_plus;
        std::array<double, n> f_base;
        std::array<double, n> f_plus;
        std::copy(z, z + n, z_plus.begin());
        const double v_norm = norm2(std::span<const double, n>{v, n});
        if (v_norm <= 0.0)
        {
            std::fill(jv, jv + n, 0.0);
            return;
        }
        const double z_norm = norm2(std::span<const double, n>{z, n});
        const double step   = std::sqrt(1.0e-12) * (1.0 + z_norm) / v_norm;
        (void)scaled_residual_z_no_count(context, z, f_base.data());
        for (size_t i = 0; i < n; ++i)
            z_plus[i] += step * v[i];
        (void)scaled_residual_z_no_count(context, z_plus.data(), f_plus.data());
        for (size_t i = 0; i < n; ++i)
            jv[i] = (f_plus[i] - f_base[i]) / step;
    }

    int sundials_dense_jacobian_callback(
        N_Vector u, N_Vector, SUNMatrix jacobian, void* user_data, N_Vector, N_Vector)
    {
        constexpr size_t n    = BenchShape::x_size;
        auto&            data = *static_cast<SundialsSolveData*>(user_data);
        auto*            z    = N_VGetArrayPointer(u);
        ++data.jacobian_evaluations;
        const auto started = std::chrono::steady_clock::now();

#ifdef ENABLE_ENZYME
        std::array<double, n * n> column_major{};
        fill_enzyme_jacobian_z(*data.context, z, column_major.data(), static_cast<int>(n));
        for (size_t col = 0; col < n; ++col)
            for (size_t row = 0; row < n; ++row)
                SM_ELEMENT_D(jacobian, static_cast<sunindextype>(row), static_cast<sunindextype>(col)) =
                    column_major[row + n * col];
#else
        std::array<double, n * n> row_major{};
        fill_finite_difference_jacobian_z(*data.context, z, row_major.data());
        for (size_t col = 0; col < n; ++col)
            for (size_t row = 0; row < n; ++row)
                SM_ELEMENT_D(jacobian, static_cast<sunindextype>(row), static_cast<sunindextype>(col)) =
                    row_major[row * n + col];
#endif
        data.context->jacobian_callback_ms += elapsed_ms_since(started);
        return 0;
    }

    int sundials_jvp_callback(N_Vector v, N_Vector jv, N_Vector u, booleantype*, void* user_data)
    {
        auto& data = *static_cast<SundialsSolveData*>(user_data);
        ++data.jvp_evaluations;
        const auto started = std::chrono::steady_clock::now();
#ifdef ENABLE_ENZYME
        fill_enzyme_jvp_z(
            *data.context,
            N_VGetArrayPointer(u),
            N_VGetArrayPointer(v),
            N_VGetArrayPointer(jv)
        );
#else
        fill_finite_difference_jvp_z(
            *data.context,
            N_VGetArrayPointer(u),
            N_VGetArrayPointer(v),
            N_VGetArrayPointer(jv)
        );
#endif
        data.context->jvp_callback_ms += elapsed_ms_since(started);
        return 0;
    }

    template <typename Context>
    int jacobian_evaluation_count(const Context& context) noexcept
    {
        if constexpr (requires { context.jacobian_evaluations; })
            return context.jacobian_evaluations;
        else
            return 0;
    }

    template <typename Context>
    int jvp_evaluation_count(const Context& context) noexcept
    {
        if constexpr (requires { context.jvp_evaluations; })
            return context.jvp_evaluations;
        else
            return 0;
    }

    template <typename Context>
    int linear_iteration_count(const Context& context) noexcept
    {
        if constexpr (requires { context.linear_iterations; })
            return context.linear_iterations;
        else
            return 0;
    }

    void fill_solve_result_from_z(SolveContext& context,
                                  SolveResult&  result,
                                  const double* z,
                                  int           info,
                                  int           nfev,
                                  int           njev,
                                  int           callbacks)
    {
        result.info      = info;
        result.nfev      = nfev;
        result.njev      = njev;
        result.callbacks = callbacks;
        result.x         = decode_z_to_x(
            std::span<const double, BenchShape::x_size>{z, BenchShape::x_size},
            context.input.x_scale
        );
        const auto final_residual_started = std::chrono::steady_clock::now();
        context.raw_residual(
            std::span<const double, BenchShape::x_size>{result.x.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{result.raw.data(), BenchShape::x_size}
        );
        result.final_residual_ms = elapsed_ms_since(final_residual_started);
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
        result.raw_norm = norm2(std::span<const double, BenchShape::x_size>{
            result.raw.data(),
            BenchShape::x_size,
        });
        result.scaled_norm = norm2(std::span<const double, BenchShape::x_size>{
            result.scaled.data(),
            BenchShape::x_size,
        });
        result.residual_callback_ms = context.residual_callback_ms;
        result.residual_kernel_ms   = context.residual_kernel_ms;
        result.residual_scale_ms    = context.residual_scale_ms;
        result.jacobian_callback_ms = context.jacobian_callback_ms;
        result.jvp_callback_ms      = context.jvp_callback_ms;
        result.linear_solve_ms      = context.linear_solve_ms;
        result.alpha    = {context.op.workspace.source_runtime.alpha1, context.op.workspace.source_runtime.alpha2};
        result.accepted = result.raw_norm <= veqpy_acceptance_threshold();
    }

    SolveResult run_hybrd_once(SolveContext& context)
    {
        context.reset_solve_counters();
        auto z = encode_x_to_z(context.input.x0, context.input.x_scale);
        PackedVector fvec{uninitialized};

        constexpr int n  = static_cast<int>(BenchShape::x_size);
        constexpr int ml = n - 1;
        constexpr int mu = n - 1;
        constexpr int lr = n * (n + 1) / 2;
        std::array<double, BenchShape::x_size> diag;
        std::array<double, BenchShape::x_size * BenchShape::x_size> fjac;
        std::array<double, static_cast<size_t>(lr)>                 r;
        std::array<double, BenchShape::x_size>                      qtf;
        std::array<double, BenchShape::x_size>                      wa1;
        std::array<double, BenchShape::x_size>                      wa2;
        std::array<double, BenchShape::x_size>                      wa3;
        std::array<double, BenchShape::x_size>                      wa4;
        int                                                         nfev = 0;
        const int info = hybrd(
            pf_residual_z,
            &context,
            n,
            z.data(),
            fvec.data(),
            veqpy_max_residual,
            veqpy_maxfev,
            ml,
            mu,
            veqpy_hybr_eps,
            diag.data(),
            veqpy_hybr_mode,
            veqpy_hybr_factor,
            veqpy_hybr_nprint,
            &nfev,
            fjac.data(),
            n,
            r.data(),
            lr,
            qtf.data(),
            wa1.data(),
            wa2.data(),
            wa3.data(),
            wa4.data()
        );

        SolveResult result{};
        fill_solve_result_from_z(context, result, z.data(), info, nfev, 0, context.evaluations);
        result.jacobian_component_evaluations = context.jacobian_component_evaluations;
        return result;
    }

#ifdef ENABLE_ENZYME
    SolveResult run_hybrj_once(SolveContext& context)
    {
        context.reset_solve_counters();
        auto z = encode_x_to_z(context.input.x0, context.input.x_scale);
        PackedVector fvec{uninitialized};

        constexpr int n  = static_cast<int>(BenchShape::x_size);
        constexpr int lr = n * (n + 1) / 2;
        std::array<double, BenchShape::x_size>                      diag;
        std::array<double, BenchShape::x_size * BenchShape::x_size> fjac;
        std::array<double, static_cast<size_t>(lr)>                 r;
        std::array<double, BenchShape::x_size>                      qtf;
        std::array<double, BenchShape::x_size>                      wa1;
        std::array<double, BenchShape::x_size>                      wa2;
        std::array<double, BenchShape::x_size>                      wa3;
        std::array<double, BenchShape::x_size>                      wa4;
        int                                                         nfev = 0;
        int                                                         njev = 0;
        const int info = hybrj(
            pf_residual_jacobian_z,
            &context,
            n,
            z.data(),
            fvec.data(),
            fjac.data(),
            n,
            veqpy_max_residual,
            veqpy_maxfev,
            diag.data(),
            veqpy_hybr_mode,
            veqpy_hybr_factor,
            veqpy_hybr_nprint,
            &nfev,
            &njev,
            r.data(),
            lr,
            qtf.data(),
            wa1.data(),
            wa2.data(),
            wa3.data(),
            wa4.data()
        );

        SolveResult result{};
        fill_solve_result_from_z(context, result, z.data(), info, nfev, njev, context.evaluations);
        result.jacobian_component_evaluations = context.jacobian_component_evaluations;
        return result;
    }
#endif

    template <typename Policy>
    SolveResult run_nonlinear_policy_once(SolveContext& context)
    {
        context.reset_solve_counters();
        const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
        tensor::Vector<double, BenchShape::x_size> z{uninitialized};
        std::copy(encoded.begin(), encoded.end(), z.begin());

        ScaledResidualProblem problem{&context};
        auto                  solver = nonlinear::make_solver<Policy>(problem);
        solver.context.tolerance     = veqpy_max_residual;
        if constexpr (requires { solver.context.max_iterations; })
            solver.context.max_iterations = veqpy_maxfev;
        if constexpr (requires { solver.context.max_dimension; })
            solver.context.max_dimension = static_cast<int>(BenchShape::x_size);

        solver.optimize_inplace(z);

        SolveResult result{};
        fill_solve_result_from_z(
            context,
            result,
            z.data(),
            solver.context.info,
            solver.context.evaluations,
            jacobian_evaluation_count(solver.context),
            solver.context.evaluations
        );
        result.jacobian_component_evaluations = context.jacobian_component_evaluations;
        result.jvp_evaluations                = jvp_evaluation_count(solver.context);
        result.linear_iterations              = linear_iteration_count(solver.context);
        return result;
    }

    template <typename Policy>
    SolveResult run_nonlinear_residual_policy_once(SolveContext& context)
    {
        context.reset_solve_counters();
        const auto encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
        tensor::Vector<double, BenchShape::x_size> z{uninitialized};
        std::copy(encoded.begin(), encoded.end(), z.begin());

        ScaledResidualOnlyProblem problem{&context};
        auto                      solver = nonlinear::make_solver<Policy>(problem);
        solver.context.tolerance         = veqpy_max_residual;

        solver.optimize_inplace(z);

        SolveResult result{};
        fill_solve_result_from_z(
            context,
            result,
            z.data(),
            solver.context.info,
            solver.context.evaluations,
            jacobian_evaluation_count(solver.context),
            solver.context.evaluations
        );
        return result;
    }

    void cleanup_sundials(
        void*& kin_mem, N_Vector& y, N_Vector& scale, SUNMatrix& matrix, SUNLinearSolver& linear_solver, SUNContext& sunctx)
    {
        if (linear_solver)
            SUNLinSolFree(linear_solver);
        if (matrix)
            SUNMatDestroy(matrix);
        if (scale)
            N_VDestroy(scale);
        if (y)
            N_VDestroy(y);
        if (kin_mem)
            KINFree(&kin_mem);
        if (sunctx)
            SUNContext_Free(&sunctx);
        linear_solver = nullptr;
        matrix        = nullptr;
        scale         = nullptr;
        y             = nullptr;
        kin_mem       = nullptr;
        sunctx        = nullptr;
    }

    SolveResult run_sundials_once(SolveContext& context, SolverKind solver)
    {
        constexpr size_t n = BenchShape::x_size;
        context.reset_solve_counters();
        auto z = encode_x_to_z(context.input.x0, context.input.x_scale);

        SUNContext      sunctx        = nullptr;
        N_Vector        y             = nullptr;
        N_Vector        scale         = nullptr;
        void*           kin_mem       = nullptr;
        SUNMatrix       matrix        = nullptr;
        SUNLinearSolver linear_solver = nullptr;
        SundialsSolveData data{&context};

        int info = SUNContext_Create(nullptr, &sunctx);
        if (info != 0)
        {
            SolveResult result{};
            fill_solve_result_from_z(context, result, z.data(), KIN_CONTEXT_ERR, 0, 0, 0);
            return result;
        }

        y     = N_VMake_Serial(static_cast<sunindextype>(n), z.data(), sunctx);
        scale = N_VNew_Serial(static_cast<sunindextype>(n), sunctx);
        if (!y || !scale)
        {
            cleanup_sundials(kin_mem, y, scale, matrix, linear_solver, sunctx);
            SolveResult result{};
            fill_solve_result_from_z(context, result, z.data(), KIN_MEM_FAIL, 0, 0, 0);
            return result;
        }
        N_VConst(1.0, scale);

        kin_mem = KINCreate(sunctx);
        if (!kin_mem)
        {
            cleanup_sundials(kin_mem, y, scale, matrix, linear_solver, sunctx);
            SolveResult result{};
            fill_solve_result_from_z(context, result, z.data(), KIN_MEM_FAIL, 0, 0, 0);
            return result;
        }

        info = KINInit(kin_mem, sundials_residual_callback, y);
        if (info == KIN_SUCCESS)
            info = KINSetUserData(kin_mem, &data);

        if (info == KIN_SUCCESS && solver == SolverKind::SundialsNewtonRaphson)
        {
            matrix        = SUNDenseMatrix(static_cast<sunindextype>(n), static_cast<sunindextype>(n), sunctx);
            linear_solver = matrix ? SUNLinSol_Dense(y, matrix, sunctx) : nullptr;
            if (!matrix || !linear_solver)
                info = KIN_MEM_FAIL;
            if (info == KIN_SUCCESS)
                info = KINSetLinearSolver(kin_mem, linear_solver, matrix);
            if (info == KINLS_SUCCESS)
                info = KINSetJacFn(kin_mem, sundials_dense_jacobian_callback);
        }
        else if (info == KIN_SUCCESS && solver == SolverKind::SundialsNewtonKrylov)
        {
            linear_solver = SUNLinSol_SPGMR(y, PREC_NONE, static_cast<int>(n), sunctx);
            if (!linear_solver)
                info = KIN_MEM_FAIL;
            if (info == KIN_SUCCESS)
                info = KINSetLinearSolver(kin_mem, linear_solver, nullptr);
            if (info == KINLS_SUCCESS)
                info = KINSetJacTimesVecFn(kin_mem, sundials_jvp_callback);
        }

        if (info == KIN_SUCCESS)
        {
            (void)KINSetFuncNormTol(kin_mem, veqpy_max_residual);
            (void)KINSetScaledStepTol(kin_mem, veqpy_max_residual);
            (void)KINSetNumMaxIters(kin_mem, veqpy_maxfev);
            if (solver == SolverKind::SundialsNewtonKrylov)
            {
                (void)KINSetEtaForm(kin_mem, KIN_ETACONSTANT);
                (void)KINSetEtaConstValue(kin_mem, 0.1);
            }
            else
            {
                (void)KINSetEtaForm(kin_mem, KIN_ETACHOICE2);
            }
            (void)KINSetPrintLevel(kin_mem, 0);
            info = KINSol(kin_mem, y, KIN_LINESEARCH, scale, scale);
        }

        long int nfev = 0;
        long int njev = 0;
        long int linear_iterations = 0;
        long int jvp_evaluations = 0;
        if (kin_mem)
        {
            (void)KINGetNumFuncEvals(kin_mem, &nfev);
            (void)KINGetNumJacEvals(kin_mem, &njev);
            (void)KINGetNumLinIters(kin_mem, &linear_iterations);
            (void)KINGetNumJtimesEvals(kin_mem, &jvp_evaluations);
        }

        SolveResult result{};
        fill_solve_result_from_z(
            context,
            result,
            z.data(),
            info,
            static_cast<int>(nfev),
            static_cast<int>(njev),
            data.evaluations
        );
        result.jacobian_component_evaluations = context.jacobian_component_evaluations;
        result.jvp_evaluations                = static_cast<int>(jvp_evaluations);
        result.linear_iterations              = static_cast<int>(linear_iterations);
        cleanup_sundials(kin_mem, y, scale, matrix, linear_solver, sunctx);
        return result;
    }

    SolveResult run_solver_once(SolveContext& context)
    {
        if (context.input.solver == SolverKind::ResidualOnly)
            return run_hybrd_once(context);
        if (context.input.solver == SolverKind::LevenbergMarquardt)
            return run_nonlinear_residual_policy_once<nonlinear::LevenbergMarquardt>(context);
        if (context.input.solver == SolverKind::Newton)
            return run_nonlinear_policy_once<nonlinear::Newton>(context);
        if (context.input.solver == SolverKind::NewtonKrylov)
            return run_nonlinear_policy_once<nonlinear::NewtonKrylov>(context);
        if (context.input.solver == SolverKind::NewtonRaphson)
            return run_nonlinear_policy_once<nonlinear::NewtonRaphson>(context);
        if (context.input.solver == SolverKind::Powell)
            return run_nonlinear_residual_policy_once<nonlinear::Powell>(context);
        if (context.input.solver == SolverKind::SundialsNewtonKrylov ||
            context.input.solver == SolverKind::SundialsNewtonRaphson)
            return run_sundials_once(context, context.input.solver);
#ifdef ENABLE_ENZYME
        return run_hybrj_once(context);
#else
        throw std::runtime_error("Enzyme Jacobian solver requested without ENABLE_ENZYME");
#endif
    }

    nlohmann::json solve_result_json(const SolveResult& result)
    {
        return {
            {"accepted_by_veqpy", result.accepted},
            {"x", json_array(result.x)},
            {"raw_residual", json_array(result.raw)},
            {"scaled_residual", json_array(result.scaled)},
            {"alpha", json_array(result.alpha)},
            {"raw_norm", result.raw_norm},
            {"scaled_norm", result.scaled_norm},
            {"info", result.info},
            {"nfev", result.nfev},
            {"njev", result.njev},
            {"callback_evaluations", result.callbacks},
            {"jacobian_component_evaluations", result.jacobian_component_evaluations},
            {"jvp_evaluations", result.jvp_evaluations},
            {"linear_iterations", result.linear_iterations},
            {"callback_timing_ms",
             {
                 {"residual_total", result.residual_callback_ms},
                 {"residual_kernel", result.residual_kernel_ms},
                 {"residual_scale", result.residual_scale_ms},
                 {"final_residual", result.final_residual_ms},
                 {"jacobian_total", result.jacobian_callback_ms},
                 {"jvp_total", result.jvp_callback_ms},
                 {"linear_solve", result.linear_solve_ms},
             }},
        };
    }

    nlohmann::json jacobian_check_json(const JacobianCheck& check)
    {
        return {
            {"enzyme_norm", check.enzyme_norm},
            {"finite_difference_norm", check.finite_difference_norm},
            {"max_abs_diff", check.max_abs_diff},
            {"max_rel_diff", check.max_rel_diff},
        };
    }

    double mean(const std::vector<double>& values)
    {
        if (values.empty())
            return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    }

    double stddev(const std::vector<double>& values, double avg)
    {
        if (values.empty())
            return 0.0;
        double total = 0.0;
        for (double value : values)
        {
            const double delta = value - avg;
            total += delta * delta;
        }
        return std::sqrt(total / static_cast<double>(values.size()));
    }

    double percentile_sorted(const std::vector<double>& sorted, double percentile)
    {
        if (sorted.empty())
            return 0.0;
        const double pos   = (static_cast<double>(sorted.size()) - 1.0) * percentile;
        const auto   lower = static_cast<size_t>(std::floor(pos));
        const auto   upper = static_cast<size_t>(std::ceil(pos));
        if (lower == upper)
            return sorted[lower];
        const double fraction = pos - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    }

    nlohmann::json timing_json(const std::vector<double>& samples)
    {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double avg = mean(samples);
        return {
            {"repeat_count", samples.size()},
            {"samples_ms", json_array(samples)},
            {"avg_ms", avg},
            {"median_ms", percentile_sorted(sorted, 0.50)},
            {"p95_ms", percentile_sorted(sorted, 0.95)},
            {"min_ms", sorted.empty() ? 0.0 : sorted.front()},
            {"max_ms", sorted.empty() ? 0.0 : sorted.back()},
            {"std_ms", stddev(samples, avg)},
        };
    }
} // namespace

int run(int argc, char** argv)
{
    try
    {
        int repeat = 10;
        int warmup = 1;
        SolverKind solver = SolverKind::ResidualOnly;
        int enzyme_jacobian_width = 1;
        bool enable_jacobian_check = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--repeat")
            {
                if (++i >= argc)
                    throw std::runtime_error("--repeat requires a value");
                repeat = parse_nonnegative_int("--repeat", argv[i]);
            }
            else if (arg == "--warmup")
            {
                if (++i >= argc)
                    throw std::runtime_error("--warmup requires a value");
                warmup = parse_nonnegative_int("--warmup", argv[i]);
            }
            else if (arg == "--solver")
            {
                if (++i >= argc)
                    throw std::runtime_error("--solver requires a value");
                solver = parse_solver_kind(argv[i]);
            }
            else if (arg == "--enzyme-width")
            {
                if (++i >= argc)
                    throw std::runtime_error("--enzyme-width requires a value");
                enzyme_jacobian_width = parse_nonnegative_int("--enzyme-width", argv[i]);
                if (!supported_enzyme_width(enzyme_jacobian_width))
                    throw std::runtime_error(
                        "--enzyme-width must be one of 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 18"
                    );
            }
            else if (arg == "--jacobian-check")
            {
#ifdef ENABLE_ENZYME
                enable_jacobian_check = true;
#else
                throw std::runtime_error("--jacobian-check requires the clang-enzyme-release preset");
#endif
            }
            else if (arg == "--help" || arg == "-h")
            {
                std::cout << "usage: " << argv[0]
                          << " [--repeat N] [--warmup N]"
                          << " [--solver residual|enzyme|lm|newton|nk|nr|powell|sundials-nk|sundials-nr]"
                          << " [--enzyme-width 1|2|3|4|5|6|8|9|10|12|18]"
                          << " [--jacobian-check]\n";
                return EXIT_SUCCESS;
            }
            else
            {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        if (solver == SolverKind::ResidualOnly)
            enzyme_jacobian_width = 1;

        CaseInput    input = build_inline_case(repeat, warmup, solver, enzyme_jacobian_width);
        SolveContext context{input};

        PackedVector initial_raw{uninitialized};
        context.raw_residual(
            std::span<const double, BenchShape::x_size>{input.x0.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{initial_raw.data(), BenchShape::x_size}
        );
        context.input.residual_scale = build_block_rms_residual_scale(initial_raw);
        input.residual_scale         = context.input.residual_scale;
        PackedVector initial_scaled{uninitialized};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            initial_scaled[i] = initial_raw[i] / input.residual_scale[i];

        nlohmann::json jacobian_check_report = nullptr;
#ifdef ENABLE_ENZYME
        if (enable_jacobian_check)
            jacobian_check_report = jacobian_check_json(check_enzyme_jacobian_at_initial(context));
#else
        (void)enable_jacobian_check;
#endif

        for (int i = 0; i < input.warmup; ++i)
            (void)run_solver_once(context);

        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(input.repeat));
        SolveResult final{};
        for (int i = 0; i < input.repeat; ++i)
        {
            const auto started = std::chrono::steady_clock::now();
            final              = run_solver_once(context);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            samples_ms.push_back(
                std::chrono::duration<double, std::milli>(elapsed).count()
            );
        }
        if (input.repeat == 0)
            final = run_solver_once(context);

        const nlohmann::json report = {
            {"case_name", input.case_name},
            {"route", "PF/psin/uniform/Ip"},
            {"source_topology",
             {
                 {"route", "PF"},
                 {"coordinate", "psin"},
                 {"nodes", "uniform"},
                 {"constraint", "Ip"},
             }},
            {"x_size", BenchShape::x_size},
            {"grid",
             {
                 {"Nr", BenchGrid::radial_nodes},
                 {"Nt", BenchGrid::theta_rows},
                 {"L_max", BenchShape::L_max},
                 {"M_max", BenchShape::M_max},
                 {"K_max", BenchShape::K_max},
                 {"quadrature_scheme", "legendre"},
                 {"calculus_scheme", "spectral"},
             }},
            {"solver",
             {
                 {"method", solver_method(input.solver)},
                 {"entrypoint", solver_entrypoint(input.solver)},
                 {"jacobian", solver_jacobian(input)},
                 {"enzyme_width",
                  input.solver == SolverKind::EnzymeJacobian
                      ? nlohmann::json(input.enzyme_width)
                      : nlohmann::json(nullptr)},
                 {"max_residual", veqpy_max_residual},
                 {"acceptance_threshold", veqpy_acceptance_threshold()},
                 {"requested_max_evaluations", veqpy_requested_max_evaluations},
                 {"maxfev", veqpy_maxfev},
                 {"eps", veqpy_hybr_eps},
                 {"factor", veqpy_hybr_factor},
                 {"diag_mode", veqpy_hybr_mode},
             }},
            {"source",
             {
                 {"scaled_heat", json_array(input.heat)},
                 {"scaled_current", json_array(input.current)},
             }},
            {"normalization",
             {
                 {"x_scale", json_array(input.x_scale)},
                 {"residual_scale", json_array(input.residual_scale)},
                 {"x_scale_builder", "VEQPy _build_x_block_scale_vector equivalent"},
                 {"residual_scale_builder", "fast/block_rms initial residual block RMS"},
                 {"unknown_space", "z = x / x_scale"},
             }},
            {"constraints",
             {
                 {"scaled_Ip", input.Ip},
             }},
            {"jacobian_check", jacobian_check_report},
            {"timing", timing_json(samples_ms)},
            {"initial",
             {
                 {"x", json_array(input.x0)},
                 {"policy", "benchmark.py robust zero profile coefficients"},
                 {"raw_residual", json_array(initial_raw)},
                 {"scaled_residual", json_array(initial_scaled)},
                 {"raw_norm",
                  norm2(std::span<const double, BenchShape::x_size>{
                      initial_raw.data(),
                      BenchShape::x_size,
                  })},
             }},
            {"final", solve_result_json(final)},
            {"success", final.accepted && solver_info_succeeded(input.solver, final.info)},
        };

        std::cout << report.dump(2) << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "veqlib_main --mode solve: " << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}

} // namespace veqlib_pf_psin_uniform_benchmark_cli
