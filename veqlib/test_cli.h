#pragma once

// ---- PF/psin/uniform/Ip solve benchmark CLI ----
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
#include <string_view>
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
#include "source/pf_psin_uniform_ip.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_pf_psin_uniform_benchmark_cli
{

namespace
{
    using grid::Legendre;
    using grid::Spectral;
    using source::PfPsinUniformIpOperator;
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
        using Operator = PfPsinUniformIpOperator<Shape, Grid, Source>;
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

    enum class ScanPolicy
    {
        Cold,
        Warm,
        Secant,
        All,
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

    struct ScanConfig
    {
        int        points        = 0;
        double     relative_step = 5.0e-3;
        ScanPolicy policy        = ScanPolicy::All;
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

    struct InitialResidual
    {
        PackedVector raw{uninitialized};
        PackedVector scaled{uninitialized};
        double       raw_norm = 0.0;
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

    double parse_finite_double(const char* flag, const char* value)
    {
        char*        end    = nullptr;
        const double parsed = std::strtod(value, &end);
        std::string  text{value};
        for (char& ch : text)
        {
            if (ch >= 'A' && ch <= 'Z')
                ch = static_cast<char>(ch - 'A' + 'a');
        }
        if (end == value || *end != '\0' || text.find("inf") != std::string::npos ||
            text.find("nan") != std::string::npos)
            throw std::runtime_error(std::string{"invalid "} + flag + " value: " + value);
        return parsed;
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

    ScanPolicy parse_scan_policy(const char* value)
    {
        const std::string policy{value};
        if (policy == "cold")
            return ScanPolicy::Cold;
        if (policy == "warm" || policy == "warm-start")
            return ScanPolicy::Warm;
        if (policy == "secant" || policy == "predictor")
            return ScanPolicy::Secant;
        if (policy == "all")
            return ScanPolicy::All;
        throw std::runtime_error("invalid --scan-policy value: " + policy);
    }

    constexpr const char* scan_policy_name(ScanPolicy policy) noexcept
    {
        switch (policy)
        {
        case ScanPolicy::Cold:
            return "cold";
        case ScanPolicy::Warm:
            return "warm";
        case ScanPolicy::Secant:
            return "secant";
        case ScanPolicy::All:
            return "all";
        }
        return "unknown";
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

    InitialResidual prepare_initial_residual(SolveContext& context) noexcept
    {
        InitialResidual initial{};
        context.raw_residual(
            std::span<const double, BenchShape::x_size>{
                context.input.x0.data(),
                BenchShape::x_size,
            },
            std::span<double, BenchShape::x_size>{initial.raw.data(), BenchShape::x_size}
        );
        context.input.residual_scale = build_block_rms_residual_scale(initial.raw);
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            initial.scaled[i] = initial.raw[i] / context.input.residual_scale[i];
        initial.raw_norm = norm2(std::span<const double, BenchShape::x_size>{
            initial.raw.data(),
            BenchShape::x_size,
        });
        return initial;
    }

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

    nlohmann::json grid_json()
    {
        return {
            {"Nr", BenchGrid::radial_nodes},
            {"Nt", BenchGrid::theta_rows},
            {"L_max", BenchShape::L_max},
            {"M_max", BenchShape::M_max},
            {"K_max", BenchShape::K_max},
            {"quadrature_scheme", "legendre"},
            {"calculus_scheme", "spectral"},
        };
    }

    nlohmann::json source_topology_json()
    {
        return {
            {"route", "PF"},
            {"coordinate", "psin"},
            {"nodes", "uniform"},
            {"constraint", "Ip"},
        };
    }

    nlohmann::json solver_config_json(const CaseInput& input)
    {
        return {
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
        };
    }

    nlohmann::json normalization_json(const CaseInput& input)
    {
        return {
            {"x_scale", json_array(input.x_scale)},
            {"residual_scale", json_array(input.residual_scale)},
            {"x_scale_builder", "VEQPy _build_x_block_scale_vector equivalent"},
            {"residual_scale_builder", "fast/block_rms initial residual block RMS"},
            {"unknown_space", "z = x / x_scale"},
        };
    }

    nlohmann::json initial_residual_json(const CaseInput& input, const InitialResidual& initial)
    {
        return {
            {"x", json_array(input.x0)},
            {"policy", "benchmark.py robust zero profile coefficients or scan predictor"},
            {"raw_residual", json_array(initial.raw)},
            {"scaled_residual", json_array(initial.scaled)},
            {"raw_norm", initial.raw_norm},
        };
    }

    void rebuild_scanned_case_scales(CaseInput& input) noexcept
    {
        input.x_scale = build_x_block_scale_vector<BenchShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);
    }

    nlohmann::json run_parameter_scan_policy(const CaseInput& base_input,
                                             ScanPolicy       policy,
                                             int              point_count,
                                             double           relative_step)
    {
        if (point_count <= 0)
            throw std::runtime_error("--scan-points must be positive when parameter scan is enabled");

        nlohmann::json       points_json = nlohmann::json::array();
        std::vector<double>  samples_ms;
        samples_ms.reserve(static_cast<size_t>(point_count));

        std::array<double, BenchShape::x_size> previous_x{};
        std::array<double, BenchShape::x_size> previous_previous_x{};
        double                                previous_ip          = 0.0;
        double                                previous_previous_ip = 0.0;
        bool                                  have_previous        = false;
        bool                                  have_previous_previous = false;

        int cold_count                = 0;
        int warm_count                = 0;
        int secant_count              = 0;
        int predictor_fallback_count  = 0;
        int success_count             = 0;
        int nfev_total                = 0;
        int callback_total            = 0;

        for (int index = 0; index < point_count; ++index)
        {
            CaseInput point_input = base_input;
            point_input.repeat    = 1;
            point_input.warmup    = 0;
            const double ip_scale = 1.0 + relative_step * static_cast<double>(index);
            if (!(ip_scale > 0.0))
                throw std::runtime_error("scan relative step produced a non-positive Ip scale");
            point_input.Ip = base_input.Ip * ip_scale;

            const char* initial_policy  = "cold";
            const char* fallback_reason = nullptr;
            if (policy == ScanPolicy::Warm)
            {
                if (have_previous)
                {
                    point_input.x0 = previous_x;
                    initial_policy = "warm";
                }
            }
            else if (policy == ScanPolicy::Secant)
            {
                if (have_previous && have_previous_previous)
                {
                    const double old_delta = previous_ip - previous_previous_ip;
                    if (std::abs(old_delta) > 0.0)
                    {
                        const double ratio = (point_input.Ip - previous_ip) / old_delta;
                        for (size_t i = 0; i < BenchShape::x_size; ++i)
                        {
                            point_input.x0[i] =
                                previous_x[i] + ratio * (previous_x[i] - previous_previous_x[i]);
                        }
                        initial_policy = "secant";
                    }
                    else
                    {
                        point_input.x0 = previous_x;
                        initial_policy = "warm";
                        fallback_reason = "zero_previous_parameter_delta";
                    }
                }
                else if (have_previous)
                {
                    point_input.x0  = previous_x;
                    initial_policy  = "warm";
                    fallback_reason = "insufficient_history";
                }
                else
                {
                    fallback_reason = "insufficient_history";
                }
            }

            if (std::string_view{initial_policy} == "secant")
                ++secant_count;
            else if (std::string_view{initial_policy} == "warm")
                ++warm_count;
            else
                ++cold_count;
            if (policy == ScanPolicy::Secant && fallback_reason)
                ++predictor_fallback_count;

            rebuild_scanned_case_scales(point_input);
            SolveContext    context{point_input};
            InitialResidual initial = prepare_initial_residual(context);
            point_input.residual_scale = context.input.residual_scale;

            const auto started = std::chrono::steady_clock::now();
            SolveResult final  = run_solver_once(context);
            const double solve_ms = elapsed_ms_since(started);
            samples_ms.push_back(solve_ms);

            const bool point_success =
                final.accepted && solver_info_succeeded(point_input.solver, final.info);
            if (point_success)
            {
                ++success_count;
                previous_previous_x  = previous_x;
                previous_previous_ip = previous_ip;
                have_previous_previous = have_previous;
                previous_x           = final.x;
                previous_ip          = point_input.Ip;
                have_previous        = true;
            }
            nfev_total     += final.nfev;
            callback_total += final.callbacks;

            nlohmann::json point_json = {
                {"index", index},
                {"scaled_Ip", point_input.Ip},
                {"initial_policy", initial_policy},
                {"fallback_reason", fallback_reason ? nlohmann::json(fallback_reason) : nlohmann::json(nullptr)},
                {"solve_ms", solve_ms},
                {"initial", initial_residual_json(point_input, initial)},
                {"normalization", normalization_json(point_input)},
                {"final", solve_result_json(final)},
                {"success", point_success},
            };
            points_json.push_back(std::move(point_json));
        }

        return {
            {"policy", scan_policy_name(policy)},
            {"parameter", "scaled_Ip"},
            {"base_scaled_Ip", base_input.Ip},
            {"relative_step", relative_step},
            {"point_count", point_count},
            {"initial_policy_counts",
             {
                 {"cold", cold_count},
                 {"warm", warm_count},
                 {"secant", secant_count},
             }},
            {"predictor_fallback_count", predictor_fallback_count},
            {"success_count", success_count},
            {"nfev_total", nfev_total},
            {"callback_total", callback_total},
            {"timing", timing_json(samples_ms)},
            {"points", std::move(points_json)},
            {"success", success_count == point_count},
        };
    }

    nlohmann::json run_parameter_scan_report(const CaseInput& base_input, const ScanConfig& scan)
    {
        nlohmann::json scan_payload;
        bool           success = true;
        if (scan.policy == ScanPolicy::All)
        {
            nlohmann::json scans = nlohmann::json::object();
            for (ScanPolicy policy : {ScanPolicy::Cold, ScanPolicy::Warm, ScanPolicy::Secant})
            {
                auto policy_report = run_parameter_scan_policy(
                    base_input,
                    policy,
                    scan.points,
                    scan.relative_step
                );
                success = success && policy_report["success"].get<bool>();
                scans[scan_policy_name(policy)] = std::move(policy_report);
            }
            scan_payload = {
                {"policy", "all"},
                {"scans", std::move(scans)},
            };
        }
        else
        {
            auto policy_report = run_parameter_scan_policy(
                base_input,
                scan.policy,
                scan.points,
                scan.relative_step
            );
            success = policy_report["success"].get<bool>();
            scan_payload = std::move(policy_report);
        }

        return {
            {"case_name", base_input.case_name},
            {"route", "PF/psin/uniform/Ip"},
            {"source_topology", source_topology_json()},
            {"x_size", BenchShape::x_size},
            {"grid", grid_json()},
            {"solver", solver_config_json(base_input)},
            {"source",
             {
                 {"scaled_heat", json_array(base_input.heat)},
                 {"scaled_current", json_array(base_input.current)},
             }},
            {"scan", std::move(scan_payload)},
            {"success", success},
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
        ScanConfig scan{};
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
            else if (arg == "--scan-points")
            {
                if (++i >= argc)
                    throw std::runtime_error("--scan-points requires a value");
                scan.points = parse_nonnegative_int("--scan-points", argv[i]);
            }
            else if (arg == "--scan-relative-step" || arg == "--scan-step")
            {
                if (++i >= argc)
                    throw std::runtime_error(arg + " requires a value");
                scan.relative_step = parse_finite_double(arg.c_str(), argv[i]);
                if (scan.relative_step < -1.0 || scan.relative_step > 1.0)
                    throw std::runtime_error(arg + " must be in [-1, 1]");
            }
            else if (arg == "--scan-policy")
            {
                if (++i >= argc)
                    throw std::runtime_error("--scan-policy requires a value");
                scan.policy = parse_scan_policy(argv[i]);
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
                          << " [--scan-points N] [--scan-policy cold|warm|secant|all]"
                          << " [--scan-relative-step STEP]"
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
        if (scan.points > 0)
        {
            const auto report = run_parameter_scan_report(input, scan);
            std::cout << report.dump(2) << '\n';
            return report["success"].get<bool>() ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        SolveContext context{input};

        InitialResidual initial = prepare_initial_residual(context);
        input.residual_scale    = context.input.residual_scale;

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
             source_topology_json()},
            {"x_size", BenchShape::x_size},
            {"grid", grid_json()},
            {"solver", solver_config_json(input)},
            {"source",
             {
                 {"scaled_heat", json_array(input.heat)},
                 {"scaled_current", json_array(input.current)},
             }},
            {"normalization", normalization_json(input)},
            {"constraints",
             {
                 {"scaled_Ip", input.Ip},
             }},
            {"jacobian_check", jacobian_check_report},
            {"timing", timing_json(samples_ms)},
            {"initial", initial_residual_json(input, initial)},
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

// ---- PF/psin/uniform/Ip validation CLI ----
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>

#include <cminpack.h>
#include <nlohmann/json.hpp>

#include "grid.h"
#include "math.h"
#include "source/pf_psin_uniform_ip.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_pf_psin_uniform_validation_cli
{

namespace
{
    using grid::Grid;
    using grid::Legendre;
    using grid::Spectral;
    using source::PfPsinUniformIpOperator;
    using std::size_t;
    using tensor::Vector;

    constexpr auto no_c_slots = std::array<profiles::ProfileSlot, 0>{};
    constexpr auto no_s_slots = std::array<profiles::ProfileSlot, 0>{};

    using SmokeShape = profiles::ProfileShape<
        1,
        2,
        1,
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using SmokeGrid     = Grid<8, 8, 1, 1, 2, Legendre, Spectral>;
    using SmokeSource   = source::UniformSourceShape<5>;
    using SmokeOperator = PfPsinUniformIpOperator<SmokeShape, SmokeGrid, SmokeSource>;
    using PackedVector  = SmokeOperator::PackedVector;

    static_assert(SmokeShape::x_size == 1);

    constexpr double veqpy_max_residual               = 1.0e-6;
    constexpr int    veqpy_requested_max_evaluations  = 1000;
    constexpr int    veqpy_maxfev                     = veqpy_requested_max_evaluations > 500
                                                            ? veqpy_requested_max_evaluations
                                                            : 500;
    constexpr double veqpy_hybr_eps                   = 1.0e-6;
    constexpr double veqpy_hybr_factor                = 1.0;
    constexpr int    veqpy_hybr_mode                  = 1;
    constexpr int    veqpy_hybr_nprint                = 0;
    constexpr double veqpy_accepted_residual_factor   = 10.0;
    constexpr double veqpy_accepted_residual_floor    = 1.0e-5;
    constexpr double veqpy_x_scale_floor              = 1.0e-2;
    constexpr double veqpy_core_profile_prior         = 1.5e-1;
    constexpr double veqpy_psin_profile_scale_default = 1.0;

    double norm2(std::span<const double, SmokeShape::x_size> values) noexcept
    {
        double total = 0.0;
        for (double value : values)
            total += value * value;
        return std::sqrt(total);
    }

    template <typename Values>
    nlohmann::json json_array(const Values& values)
    {
        nlohmann::json out = nlohmann::json::array();
        for (double value : values)
            out.push_back(value);
        return out;
    }

    template <typename MatrixType>
    nlohmann::json json_matrix_row(const MatrixType& values, size_t row)
    {
        nlohmann::json out = nlohmann::json::array();
        for (size_t col = 0; col < MatrixType::shape[1]; ++col)
            out.push_back(values(row, col));
        return out;
    }

    template <typename MatrixType>
    nlohmann::json json_matrix_col(const MatrixType& values, size_t col)
    {
        nlohmann::json out = nlohmann::json::array();
        for (size_t row = 0; row < MatrixType::shape[0]; ++row)
            out.push_back(values(row, col));
        return out;
    }

    template <typename ResidualType>
    nlohmann::json json_residual_surface_row(const ResidualType& residual, size_t row)
    {
        nlohmann::json out = nlohmann::json::array();
        for (size_t i = 0; i < ResidualType::radial_nodes; ++i)
        {
            nlohmann::json radial = nlohmann::json::array();
            for (size_t j = 0; j < ResidualType::theta_rows; ++j)
                radial.push_back(residual.surface_field(row, i, j));
            out.push_back(radial);
        }
        return out;
    }

    nlohmann::json snapshot_state(const SmokeOperator& op, const PackedVector& raw)
    {
        return {
            {"raw_residual", json_array(raw)},
            {"alpha", {op.workspace.source_runtime.alpha1, op.workspace.source_runtime.alpha2}},
            {"profiles",
             {
                 {"psin", json_matrix_col(op.workspace.profiles.profile_matrix<SmokeShape::psin_profile_id>(), 0)},
                 {"psin_r", json_matrix_col(op.workspace.profiles.profile_matrix<SmokeShape::psin_profile_id>(), 1)},
                 {"psin_rr", json_matrix_col(op.workspace.profiles.profile_matrix<SmokeShape::psin_profile_id>(), 2)},
                 {"k", json_matrix_col(op.workspace.profiles.profile_matrix<SmokeShape::kappa_profile_id>(), 0)},
                 {"c0", json_matrix_col(op.workspace.profiles.profile_matrix<SmokeShape::c_profile_id<0>()>(), 0)},
             }},
            {"source",
             {
                 {"source_psin_query", json_array(op.workspace.source_runtime.source_psin_query)},
                 {"source_parameter_query", json_array(op.workspace.source_runtime.source_parameter_query)},
                 {"materialized_heat_input", json_array(op.workspace.source_runtime.materialized_heat_input)},
                 {"materialized_current_input", json_array(op.workspace.source_runtime.materialized_current_input)},
                 {"profile_root_psin",
                  json_matrix_row(op.workspace.source_runtime.profile_root_fields, source::root_psin)},
                 {"profile_root_psin_r",
                  json_matrix_row(op.workspace.source_runtime.profile_root_fields, source::root_psin_r)},
                 {"profile_root_psin_rr",
                  json_matrix_row(op.workspace.source_runtime.profile_root_fields, source::root_psin_rr)},
                 {"source_target_psin",
                  json_matrix_row(op.workspace.source_runtime.source_target_root_fields, source::root_psin)},
                 {"source_target_psin_r",
                  json_matrix_row(op.workspace.source_runtime.source_target_root_fields, source::root_psin_r)},
                 {"source_target_psin_rr",
                  json_matrix_row(op.workspace.source_runtime.source_target_root_fields, source::root_psin_rr)},
                 {"FFn_psin", json_array(op.workspace.source_runtime.FFn_psin)},
                 {"Pn_psin", json_array(op.workspace.source_runtime.Pn_psin)},
             }},
            {"geometry",
             {
                 {"S_r", json_matrix_row(op.workspace.geometry.radial_fields, geometry::radial_S_r)},
                 {"V_r", json_matrix_row(op.workspace.geometry.radial_fields, geometry::radial_V_r)},
                 {"Kn", json_matrix_row(op.workspace.geometry.radial_fields, geometry::radial_Kn)},
                 {"Kn_r", json_matrix_row(op.workspace.geometry.radial_fields, geometry::radial_Kn_r)},
                 {"Ln_r", json_matrix_row(op.workspace.geometry.radial_fields, geometry::radial_Ln_r)},
             }},
            {"residual_surface",
             {
                 {"G", json_residual_surface_row(op.workspace.residual, residual::surface_G)},
                 {"Gpsin_R", json_residual_surface_row(op.workspace.residual, residual::surface_Gpsin_R)},
                 {"Gpsin_Z", json_residual_surface_row(op.workspace.residual, residual::surface_Gpsin_Z)},
                 {"Gpsin_R_sin_tb",
                  json_residual_surface_row(op.workspace.residual, residual::surface_Gpsin_R_sin_tb)},
             }},
        };
    }

    constexpr double veqpy_acceptance_threshold() noexcept
    {
        constexpr double scaled = veqpy_max_residual * veqpy_accepted_residual_factor;
        return scaled > veqpy_accepted_residual_floor ? scaled : veqpy_accepted_residual_floor;
    }

    std::array<double, SmokeShape::x_size> decode_z_to_x(
        const std::array<double, SmokeShape::x_size>& z,
        const std::array<double, SmokeShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, SmokeShape::x_size> x{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            x[i] = z[i] * x_scale[i];
        return x;
    }

    std::array<double, SmokeShape::x_size> encode_x_to_z(
        const std::array<double, SmokeShape::x_size>& x,
        const std::array<double, SmokeShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, SmokeShape::x_size> z{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            z[i] = x[i] / x_scale[i];
        return z;
    }

    struct SolveContext
    {
        SmokeOperator op{};
        std::array<double, SmokeShape::x_size> x_scale{};
        std::array<double, SmokeShape::x_size> residual_scale{};
        int                                    evaluations = 0;

        SolveContext()
        {
            SmokeOperator::RuntimeParams params{};
            params.a = 0.42;
            params.R0 = 1.8;
            params.Z0 = -0.25;
            params.B0 = 2.1;
            params.Ip = 3.7699111843077517;
            params.fix_rho = 0.0;
            params.profile_params.offsets[SmokeShape::kappa_profile_id] = 1.45;
            params.profile_params.offsets[SmokeShape::c_profile_id<0>()] = 0.0;
            op.set_runtime_params(params);

            constexpr std::array<double, SmokeSource::sample_count> heat{
                2.0,
                2.75,
                3.5,
                4.25,
                5.0,
            };
            constexpr std::array<double, SmokeSource::sample_count> current{
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
            };
            op.set_uniform_sources(
                std::span<const double, SmokeSource::sample_count>{heat.data(), heat.size()},
                std::span<const double, SmokeSource::sample_count>{current.data(), current.size()}
            );
        }

        void raw_residual(
            std::span<const double, SmokeShape::x_size> x,
            std::span<double, SmokeShape::x_size>       residual
        ) noexcept
        {
            PackedVector raw{};
            op.evaluate(x, raw);
            for (size_t i = 0; i < SmokeShape::x_size; ++i)
                residual[i] = raw[i];
        }

        void configure_veqpy_scales(const std::array<double, SmokeShape::x_size>& x0) noexcept
        {
            const double guess_rms = norm2(std::span<const double, SmokeShape::x_size>{
                x0.data(),
                SmokeShape::x_size,
            });
            double psin_scale = veqpy_psin_profile_scale_default;
            if (std::abs(psin_scale - 1.0) <= 1.0e-12)
                psin_scale = veqpy_core_profile_prior;
            x_scale[0] = std::max(
                {psin_scale, veqpy_core_profile_prior, guess_rms, veqpy_x_scale_floor}
            );

            PackedVector initial_raw{};
            op.evaluate(
                std::span<const double, SmokeShape::x_size>{x0.data(), SmokeShape::x_size},
                initial_raw
            );
            const double initial_norm = norm2(std::span<const double, SmokeShape::x_size>{
                initial_raw.data(),
                SmokeShape::x_size,
            });
            const double initial_rms =
                initial_norm / std::sqrt(static_cast<double>(SmokeShape::x_size));
            const double block_scale = initial_rms > 1.0 ? initial_rms : 1.0;
            for (double& scale : residual_scale)
                scale = block_scale;
        }
    };

    int pf_residual_z(void* data, int n, const double* z, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != static_cast<int>(SmokeShape::x_size))
            return 0;

        auto& context = *static_cast<SolveContext*>(data);
        ++context.evaluations;

        std::array<double, SmokeShape::x_size> z_eval{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            z_eval[i] = z[i];
        const auto x = decode_z_to_x(z_eval, context.x_scale);

        PackedVector raw{};
        context.raw_residual(
            std::span<const double, SmokeShape::x_size>{x.data(), SmokeShape::x_size},
            std::span<double, SmokeShape::x_size>{raw.data(), SmokeShape::x_size}
        );

        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            fvec[i] = raw[i] / context.residual_scale[i];
        return 0;
    }
} // namespace

int run(int, char**)
{
    SolveContext context;

    std::array<double, SmokeShape::x_size> x_initial{};
    context.configure_veqpy_scales(x_initial);

    PackedVector initial{};
    context.raw_residual(
        std::span<const double, SmokeShape::x_size>{x_initial.data(), SmokeShape::x_size},
        std::span<double, SmokeShape::x_size>{initial.data(), SmokeShape::x_size}
    );
    const double initial_norm = norm2(std::span<const double, SmokeShape::x_size>{
        initial.data(),
        SmokeShape::x_size,
    });
    PackedVector initial_scaled{};
    for (size_t i = 0; i < SmokeShape::x_size; ++i)
        initial_scaled[i] = initial[i] / context.residual_scale[i];
    const double initial_scaled_norm = norm2(std::span<const double, SmokeShape::x_size>{
        initial_scaled.data(),
        SmokeShape::x_size,
    });
    const auto initial_state = snapshot_state(context.op, initial);

    auto z = encode_x_to_z(x_initial, context.x_scale);
    PackedVector fvec{};

    constexpr int n = static_cast<int>(SmokeShape::x_size);
    constexpr int ml = n - 1;
    constexpr int mu = n - 1;
    constexpr int lr = n * (n + 1) / 2;
    std::array<double, SmokeShape::x_size> diag{};
    diag.fill(1.0);
    std::array<double, SmokeShape::x_size * SmokeShape::x_size> fjac{};
    std::array<double, static_cast<size_t>(lr)>                 r{};
    std::array<double, SmokeShape::x_size>                      qtf{};
    std::array<double, SmokeShape::x_size>                      wa1{};
    std::array<double, SmokeShape::x_size>                      wa2{};
    std::array<double, SmokeShape::x_size>                      wa3{};
    std::array<double, SmokeShape::x_size>                      wa4{};
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

    const auto x_final = decode_z_to_x(z, context.x_scale);
    PackedVector final{};
    context.raw_residual(
        std::span<const double, SmokeShape::x_size>{x_final.data(), SmokeShape::x_size},
        std::span<double, SmokeShape::x_size>{final.data(), SmokeShape::x_size}
    );
    const double final_norm = norm2(std::span<const double, SmokeShape::x_size>{
        final.data(),
        SmokeShape::x_size,
    });
    PackedVector final_scaled{};
    for (size_t i = 0; i < SmokeShape::x_size; ++i)
        final_scaled[i] = final[i] / context.residual_scale[i];
    const double final_scaled_norm = norm2(std::span<const double, SmokeShape::x_size>{
        final_scaled.data(),
        SmokeShape::x_size,
    });
    const bool accepted_by_veqpy = final_norm <= veqpy_acceptance_threshold();
    const auto final_state = snapshot_state(context.op, final);

    nlohmann::json report = {
        {"route", "PF/psin/uniform/Ip"},
        {"x_size", SmokeShape::x_size},
        {"solver",
         {
             {"method", "hybr"},
             {"entrypoint", "cminpack::hybrd"},
             {"initial_policy", "auto/zero"},
             {"residual_normalization", "fast"},
             {"max_residual", veqpy_max_residual},
             {"acceptance_threshold", veqpy_acceptance_threshold()},
             {"requested_max_evaluations", veqpy_requested_max_evaluations},
             {"maxfev", veqpy_maxfev},
             {"eps", veqpy_hybr_eps},
             {"factor", veqpy_hybr_factor},
             {"diag_mode", veqpy_hybr_mode},
             {"ml", ml},
             {"mu", mu},
         }},
        {"normalization",
         {
             {"x_scale", json_array(context.x_scale)},
             {"residual_scale", json_array(context.residual_scale)},
             {"unknown_space", "z = x / x_scale"},
         }},
        {"initial",
         {
             {"x", json_array(x_initial)},
             {"z", json_array(encode_x_to_z(x_initial, context.x_scale))},
             {"raw_residual", json_array(initial)},
             {"scaled_residual", json_array(initial_scaled)},
             {"raw_norm", initial_norm},
             {"scaled_norm", initial_scaled_norm},
             {"state", initial_state},
         }},
        {"final",
         {
             {"x", json_array(x_final)},
             {"z", json_array(z)},
             {"raw_residual", json_array(final)},
             {"scaled_residual", json_array(final_scaled)},
             {"raw_norm", final_norm},
             {"scaled_norm", final_scaled_norm},
             {"accepted_by_veqpy", accepted_by_veqpy},
             {"state", final_state},
         }},
        {"cminpack",
         {
             {"info", info},
             {"success", info == 1},
             {"nfev", nfev},
             {"callback_evaluations", context.evaluations},
         }},
    };

    std::cout << report.dump(2) << '\n';
    return (accepted_by_veqpy && info > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace veqlib_pf_psin_uniform_validation_cli

// ---- stage benchmark CLI ----
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "geometry.h"
#include "grid.h"
#include "math.h"
#include "source/pf_psin_uniform_ip.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_stage_benchmark_cli
{

namespace
{
    using config::DefaultTopology;
    using geometry::radial_Kn;
    using geometry::radial_Kn_r;
    using geometry::radial_Ln_r;
    using geometry::radial_S_r;
    using geometry::radial_V_r;
    using geometry::surface_R;
    using grid::Grid;
    using grid::Legendre;
    using grid::Spectral;
    using math::cos;
    using math::sin;
    using source::PfPsinUniformIpOperator;
    using profiles::OptimizedProfileShapeFromCountsT;
    using residual::surface_G;
    using source::axis_fix_count;
    using source::UniformSourceShape;
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    template <auto Counts>
    consteval size_t max_profile_count() noexcept
    {
        size_t value = 0;
        for (size_t count : Counts)
            value = value < count ? count : value;
        return value;
    }

    template <auto CFamilyCounts, auto SFamilyCounts>
    consteval size_t inferred_M_max() noexcept
    {
        constexpr size_t c_max = CFamilyCounts.size() == 0 ? 0 : CFamilyCounts.size() - 1;
        constexpr size_t s_max = SFamilyCounts.size();
        const size_t     value = c_max < s_max ? s_max : c_max;
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
        size_t count             = HCount;
        count                    = count < VCount ? VCount : count;
        count                    = count < KappaCount ? KappaCount : count;
        count                    = count < PsinCount ? PsinCount : count;
        count                    = count < FCount ? FCount : count;
        constexpr size_t c_count = max_profile_count<CFamilyCounts>();
        constexpr size_t s_count = max_profile_count<SFamilyCounts>();
        count                    = count < c_count ? c_count : count;
        count                    = count < s_count ? s_count : count;
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
        static constexpr size_t L_max =
            inferred_L_max<HCount, VCount, KappaCount, PsinCount, FCount, CFamilyCounts, SFamilyCounts>();
        static constexpr size_t M_max = inferred_M_max<CFamilyCounts, SFamilyCounts>();
        static constexpr size_t K_max = inferred_K_max<M_max>();

        using Shape    = OptimizedProfileShapeFromCountsT<L_max,
                                                          K_max,
                                                          HCount,
                                                          VCount,
                                                          KappaCount,
                                                          PsinCount,
                                                          FCount,
                                                          CFamilyCounts,
                                                          SFamilyCounts>;
        using Grid     = Grid<Nr, Nt, Shape::L_max, Shape::M_max, Shape::K_max, QuadratureScheme, CalculusScheme>;
        using Source   = UniformSourceShape<SourceSamples>;
        using Operator = PfPsinUniformIpOperator<Shape, Grid, Source>;
    };

    constexpr auto bench_c_counts = DefaultTopology::c_family_counts;
    constexpr auto bench_s_counts = DefaultTopology::s_family_counts;

    using BenchTopology =
        PfPsinUniformIpTopology<DefaultTopology::Nr,
                                DefaultTopology::Nt,
                                51,
                                DefaultTopology::h_count,
                                DefaultTopology::v_count,
                                DefaultTopology::kappa_count,
                                DefaultTopology::psin_count,
                                DefaultTopology::F_count,
                                bench_c_counts,
                                bench_s_counts,
                                Legendre,
                                Spectral>;
    using BenchShape    = BenchTopology::Shape;
    using BenchGrid     = BenchTopology::Grid;
    using BenchSource   = BenchTopology::Source;
    using BenchOperator = BenchTopology::Operator;
    using PackedVector        = BenchOperator::PackedVector;
    using SourceRadialVector  = Vector<double, BenchGrid::radial_nodes>;
    using ResidualMomentRows  = BenchOperator::Residual::MomentRows;

    static_assert(BenchShape::L_max == DefaultTopology::L_max);
    static_assert(BenchShape::M_max == DefaultTopology::M_max);
    static_assert(BenchShape::K_max == DefaultTopology::K_max);

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_heat = {
        -0.789683058574694,    -0.7925936329632908, -0.7953979059157582, -0.7981175242684836, -0.8007734699426484,
        -0.8033829643453037,   -0.8059602413311435, -0.8085160883171674, -0.811058402798229,  -0.8135924782601793,
        -0.8161210171627857,   -0.8186443668543919, -0.8211602449193173, -0.82366353909573,   -0.8261466407410907,
        -0.8286005031116496,   -0.8310129351634512, -0.8333638631960294, -0.8356348482673257, -0.8378088362996647,
        -0.8398452317867598,   -0.8417128477427658, -0.8433930625296288, -0.8448046915616387, -0.8459047888622718,
        -0.8467054195394251,   -0.8468550092753417, -0.8466915344330476, -0.8459189587408882, -0.844353928057723,
        -0.8418811874023397,   -0.8384860115442367, -0.8339038151163928, -0.8279318154386046, -0.8204511956034657,
        -0.8111301194374045,   -0.7996783681429246, -0.7858549460849861, -0.7692183653145488, -0.7492833542853738,
        -0.7256038086449172,   -0.6975411195823384, -0.6643030775956854, -0.6249342427240242, -0.5782460674488447,
        -0.5227081311582821,   -0.4562798131359923, -0.3761603550417907, -0.2784903156513912, -0.15751845630174122,
        -0.004428769494182179,
    };

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_current = {
        -0.2884247371510828,    -0.28903009704030685, -0.28957011555052764, -0.29005075790401863, -0.2904781886031161,
        -0.2908575435658439,    -0.2911919683745232,  -0.2914838716684553,  -0.2917344967330943,  -0.2919437907627883,
        -0.29211086667346203,   -0.2922346427093772,  -0.2923145136340828,  -0.29232232828599186, -0.2923070658458569,
        -0.2922075036101631,    -0.29204221633819877, -0.2918026604476458,  -0.2914746207679046,  -0.29104564954616613,
        -0.2905098257293566,    -0.28984851066007944, -0.2890431993822433,  -0.2880834160319397,  -0.2869422749878048,
        -0.28559761849429993,   -0.28403023480001177, -0.2822015304104101,  -0.2800887239343856,  -0.27765576104313033,
        -0.274853815959196,     -0.27165492157456717, -0.26800248440388824, -0.2638385916755866,  -0.25912135836327627,
        -0.25377334278513203,   -0.2477205935589761,  -0.2409002037441097,  -0.23321180576616163, -0.22454617787318162,
        -0.2148028033443947,    -0.20384443014756062, -0.191504616045878,   -0.17758595047988296, -0.1618458187382766,
        -0.143974507066849,     -0.12356213030456985, -0.10004932271682981, -0.07266570051589127, -0.040268147498330784,
        -0.0011074929612556953,
    };

    enum class StageKind
    {
        ProfilesFixed,
        ProfilesActive,
        ProfilesAll,
        GeometryPhase,
        GeometryPhaseSincos,
        GeometryPhaseSplitSincos,
        GeometryMetricNoStore,
        Geometry,
        SourceMaterialize,
        SourceCopyRegularize,
        SourceDpsin,
        SourceApsin,
        SourceDApsin,
        SourceDApsinBlock4,
        SourceInterpolatePair,
        SourceIntegrand,
        SourceAIntegrandRowdot,
        SourceAIntegrand,
        SourceNormalize,
        SourceDNormalized,
        SourceAlpha,
        SourceUpdate,
        ResidualUpdate,
        ResidualThetaReduce,
        ResidualRadialProject,
        ResidualPack,
        Evaluate,
        EvaluateRing,
    };

    struct Options
    {
        std::string stage  = "all";
        size_t      repeat = 30;
        size_t      warmup = 5;
        size_t      inner  = 1000;
        size_t      ring_size = 16;
    };

    struct Stats
    {
        double avg_ns    = 0.0;
        double min_ns    = 0.0;
        double max_ns    = 0.0;
        double median_ns = 0.0;
        double p95_ns    = 0.0;
        double std_ns    = 0.0;
    };

    volatile double benchmark_sink = 0.0;

    void compiler_barrier(const void* pointer) noexcept { asm volatile("" : : "r"(pointer) : "memory"); }

    std::span<const double, BenchShape::x_size> x_span(const std::array<double, BenchShape::x_size>& x) noexcept
    {
        return std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size};
    }

    void configure_operator(BenchOperator& op) noexcept
    {
        BenchOperator::RuntimeParams params{};
        params.a                                                     = 1.05 / 1.85;
        params.R0                                                    = 1.05;
        params.Z0                                                    = 0.0;
        params.B0                                                    = 3.0;
        params.Ip                                                    = 3.7699111867885415;
        params.fix_rho                                               = 0.05;
        params.profile_params.offsets[BenchShape::kappa_profile_id]  = 2.2;
        params.profile_params.offsets[BenchShape::c_profile_id<0>()] = 0.0;
        params.profile_params.offsets[BenchShape::s_profile_id<1>()] = 0.52359877559829887308;
        op.set_runtime_params(params);
        op.set_uniform_sources(
            std::span<const double, BenchSource::sample_count>{
                benchmark_scaled_heat.data(),
                BenchSource::sample_count,
            },
            std::span<const double, BenchSource::sample_count>{
                benchmark_scaled_current.data(),
                BenchSource::sample_count,
            });
    }

    void refresh_profiles(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        const auto& params = op.runtime_params();
        op.refresh_static_plan();
        op.workspace.profiles.refresh_active(x, params.profile_params);
    }

    void prepare_geometry(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        const auto& params = op.runtime_params();
        refresh_profiles(op, x);
        op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
    }

    void prepare_source_materialized(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
    }

    void prepare_source_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        const auto& params = op.runtime_params();
        prepare_geometry(op, x);
        op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
        op.workspace.source_runtime.update_pf_psin_uniform_ip(
            op.workspace.geometry, params.Ip, op.plan.n_axis_fix);
    }

    void prepare_residual_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        prepare_source_updated(op, x);
        op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
    }

    void prepare_source_profile_root(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        op.workspace.source_runtime.benchmark_copy_profile_psin_r(op.workspace.profiles);
        op.workspace.source_runtime.benchmark_regularize_psin_r(op.plan.n_axis_fix);
    }

    void prepare_source_integrand(BenchOperator&                              op,
                                  std::span<const double, BenchShape::x_size> x,
                                  SourceRadialVector&                         integrand) noexcept
    {
        const auto& params = op.runtime_params();
        prepare_source_materialized(op, x);
        op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
        op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(integrand, op.workspace.geometry);
    }

    void prepare_source_accumulated_integrand(BenchOperator&                              op,
                                              std::span<const double, BenchShape::x_size> x,
                                              SourceRadialVector&                         accumulated) noexcept
    {
        SourceRadialVector integrand{uninitialized};
        prepare_source_integrand(op, x, integrand);
        op.workspace.source_runtime.benchmark_A_integrand_into(accumulated, integrand);
    }

    void prepare_source_normalized_psin(BenchOperator&                              op,
                                        std::span<const double, BenchShape::x_size> x,
                                        SourceRadialVector&                         normalized) noexcept
    {
        SourceRadialVector accumulated{uninitialized};
        prepare_source_accumulated_integrand(op, x, accumulated);
        (void)op.workspace.source_runtime.benchmark_normalize_psin_r_into(
            normalized, accumulated, op.workspace.geometry, op.plan.n_axis_fix);
    }

    enum class GeometryProbeMode
    {
        Phase,
        PhaseSincos,
        PhaseSplitSincos,
        MetricNoStore,
    };

    template <GeometryProbeMode Mode>
    void run_geometry_probe(BenchOperator& op) noexcept
    {
        const auto& params = op.runtime_params();
        using Shape = BenchShape;

        constexpr size_t profile_value   = 0;
        constexpr size_t profile_radial  = 1;
        constexpr size_t profile_radial2 = 2;
        constexpr size_t c_limit         = BenchGrid::harmonic_rows;
        constexpr size_t s_limit         = BenchGrid::harmonic_rows;
        constexpr double probe_pi        = 3.141592653589793238462643383279502884;

        const auto& runtime_profiles = op.workspace.profiles;
        double      sink             = 0.0;

        for (size_t i = 0; i < BenchGrid::radial_nodes; ++i)
        {
            const double rho_i   = BenchGrid::nodes[i];
            const double c0_i    = runtime_profiles.c_family_fields(0, i, profile_value);
            const double c0_r_i  = runtime_profiles.c_family_fields(0, i, profile_radial);
            const double c0_rr_i = runtime_profiles.c_family_fields(0, i, profile_radial2);

            Matrix<double, c_limit, 3> c_fields{uninitialized};
            Matrix<double, s_limit, 3> s_fields{uninitialized};
            for (size_t order = 1; order < c_limit; ++order)
            {
                c_fields(order, profile_value) = runtime_profiles.c_family_fields(order, i, profile_value);
                c_fields(order, profile_radial) = runtime_profiles.c_family_fields(order, i, profile_radial);
                c_fields(order, profile_radial2) = runtime_profiles.c_family_fields(order, i, profile_radial2);
            }
            for (size_t order = 1; order < s_limit; ++order)
            {
                s_fields(order, profile_value) = runtime_profiles.s_family_fields(order, i, profile_value);
                s_fields(order, profile_radial) = runtime_profiles.s_family_fields(order, i, profile_radial);
                s_fields(order, profile_radial2) = runtime_profiles.s_family_fields(order, i, profile_radial2);
            }

            if constexpr (Mode == GeometryProbeMode::MetricNoStore)
            {
                const double h_i    = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_value);
                const double h_r_i  = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial);
                const double h_rr_i = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial2);
                const double v_r_i  = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial);
                const double v_rr_i = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial2);
                const double k_i    = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_value);
                const double k_r_i  = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial);
                const double k_rr_i = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial2);

                double sum_J          = 0.0;
                double sum_JR         = 0.0;
                double sum_gttdivJR   = 0.0;
                double sum_gttdivJR_r = 0.0;
                double sum_JdivR      = 0.0;
                double sum_grtdivJR_t = 0.0;

                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_r_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_t_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rr_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_tt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> sin_tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> cos_tb_values;

                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    double tb_ij    = BenchGrid::theta[j] + c0_i;
                    double tb_r_ij  = c0_r_i;
                    double tb_t_ij  = 1.0;
                    double tb_rr_ij = c0_rr_i;
                    double tb_rt_ij = 0.0;
                    double tb_tt_ij = 0.0;

                    for (size_t order = 1; order < c_limit; ++order)
                    {
                        const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                        const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                        const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                        const double c_i       = c_fields(order, profile_value);
                        const double c_r_i     = c_fields(order, profile_radial);
                        const double c_rr_i    = c_fields(order, profile_radial2);

                        tb_ij += c_i * cos_kt;
                        tb_r_ij += c_r_i * cos_kt;
                        tb_t_ij -= c_i * k_sin_kt;
                        tb_rr_ij += c_rr_i * cos_kt;
                        tb_rt_ij -= c_r_i * k_sin_kt;
                        tb_tt_ij -= c_i * k2_cos_kt;
                    }

                    for (size_t order = 1; order < s_limit; ++order)
                    {
                        const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                        const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                        const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                        const double s_i       = s_fields(order, profile_value);
                        const double s_r_i     = s_fields(order, profile_radial);
                        const double s_rr_i    = s_fields(order, profile_radial2);

                        tb_ij += s_i * sin_kt;
                        tb_r_ij += s_r_i * sin_kt;
                        tb_t_ij += s_i * k_cos_kt;
                        tb_rr_ij += s_rr_i * sin_kt;
                        tb_rt_ij += s_r_i * k_cos_kt;
                        tb_tt_ij -= s_i * k2_sin_kt;
                    }

                    tb_values[j] = tb_ij;
                    tb_r_values[j] = tb_r_ij;
                    tb_t_values[j] = tb_t_ij;
                    tb_rr_values[j] = tb_rr_ij;
                    tb_rt_values[j] = tb_rt_ij;
                    tb_tt_values[j] = tb_tt_ij;
                }

#pragma clang loop vectorize(enable)
                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    const double tb_ij = tb_values[j];
                    geometry::detail::reduced_taylor_sincos(tb_ij, sin_tb_values[j], cos_tb_values[j]);
                }

                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    const double sin_t = BenchGrid::sin_mtheta(1, j);
                    const double cos_t = BenchGrid::cos_mtheta(1, j);

                    const double tb_r_ij   = tb_r_values[j];
                    const double tb_t_ij   = tb_t_values[j];
                    const double tb_rr_ij  = tb_rr_values[j];
                    const double tb_rt_ij  = tb_rt_values[j];
                    const double tb_tt_ij  = tb_tt_values[j];
                    const double cos_tb_ij = cos_tb_values[j];
                    const double sin_tb_ij = sin_tb_values[j];

                    double R_ij = params.R0 + params.a * (h_i + rho_i * cos_tb_ij);
                    if (R_ij < 1.0e-6)
                        R_ij = 1.0e-6;

                    const double R_r_ij = params.a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij);
                    const double R_t_ij = -params.a * rho_i * sin_tb_ij * tb_t_ij;
                    const double R_rr_ij = params.a *
                        (h_rr_i - 2.0 * sin_tb_ij * tb_r_ij -
                         rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij));
                    const double R_rt_ij = -params.a *
                        (sin_tb_ij * tb_t_ij +
                         rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij));
                    const double R_tt_ij =
                        -params.a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij);

                    const double Z_r_ij  = params.a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t);
                    const double Z_t_ij  = -params.a * rho_i * k_i * cos_t;
                    const double Z_rr_ij = params.a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t);
                    const double Z_rt_ij = -params.a * (k_i + rho_i * k_r_i) * cos_t;
                    const double Z_tt_ij = params.a * rho_i * k_i * sin_t;

                    double J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij;
                    if (J_ij < 1.0e-6)
                        J_ij = 1.0e-6;

                    const double J_r_ij =
                        -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij);
                    const double J_t_ij =
                        -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij);
                    const double JR_ij       = J_ij * R_ij;
                    const double JR_r_ij     = J_r_ij * R_ij + J_ij * R_r_ij;
                    const double JR_t_ij     = J_t_ij * R_ij + J_ij * R_t_ij;
                    const double JdivR_ij    = J_ij / R_ij;
                    const double grt_ij      = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij;
                    const double grt_t_ij    =
                        R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij;
                    const double gtt_ij         = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij;
                    const double gtt_r_ij       = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij);
                    const double inv_JR         = 1.0 / JR_ij;
                    const double grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR;
                    const double gttdivJR_ij   = gtt_ij * inv_JR;
                    const double gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR;
                    sum_J += J_ij;
                    sum_JR += JR_ij;
                    sum_gttdivJR += gttdivJR_ij;
                    sum_gttdivJR_r += gttdivJR_r_ij;
                    sum_JdivR += JdivR_ij;
                    sum_grtdivJR_t += grtdivJR_t_ij;
                }

                constexpr double theta_scale = 2.0 * probe_pi / static_cast<double>(BenchGrid::theta_rows);
                constexpr double mean_scale  = 1.0 / static_cast<double>(BenchGrid::theta_rows);
                op.workspace.geometry.radial_fields(radial_S_r, i) = sum_J * theta_scale;
                op.workspace.geometry.radial_fields(radial_V_r, i) = sum_JR * theta_scale * 2.0 * probe_pi;
                op.workspace.geometry.radial_fields(radial_Kn, i) = sum_gttdivJR * mean_scale;
                op.workspace.geometry.radial_fields(radial_Kn_r, i) = sum_gttdivJR_r * mean_scale;
                op.workspace.geometry.radial_fields(radial_Ln_r, i) = (sum_JdivR + sum_grtdivJR_t) * mean_scale;
            }
            else
            {
                if constexpr (Mode == GeometryProbeMode::PhaseSplitSincos)
                {
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_r_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_t_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rr_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rt_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_tt_values;

                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double tb_ij    = BenchGrid::theta[j] + c0_i;
                        double tb_r_ij  = c0_r_i;
                        double tb_t_ij  = 1.0;
                        double tb_rr_ij = c0_rr_i;
                        double tb_rt_ij = 0.0;
                        double tb_tt_ij = 0.0;

                        for (size_t order = 1; order < c_limit; ++order)
                        {
                            const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                            const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                            const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                            const double c_i       = c_fields(order, profile_value);
                            const double c_r_i     = c_fields(order, profile_radial);
                            const double c_rr_i    = c_fields(order, profile_radial2);

                            tb_ij += c_i * cos_kt;
                            tb_r_ij += c_r_i * cos_kt;
                            tb_t_ij -= c_i * k_sin_kt;
                            tb_rr_ij += c_rr_i * cos_kt;
                            tb_rt_ij -= c_r_i * k_sin_kt;
                            tb_tt_ij -= c_i * k2_cos_kt;
                        }

                        for (size_t order = 1; order < s_limit; ++order)
                        {
                            const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                            const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                            const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                            const double s_i       = s_fields(order, profile_value);
                            const double s_r_i     = s_fields(order, profile_radial);
                            const double s_rr_i    = s_fields(order, profile_radial2);

                            tb_ij += s_i * sin_kt;
                            tb_r_ij += s_r_i * sin_kt;
                            tb_t_ij += s_i * k_cos_kt;
                            tb_rr_ij += s_rr_i * sin_kt;
                            tb_rt_ij += s_r_i * k_cos_kt;
                            tb_tt_ij -= s_i * k2_sin_kt;
                        }

                        tb_values[j] = tb_ij;
                        tb_r_values[j] = tb_r_ij;
                        tb_t_values[j] = tb_t_ij;
                        tb_rr_values[j] = tb_rr_ij;
                        tb_rt_values[j] = tb_rt_ij;
                        tb_tt_values[j] = tb_tt_ij;
                    }

#pragma clang loop vectorize(enable)
                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double sin_tb = 0.0;
                        double cos_tb = 0.0;
                        geometry::detail::reduced_taylor_sincos(tb_values[j], sin_tb, cos_tb);
                        sink += sin_tb + cos_tb + tb_r_values[j] + tb_t_values[j] + tb_rr_values[j] + tb_rt_values[j] +
                                tb_tt_values[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double tb_ij    = BenchGrid::theta[j] + c0_i;
                        double tb_r_ij  = c0_r_i;
                        double tb_t_ij  = 1.0;
                        double tb_rr_ij = c0_rr_i;
                        double tb_rt_ij = 0.0;
                        double tb_tt_ij = 0.0;

                        for (size_t order = 1; order < c_limit; ++order)
                        {
                            const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                            const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                            const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                            const double c_i       = c_fields(order, profile_value);
                            const double c_r_i     = c_fields(order, profile_radial);
                            const double c_rr_i    = c_fields(order, profile_radial2);

                            tb_ij += c_i * cos_kt;
                            tb_r_ij += c_r_i * cos_kt;
                            tb_t_ij -= c_i * k_sin_kt;
                            tb_rr_ij += c_rr_i * cos_kt;
                            tb_rt_ij -= c_r_i * k_sin_kt;
                            tb_tt_ij -= c_i * k2_cos_kt;
                        }

                        for (size_t order = 1; order < s_limit; ++order)
                        {
                            const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                            const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                            const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                            const double s_i       = s_fields(order, profile_value);
                            const double s_r_i     = s_fields(order, profile_radial);
                            const double s_rr_i    = s_fields(order, profile_radial2);

                            tb_ij += s_i * sin_kt;
                            tb_r_ij += s_r_i * sin_kt;
                            tb_t_ij += s_i * k_cos_kt;
                            tb_rr_ij += s_rr_i * sin_kt;
                            tb_rt_ij += s_r_i * k_cos_kt;
                            tb_tt_ij -= s_i * k2_sin_kt;
                        }

                        if constexpr (Mode == GeometryProbeMode::Phase)
                        {
                            sink += tb_ij + tb_r_ij + tb_t_ij + tb_rr_ij + tb_rt_ij + tb_tt_ij;
                        }
                        else
                        {
                            sink += sin(tb_ij) + cos(tb_ij) + tb_r_ij + tb_t_ij + tb_rr_ij + tb_rt_ij + tb_tt_ij;
                        }
                    }
                }
            }
        }

        if constexpr (Mode != GeometryProbeMode::MetricNoStore)
            op.workspace.geometry.radial_fields(radial_S_r, 0) = sink;
    }

    double consume_state(const BenchOperator&       op,
                         const PackedVector&        packed,
                         const SourceRadialVector&  source_scratch,
                         const ResidualMomentRows&  residual_moments) noexcept
    {
        return op.workspace.profiles.profile_field(BenchShape::psin_profile_id, 0, 0) +
               op.workspace.geometry.surface_field(surface_R, 0, 0) + op.workspace.source_runtime.alpha1 +
               op.workspace.residual.surface_field(surface_G, 0, 0) + packed[0] + source_scratch[0] +
               residual_moments(0, 0);
    }

    void run_stage_once(StageKind                                   stage,
                        BenchOperator&                              op,
                        std::span<const double, BenchShape::x_size> x,
                        PackedVector&                               packed,
                        SourceRadialVector&                         source_scratch,
                        SourceRadialVector&                         source_aux,
        ResidualMomentRows&                         residual_moments)
    {
        const auto& params = op.runtime_params();
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            op.workspace.profiles.refresh_fixed(params.profile_params);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesActive:
            op.workspace.profiles.refresh_active(x, params.profile_params);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesAll:
            refresh_profiles(op, x);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::GeometryPhase:
            run_geometry_probe<GeometryProbeMode::Phase>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryPhaseSincos:
            run_geometry_probe<GeometryProbeMode::PhaseSincos>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryPhaseSplitSincos:
            run_geometry_probe<GeometryProbeMode::PhaseSplitSincos>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryMetricNoStore:
            run_geometry_probe<GeometryProbeMode::MetricNoStore>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::Geometry:
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            compiler_barrier(op.workspace.geometry.surface_fields.data());
            break;
        case StageKind::SourceMaterialize:
            op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.materialized_heat_input.data());
            break;
        case StageKind::SourceCopyRegularize:
            op.workspace.source_runtime.benchmark_copy_profile_psin_r(op.workspace.profiles);
            op.workspace.source_runtime.benchmark_regularize_psin_r(op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceDpsin:
            op.workspace.source_runtime.benchmark_D_psin_into_rr();
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceApsin:
            op.workspace.source_runtime.benchmark_A_psin_into(source_scratch);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceDApsin:
            op.workspace.source_runtime.benchmark_DA_psin_into(source_scratch, source_aux);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceDApsinBlock4:
            op.workspace.source_runtime.benchmark_DA_psin_block4_into(source_scratch, source_aux);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceInterpolatePair:
            op.workspace.source_runtime.benchmark_prepare_psin_queries();
            op.workspace.source_runtime.benchmark_interpolate_pair();
            compiler_barrier(op.workspace.source_runtime.materialized_heat_input.data());
            break;
        case StageKind::SourceIntegrand:
            op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(source_scratch, op.workspace.geometry);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceAIntegrand:
            op.workspace.source_runtime.benchmark_A_integrand_into(source_aux, source_scratch);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceAIntegrandRowdot:
            op.workspace.source_runtime.benchmark_A_integrand_rowdot_into(source_aux, source_scratch);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceNormalize:
            (void)op.workspace.source_runtime.benchmark_normalize_psin_r_into(
                source_aux, source_scratch, op.workspace.geometry, op.plan.n_axis_fix);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceDNormalized:
            op.workspace.source_runtime.benchmark_D_normalized_psin_into_rr(source_scratch);
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceAlpha:
            op.workspace.source_runtime.benchmark_update_alpha_from_integral(op.workspace.geometry, params.Ip, 1.0);
            compiler_barrier(&op.workspace.source_runtime.alpha1);
            break;
        case StageKind::SourceUpdate:
            op.workspace.source_runtime.update_pf_psin_uniform_ip(
                op.workspace.geometry, params.Ip, op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.FFn_psin.data());
            break;
        case StageKind::ResidualUpdate:
            op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
            compiler_barrier(op.workspace.residual.surface_fields.data());
            break;
        case StageKind::ResidualThetaReduce:
            op.workspace.residual.benchmark_theta_reduce_into(residual_moments);
            compiler_barrier(residual_moments.data());
            break;
        case StageKind::ResidualRadialProject:
            op.workspace.residual.benchmark_radial_project_from(
                packed, residual_moments, params.a, params.R0, params.B0);
            compiler_barrier(packed.data());
            break;
        case StageKind::ResidualPack:
            op.workspace.residual.pack_into(packed, params.a, params.R0, params.B0);
            compiler_barrier(packed.data());
            break;
        case StageKind::Evaluate:
            op.evaluate(x, packed);
            compiler_barrier(packed.data());
            break;
        case StageKind::EvaluateRing:
            op.evaluate(x, packed);
            compiler_barrier(packed.data());
            break;
        }
    }

    const char* stage_name(StageKind stage) noexcept
    {
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            return "profiles_fixed";
        case StageKind::ProfilesActive:
            return "profiles_active";
        case StageKind::ProfilesAll:
            return "profiles_all";
        case StageKind::GeometryPhase:
            return "geometry_phase";
        case StageKind::GeometryPhaseSincos:
            return "geometry_phase_sincos";
        case StageKind::GeometryPhaseSplitSincos:
            return "geometry_phase_split_sincos";
        case StageKind::GeometryMetricNoStore:
            return "geometry_metric_no_store";
        case StageKind::Geometry:
            return "geometry";
        case StageKind::SourceMaterialize:
            return "source_materialize";
        case StageKind::SourceCopyRegularize:
            return "source_copy_regularize";
        case StageKind::SourceDpsin:
            return "source_D_psin";
        case StageKind::SourceApsin:
            return "source_A_psin";
        case StageKind::SourceDApsin:
            return "source_DA_psin";
        case StageKind::SourceDApsinBlock4:
            return "source_DA_psin_block4";
        case StageKind::SourceInterpolatePair:
            return "source_interpolate_pair";
        case StageKind::SourceIntegrand:
            return "source_integrand";
        case StageKind::SourceAIntegrand:
            return "source_A_integrand";
        case StageKind::SourceAIntegrandRowdot:
            return "source_A_integrand_rowdot";
        case StageKind::SourceNormalize:
            return "source_normalize";
        case StageKind::SourceDNormalized:
            return "source_D_normalized";
        case StageKind::SourceAlpha:
            return "source_alpha";
        case StageKind::SourceUpdate:
            return "source_update";
        case StageKind::ResidualUpdate:
            return "residual_update";
        case StageKind::ResidualThetaReduce:
            return "residual_theta_reduce";
        case StageKind::ResidualRadialProject:
            return "residual_radial_project";
        case StageKind::ResidualPack:
            return "residual_pack";
        case StageKind::Evaluate:
            return "evaluate";
        case StageKind::EvaluateRing:
            return "evaluate_ring";
        }
        return "unknown";
    }

    StageKind parse_stage_one(const std::string& value)
    {
        if (value == "profiles_fixed")
            return StageKind::ProfilesFixed;
        if (value == "profiles_active")
            return StageKind::ProfilesActive;
        if (value == "profiles_all")
            return StageKind::ProfilesAll;
        if (value == "geometry_phase")
            return StageKind::GeometryPhase;
        if (value == "geometry_phase_sincos")
            return StageKind::GeometryPhaseSincos;
        if (value == "geometry_phase_split_sincos")
            return StageKind::GeometryPhaseSplitSincos;
        if (value == "geometry_metric_no_store")
            return StageKind::GeometryMetricNoStore;
        if (value == "geometry")
            return StageKind::Geometry;
        if (value == "source_materialize")
            return StageKind::SourceMaterialize;
        if (value == "source_copy_regularize")
            return StageKind::SourceCopyRegularize;
        if (value == "source_D_psin")
            return StageKind::SourceDpsin;
        if (value == "source_A_psin")
            return StageKind::SourceApsin;
        if (value == "source_DA_psin")
            return StageKind::SourceDApsin;
        if (value == "source_DA_psin_block4")
            return StageKind::SourceDApsinBlock4;
        if (value == "source_interpolate_pair")
            return StageKind::SourceInterpolatePair;
        if (value == "source_integrand")
            return StageKind::SourceIntegrand;
        if (value == "source_A_integrand")
            return StageKind::SourceAIntegrand;
        if (value == "source_A_integrand_rowdot")
            return StageKind::SourceAIntegrandRowdot;
        if (value == "source_normalize")
            return StageKind::SourceNormalize;
        if (value == "source_D_normalized")
            return StageKind::SourceDNormalized;
        if (value == "source_alpha")
            return StageKind::SourceAlpha;
        if (value == "source_update")
            return StageKind::SourceUpdate;
        if (value == "residual_update")
            return StageKind::ResidualUpdate;
        if (value == "residual_theta_reduce")
            return StageKind::ResidualThetaReduce;
        if (value == "residual_radial_project")
            return StageKind::ResidualRadialProject;
        if (value == "residual_pack")
            return StageKind::ResidualPack;
        if (value == "evaluate")
            return StageKind::Evaluate;
        if (value == "evaluate_ring")
            return StageKind::EvaluateRing;
        throw std::invalid_argument("unknown --stage: " + value);
    }

    std::vector<StageKind> stages_for(const std::string& value)
    {
        if (value != "all")
            return {parse_stage_one(value)};
        return {
            StageKind::ProfilesFixed,
            StageKind::ProfilesActive,
            StageKind::ProfilesAll,
            StageKind::GeometryPhase,
            StageKind::GeometryPhaseSincos,
            StageKind::GeometryPhaseSplitSincos,
            StageKind::GeometryMetricNoStore,
            StageKind::Geometry,
            StageKind::SourceMaterialize,
            StageKind::SourceCopyRegularize,
            StageKind::SourceDpsin,
            StageKind::SourceApsin,
            StageKind::SourceDApsin,
            StageKind::SourceDApsinBlock4,
            StageKind::SourceInterpolatePair,
            StageKind::SourceIntegrand,
            StageKind::SourceAIntegrandRowdot,
            StageKind::SourceAIntegrand,
            StageKind::SourceNormalize,
            StageKind::SourceDNormalized,
            StageKind::SourceAlpha,
            StageKind::SourceUpdate,
            StageKind::ResidualUpdate,
            StageKind::ResidualThetaReduce,
            StageKind::ResidualRadialProject,
            StageKind::ResidualPack,
            StageKind::Evaluate,
            StageKind::EvaluateRing,
        };
    }

    size_t parse_size_arg(const std::string& name, const std::string& value, bool allow_zero)
    {
        if (value.empty() || value.front() == '+' || value.front() == '-')
            throw std::invalid_argument(name + " must be a base-10 non-negative integer");

        size_t      parsed = 0;
        const char* first  = value.data();
        const char* last   = value.data() + value.size();
        const auto  result = std::from_chars(first, last, parsed, 10);
        if (result.ec != std::errc{} || result.ptr != last)
            throw std::invalid_argument(name + " must be a base-10 non-negative integer");
        if (!allow_zero && parsed == 0)
            throw std::invalid_argument(name + " must be positive");
        return parsed;
    }

    Options parse_args(int argc, char** argv)
    {
        Options options{};
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help")
            {
                std::cout << "usage: veqlib_main --mode stage [--stage all|profiles_fixed|profiles_active|"
                             "profiles_all|geometry_phase|geometry_phase_sincos|geometry_phase_split_sincos|"
                             "geometry_metric_no_store|geometry|source_materialize|source_copy_regularize|"
                             "source_D_psin|source_A_psin|source_DA_psin|source_DA_psin_block4|"
                             "source_interpolate_pair|source_integrand|"
                             "source_A_integrand|source_A_integrand_rowdot|source_normalize|source_D_normalized|source_alpha|"
                             "source_update|residual_update|residual_theta_reduce|residual_radial_project|residual_pack|"
                             "evaluate|evaluate_ring] [--repeat N] [--warmup N] "
                             "[--inner N] [--ring-size N]\n";
                std::exit(0);
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--stage")
                options.stage = value;
            else if (arg == "--repeat")
                options.repeat = parse_size_arg(arg, value, false);
            else if (arg == "--warmup")
                options.warmup = parse_size_arg(arg, value, true);
            else if (arg == "--inner")
                options.inner = parse_size_arg(arg, value, false);
            else if (arg == "--ring-size")
                options.ring_size = parse_size_arg(arg, value, false);
            else
                throw std::invalid_argument("unknown argument: " + arg);
        }
        return options;
    }

    Stats compute_stats(const std::vector<double>& samples)
    {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());

        double total = 0.0;
        for (double sample : samples)
            total += sample;
        const double avg = total / static_cast<double>(samples.size());

        double variance = 0.0;
        for (double sample : samples)
        {
            const double delta = sample - avg;
            variance += delta * delta;
        }
        variance /= static_cast<double>(samples.size());

        const size_t median_index = sorted.size() / 2;
        const double median =
            sorted.size() % 2 == 0 ? 0.5 * (sorted[median_index - 1] + sorted[median_index]) : sorted[median_index];
        const size_t p95_index = ((sorted.size() * 95 + 99) / 100) - 1;

        return {
            .avg_ns    = avg,
            .min_ns    = sorted.front(),
            .max_ns    = sorted.back(),
            .median_ns = median,
            .p95_ns    = sorted[p95_index],
            .std_ns    = std::sqrt(variance),
        };
    }

    void prepare_for_stage(StageKind                                   stage,
                           BenchOperator&                              op,
                           std::span<const double, BenchShape::x_size> x,
                           SourceRadialVector&                         source_scratch,
                           SourceRadialVector&                         source_aux,
                           ResidualMomentRows&                         residual_moments)
    {
        const auto& params = op.runtime_params();
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            break;
        case StageKind::ProfilesActive:
            op.refresh_static_plan();
            break;
        case StageKind::ProfilesAll:
            break;
        case StageKind::GeometryPhase:
        case StageKind::GeometryPhaseSincos:
        case StageKind::GeometryPhaseSplitSincos:
        case StageKind::GeometryMetricNoStore:
        case StageKind::Geometry:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceMaterialize:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceCopyRegularize:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceDpsin:
        case StageKind::SourceApsin:
        case StageKind::SourceDApsin:
        case StageKind::SourceDApsinBlock4:
            prepare_source_profile_root(op, x);
            break;
        case StageKind::SourceInterpolatePair:
            prepare_source_materialized(op, x);
            break;
        case StageKind::SourceIntegrand:
            prepare_source_materialized(op, x);
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            break;
        case StageKind::SourceAIntegrand:
        case StageKind::SourceAIntegrandRowdot:
            prepare_source_integrand(op, x, source_scratch);
            break;
        case StageKind::SourceNormalize:
            prepare_source_accumulated_integrand(op, x, source_scratch);
            break;
        case StageKind::SourceDNormalized:
            prepare_source_normalized_psin(op, x, source_scratch);
            break;
        case StageKind::SourceAlpha:
            prepare_source_updated(op, x);
            break;
        case StageKind::SourceUpdate:
            prepare_source_materialized(op, x);
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            break;
        case StageKind::ResidualUpdate:
            prepare_source_updated(op, x);
            break;
        case StageKind::ResidualThetaReduce:
            prepare_residual_updated(op, x);
            break;
        case StageKind::ResidualRadialProject:
            prepare_residual_updated(op, x);
            op.workspace.residual.benchmark_theta_reduce_into(residual_moments);
            break;
        case StageKind::ResidualPack:
            prepare_residual_updated(op, x);
            break;
        case StageKind::Evaluate:
            break;
        case StageKind::EvaluateRing:
            break;
        }
    }

    std::vector<std::array<double, BenchShape::x_size>> make_state_ring(size_t ring_size)
    {
        std::vector<std::array<double, BenchShape::x_size>> states(ring_size);
        for (size_t state = 0; state < ring_size; ++state)
        {
            const double phase = static_cast<double>(state + 1) / static_cast<double>(ring_size);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
            {
                const double mode = static_cast<double>((i % 5) + 1);
                states[state][i] = 0.02 * std::sin(mode * phase) + 0.01 * std::cos(mode + phase);
            }
        }
        return states;
    }

    nlohmann::json run_benchmark(StageKind stage, const Options& options)
    {
        auto op = std::make_unique<BenchOperator>();
        configure_operator(*op);
        std::array<double, BenchShape::x_size> x{};
        auto                                  state_ring = make_state_ring(options.ring_size);
        PackedVector                           packed{};
        SourceRadialVector                     source_scratch{};
        SourceRadialVector                     source_aux{};
        ResidualMomentRows                     residual_moments{};
        const auto                             x_values = x_span(x);

        prepare_for_stage(stage, *op, x_values, source_scratch, source_aux, residual_moments);

        for (size_t sample = 0; sample < options.warmup; ++sample)
            for (size_t i = 0; i < options.inner; ++i)
            {
                const auto input = stage == StageKind::EvaluateRing
                    ? x_span(state_ring[(sample * options.inner + i) % state_ring.size()])
                    : x_values;
                run_stage_once(stage, *op, input, packed, source_scratch, source_aux, residual_moments);
            }

        std::vector<double> samples;
        samples.reserve(options.repeat);
        using clock = std::chrono::steady_clock;
        for (size_t sample = 0; sample < options.repeat; ++sample)
        {
            const auto start = clock::now();
            for (size_t i = 0; i < options.inner; ++i)
            {
                const auto input = stage == StageKind::EvaluateRing
                    ? x_span(state_ring[(sample * options.inner + i) % state_ring.size()])
                    : x_values;
                run_stage_once(stage, *op, input, packed, source_scratch, source_aux, residual_moments);
            }
            const auto                                     stop    = clock::now();
            const std::chrono::duration<double, std::nano> elapsed = stop - start;
            samples.push_back(elapsed.count() / static_cast<double>(options.inner));
        }

        benchmark_sink += consume_state(*op, packed, source_scratch, residual_moments);
        const Stats stats = compute_stats(samples);

        return {
            {"stage", stage_name(stage)},
            {"repeat", options.repeat},
            {"warmup", options.warmup},
            {"inner", options.inner},
            {"ring_size", stage == StageKind::EvaluateRing ? options.ring_size : size_t{1}},
            {"calls", options.repeat * options.inner},
            {"avg_ns", stats.avg_ns},
            {"min_ns", stats.min_ns},
            {"max_ns", stats.max_ns},
            {"median_ns", stats.median_ns},
            {"p95_ns", stats.p95_ns},
            {"std_ns", stats.std_ns},
            {"samples_ns", samples},
        };
    }
} // namespace

int run(int argc, char** argv)
{
    try
    {
        const Options  options = parse_args(argc, argv);
        nlohmann::json results = nlohmann::json::array();
        for (StageKind stage : stages_for(options.stage))
            results.push_back(run_benchmark(stage, options));

        const double         sink   = benchmark_sink;
        const nlohmann::json report = {
            {"schema_version", 1},
            {"case_name", "PF_psin_uniform_Ip"},
            {"topology",
             {
                 {"Nr", BenchGrid::radial_nodes},
                 {"Nt", BenchGrid::theta_rows},
                 {"L_max", BenchShape::L_max},
                 {"M_max", BenchShape::M_max},
                 {"K_max", BenchShape::K_max},
                 {"x_size", BenchShape::x_size},
             }},
            {"results", results},
            {"sink", sink},
        };
        std::cout << report.dump(2) << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "veqlib_main --mode stage: " << error.what() << '\n';
        return 2;
    }
}

} // namespace veqlib_stage_benchmark_cli

// ---- temporary validation CLI ----
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>

#include <cminpack.h>
#include <gcem.hpp>
#include <lapacke.h>
#include <nlohmann/json.hpp>

#include "config.h"
#include "geometry.h"
#include "grid.h"
#include "linalg.h"
#include "math.h"
#include "source/pf_psin_uniform_ip.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_temp_validation_cli
{

namespace
{
    using grid::CFD33;
    using grid::CFD35;
    using grid::CFD55;
    using grid::Chebyshev;
    using grid::Grid;
    using grid::Legendre;
    using grid::Lobatto;
    using grid::Radau;
    using grid::Spectral;
    using geometry::GeometryRuntime;
    using geometry::radial_Kn;
    using geometry::radial_Ln_r;
    using geometry::radial_S_r;
    using geometry::radial_V_r;
    using geometry::surface_J;
    using geometry::surface_JdivR;
    using geometry::surface_R;
    using geometry::surface_R_t;
    using geometry::surface_sin_tb;
    using geometry::surface_Z_t;
    using linalg::BunchKaufman;
    using linalg::Cholesky;
    using linalg::Context;
    using linalg::Doolittle;
    using linalg::GolubReinsch;
    using linalg::Householder;
    using linalg::Thomas;
    using source::PfPsinUniformIpOperator;
    using residual::ResidualRuntime;
    using linalg::factorize;
    using linalg::factorize_into;
    using linalg::matmul;
    using linalg::matmul_into;
    using linalg::solve;
    using linalg::solve_into;
    using linalg::transpose;
    using linalg::transpose_into;
    using std::size_t;
    using source::PfPsinUniformIpSourceRuntime;
    using source::UniformSourceShape;
    using source::root_psin;
    using source::root_psin_r;
    using source::root_psin_rr;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    using Topology = config::DefaultTopology;
    using ProbeGrid = Grid<
        Topology::Nr,
        Topology::Nt,
        Topology::L_max,
        Topology::M_max,
        Topology::K_max,
        Legendre,
        Spectral>;

    using ProbeProfilesFromCounts = profiles::Profiles<
        Topology::L_max,
        Topology::K_max,
        Topology::h_count,
        Topology::v_count,
        Topology::kappa_count,
        Topology::psin_count,
        Topology::F_count,
        Topology::c_family_counts,
        Topology::s_family_counts>;

    constexpr auto topology_c_slots = profiles::tail_optimized_slots_from_counts<Topology::c_family_counts>();
    constexpr auto topology_s_slots = profiles::optimized_slots_from_counts<Topology::s_family_counts>();

    using ProbeProfileShape = profiles::ProfileShape<
        Topology::L_max,
        Topology::K_max,
        Topology::M_max,
        profiles::optimized_slot_from_count(Topology::h_count),
        profiles::optimized_slot_from_count(Topology::v_count),
        profiles::optimized_slot_from_count(Topology::kappa_count),
        profiles::first_optimized_slot_from_counts<Topology::c_family_counts>(),
        profiles::optimized_slot_from_count(Topology::psin_count),
        profiles::optimized_slot_from_count(Topology::F_count),
        topology_c_slots,
        topology_s_slots>;
    using ProbeProfiles = profiles::ProfileEvaluator<ProbeProfileShape>;

    constexpr auto no_c_slots = std::array<profiles::ProfileSlot, 0>{};
    constexpr auto no_s_slots = std::array<profiles::ProfileSlot, 0>{};

    using FixedOnlyProfileShape = profiles::ProfileShape<
        1,
        2,
        1,
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;

    using CircularGeometryShape = profiles::ProfileShape<
        2,
        2,
        2,
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using CircularGeometryGrid     = Grid<8, 8, 2, 2, 2, Legendre, Spectral>;
    using CircularGeometryProfiles = profiles::RuntimeProfiles<CircularGeometryShape, CircularGeometryGrid>;
    using CircularGeometryRuntime  = GeometryRuntime<CircularGeometryGrid>;

    using SourceMaterializationShape = profiles::ProfileShape<
        3,
        2,
        1,
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::optimized_slot(3),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using SourceMaterializationGrid     = Grid<8, 8, 3, 1, 2, Legendre, Spectral>;
    using SourceMaterializationProfiles = profiles::RuntimeProfiles<SourceMaterializationShape, SourceMaterializationGrid>;
    using SourceMaterializationRuntime =
        PfPsinUniformIpSourceRuntime<SourceMaterializationGrid, UniformSourceShape<5>>;

    constexpr auto residual_c_slots = std::array{
        profiles::optimized_slot(2),
    };
    constexpr auto residual_s_slots = std::array{
        profiles::optimized_slot(2),
    };

    using ResidualProbeShape = profiles::ProfileShape<
        2,
        2,
        1,
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::optimized_slot(2),
        profiles::absent_slot(),
        residual_c_slots,
        residual_s_slots>;
    using ResidualProbeGrid     = Grid<8, 8, 2, 1, 2, Legendre, Spectral>;
    using ResidualProbeProfiles = profiles::RuntimeProfiles<ResidualProbeShape, ResidualProbeGrid>;
    using ResidualProbeSource   = PfPsinUniformIpSourceRuntime<ResidualProbeGrid, UniformSourceShape<5>>;
    using ResidualProbeGeometry = GeometryRuntime<ResidualProbeGrid>;
    using ResidualProbeRuntime  = ResidualRuntime<ResidualProbeShape, ResidualProbeGrid>;
    using ResidualProbeOperator = PfPsinUniformIpOperator<ResidualProbeShape, ResidualProbeGrid, UniformSourceShape<5>>;

    constexpr auto mixed_c_slots = std::array{
        profiles::optimized_slot(2),
        profiles::fixed_slot(),
    };
    constexpr auto mixed_s_slots = std::array{
        profiles::absent_slot(),
        profiles::optimized_slot(3),
    };

    using MixedProfileShape = profiles::ProfileShape<
        2,
        2,
        2,
        profiles::optimized_slot(2),
        profiles::fixed_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        profiles::absent_slot(),
        mixed_c_slots,
        mixed_s_slots>;
    using RuntimeProbeGrid     = Grid<8, 8, 2, 2, 2, Legendre, Spectral>;
    using MixedRuntimeProfiles = profiles::RuntimeProfiles<MixedProfileShape, RuntimeProbeGrid>;

    constexpr auto semantic_c_slots = std::array{
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
    };
    constexpr auto semantic_s_slots = std::array{
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
    };

    using RuntimeSemanticShape = profiles::ProfileShape<
        2,
        2,
        4,
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::absent_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        profiles::optimized_slot(2),
        semantic_c_slots,
        semantic_s_slots>;
    using RuntimeSemanticGrid     = Grid<8, 8, 2, 4, 2, Legendre, Spectral>;
    using RuntimeSemanticProfiles = profiles::RuntimeProfiles<RuntimeSemanticShape, RuntimeSemanticGrid>;
    using RuntimeSemanticEvaluator = profiles::ProfileEvaluator<RuntimeSemanticShape>;

    template <typename Shape, size_t HCount>
    consteval bool h_profile_L_matches()
    {
        if constexpr (HCount > 0)
            return Shape::profile_L[Shape::h_profile_id] == static_cast<int>(HCount - 1);
        else
            return Shape::profile_L[Shape::h_profile_id] == -1;
    }

    template <typename Shape, typename TopologyType, size_t SMax>
    consteval bool highest_s_profile_L_matches()
    {
        if constexpr (SMax > 0)
            return Shape::profile_L[Shape::template s_profile_id<SMax>()] ==
                   static_cast<int>(TopologyType::template s_count<SMax>() - 1);
        else
            return Shape::s_family_source_profile_ids[0] == -1;
    }

    static_assert(Topology::fourier_power<Topology::K_max + 7>() == Topology::K_max);
    static_assert(ProbeProfiles::fourier_power<Topology::K_max + 7>() == Topology::K_max);

    static_assert(ProbeProfileShape::h_profile_id == 0);
    static_assert(ProbeProfiles::shape::profile_count == ProbeProfilesFromCounts::shape::profile_count);
    static_assert(ProbeProfiles::shape::x_size == ProbeProfilesFromCounts::shape::x_size);
    static_assert(ProbeProfileShape::v_profile_id == 1);
    static_assert(ProbeProfileShape::kappa_profile_id == 2);
    static_assert(ProbeProfileShape::c_profile_id<0>() == 3);
    static_assert(ProbeProfileShape::s_profile_id<1>() == Topology::M_max + 4);
    static_assert(ProbeProfileShape::psin_profile_id == 2 * Topology::M_max + 4);
    static_assert(ProbeProfileShape::F_profile_id == 2 * Topology::M_max + 5);
    static_assert(ProbeProfileShape::profile_count == 2 * Topology::M_max + 6);
    static_assert(h_profile_L_matches<ProbeProfileShape, Topology::h_count>());
    static_assert(highest_s_profile_L_matches<ProbeProfileShape, Topology, Topology::S_max>());
    static_assert(ProbeProfileShape::coeff_index[ProbeProfileShape::h_profile_id][0] == 0);
    static_assert(ProbeProfileShape::order_offsets[0] == 0);
    static_assert(ProbeProfileShape::order_offsets[ProbeProfileShape::max_active_len] ==
                  static_cast<int>(ProbeProfileShape::x_size));
    static_assert(ProbeProfileShape::s_family_source_profile_ids[0] == -1);

    static_assert(FixedOnlyProfileShape::profile_count == 8);
    static_assert(FixedOnlyProfileShape::active_count == 0);
    static_assert(FixedOnlyProfileShape::max_active_len == 0);
    static_assert(FixedOnlyProfileShape::x_size == 0);
    static_assert(FixedOnlyProfileShape::profile_L[FixedOnlyProfileShape::h_profile_id] == -1);
    static_assert(FixedOnlyProfileShape::c_family_source_profile_ids[1] == -1);
    static_assert(FixedOnlyProfileShape::s_family_source_profile_ids[0] == -1);

    static_assert(MixedProfileShape::profile_count == 10);
    static_assert(MixedProfileShape::active_count == 4);
    static_assert(MixedProfileShape::max_active_len == 3);
    static_assert(MixedProfileShape::x_size == 8);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::h_profile_id] == 1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::v_profile_id] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<0>()] == 0);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<1>()] == 1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::c_profile_id<2>()] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::s_profile_id<1>()] == -1);
    static_assert(MixedProfileShape::profile_L[MixedProfileShape::s_profile_id<2>()] == 2);
    static_assert(MixedProfileShape::active_profile_ids[0] == MixedProfileShape::h_profile_id);
    static_assert(MixedProfileShape::active_profile_ids[1] == MixedProfileShape::c_profile_id<0>());
    static_assert(MixedProfileShape::active_profile_ids[2] == MixedProfileShape::c_profile_id<1>());
    static_assert(MixedProfileShape::active_profile_ids[3] == MixedProfileShape::s_profile_id<2>());
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::h_profile_id][0] == 0);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<0>()][0] == 1);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<1>()][0] == 2);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][0] == 3);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::h_profile_id][1] == 4);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::c_profile_id<1>()][1] == 5);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][1] == 6);
    static_assert(MixedProfileShape::coeff_index[MixedProfileShape::s_profile_id<2>()][2] == 7);
    static_assert(MixedProfileShape::order_offsets[0] == 0);
    static_assert(MixedProfileShape::order_offsets[1] == 4);
    static_assert(MixedProfileShape::order_offsets[2] == 7);
    static_assert(MixedProfileShape::order_offsets[3] == 8);
    static_assert(MixedProfileShape::c_family_source_profile_ids[0] ==
                  static_cast<int>(MixedProfileShape::c_profile_id<0>()));
    static_assert(MixedProfileShape::c_family_source_profile_ids[2] ==
                  static_cast<int>(MixedProfileShape::c_profile_id<2>()));
    static_assert(MixedProfileShape::s_family_source_profile_ids[0] == -1);
    static_assert(MixedProfileShape::s_family_source_profile_ids[1] == -1);
    static_assert(MixedProfileShape::s_family_source_profile_ids[2] ==
                  static_cast<int>(MixedProfileShape::s_profile_id<2>()));
    static_assert(MixedRuntimeProfiles::profile_field_count == MixedProfileShape::profile_count);
    static_assert(MixedRuntimeProfiles::family_field_count == MixedProfileShape::M_max + 1);
    static_assert(RuntimeSemanticEvaluator::fourier_power<4>() == RuntimeSemanticShape::K_max);

    constexpr double tolerance = 1.0e-8;

    constexpr bool close(double lhs, double rhs, double tol = tolerance) { return math::abs(lhs - rhs) <= tol; }

    constexpr double pow_integer(double base, size_t exponent)
    {
        double value = 1.0;
        for (size_t i = 0; i < exponent; ++i)
            value *= base;
        return value;
    }

    constexpr double power_value(double rho, size_t power) { return pow_integer(rho, power); }

    constexpr double power_radial(double rho, size_t power)
    {
        return power == 0 ? 0.0 : static_cast<double>(power) * pow_integer(rho, power - 1);
    }

    constexpr double power_radial2(double rho, size_t power)
    {
        return power < 2 ? 0.0 : static_cast<double>(power * (power - 1)) * pow_integer(rho, power - 2);
    }

    template <size_t Count>
    constexpr Vector<double, Count> make_profile_coefficients(double base, double step)
    {
        Vector<double, Count> coeffs{uninitialized};
        for (size_t i = 0; i < Count; ++i)
            coeffs[i] = base + step * static_cast<double>(i);
        return coeffs;
    }

    template <typename Shape, size_t ProfileId, size_t Count>
    constexpr void write_profile_coefficients(Vector<double, Shape::x_size>& x, const Vector<double, Count>& coeffs)
    {
        for (size_t degree = 0; degree < Count; ++degree)
            x[static_cast<size_t>(Shape::coeff_index[ProfileId][degree])] = coeffs[degree];
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_value(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = coeffs[0];
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_radial(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = 0.0;
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T_r(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr double profile_poly_radial2(const Vector<double, Count>& coeffs, size_t node)
    {
        double value = 0.0;
        for (size_t k = 1; k < Count; ++k)
            value += coeffs[k] * GridType::T_rr(k - 1, node);
        return value;
    }

    template <typename GridType, size_t Count>
    constexpr bool check_enveloped_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                           const Vector<double, Count>&                    coeffs,
                                           size_t                                         node)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);

        return close(profiles(node, 0), y * value) &&
               close(profiles(node, 1), -2.0 * rho * value + y * radial) &&
               close(profiles(node, 2), -2.0 * value - 4.0 * rho * radial + y * radial2);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_kappa_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                       const Vector<double, Count>&                    coeffs,
                                       size_t                                         node,
                                       double                                         ka)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;

        return close(profiles(node, 0), ka + base) &&
               close(profiles(node, 1), base_r) &&
               close(profiles(node, 2), base_rr);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_psin_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                      const Vector<double, Count>&                    coeffs,
                                      size_t                                         node)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp     = 1.0 + base;
        const double rp      = rho * rho;
        const double rp_r    = 2.0 * rho;
        const double rp_rr   = 2.0;

        return close(profiles(node, 0), rp * amp) &&
               close(profiles(node, 1), rp_r * amp + rp * base_r) &&
               close(profiles(node, 2), rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr);
    }

    template <typename GridType, size_t Count>
    constexpr bool check_F_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                   const Vector<double, Count>&                    coeffs,
                                   size_t                                         node,
                                   double                                         scale)
    {
        const double rho               = GridType::nodes[node];
        const double y                 = GridType::y[node];
        const double value             = profile_poly_value<GridType>(coeffs, node);
        const double radial            = profile_poly_radial<GridType>(coeffs, node);
        const double radial2           = profile_poly_radial2<GridType>(coeffs, node);
        const double base              = y * value;
        const double base_r            = -2.0 * rho * value + y * radial;
        const double base_rr           = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp_raw_unclamped = 1.0 + base;
        const double amp_raw           = math::max(amp_raw_unclamped, 1.0e-10);
        const double amp               = math::sqrt(amp_raw);
        const double inv_amp           = 1.0 / amp;
        const double inv_amp3          = inv_amp / amp_raw;
        const double amp_r             = 0.5 * base_r * inv_amp;
        const double amp_rr            = 0.5 * base_rr * inv_amp - 0.25 * base_r * base_r * inv_amp3;

        return close(profiles(node, 0), scale * amp) &&
               close(profiles(node, 1), scale * amp_r) &&
               close(profiles(node, 2), scale * amp_rr);
    }

    template <size_t Power, typename GridType, size_t Count>
    constexpr bool check_fourier_profile(const Matrix<double, GridType::nodes.count, 3>& profiles,
                                         const Vector<double, Count>&                    coeffs,
                                         size_t                                         node,
                                         double                                         offset)
    {
        const double rho     = GridType::nodes[node];
        const double y       = GridType::y[node];
        const double value   = profile_poly_value<GridType>(coeffs, node);
        const double radial  = profile_poly_radial<GridType>(coeffs, node);
        const double radial2 = profile_poly_radial2<GridType>(coeffs, node);
        const double base    = y * value;
        const double base_r  = -2.0 * rho * value + y * radial;
        const double base_rr = -2.0 * value - 4.0 * rho * radial + y * radial2;
        const double amp     = offset + base;
        const double rp      = power_value(rho, Power);
        const double rp_r    = power_radial(rho, Power);
        const double rp_rr   = power_radial2(rho, Power);

        return close(profiles(node, 0), rp * amp) &&
               close(profiles(node, 1), rp_r * amp + rp * base_r) &&
               close(profiles(node, 2), rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr);
    }

    template <typename Values>
    constexpr double sum_values(const Values& values)
    {
        double total = 0.0;
        for (size_t i = 0; i < Values::count; ++i)
            total += values[i];
        return total;
    }

    template <typename Quadrature, size_t N>
    constexpr double max_moment_error(size_t max_degree)
    {
        const auto& nodes   = Quadrature::template nodes<N>;
        const auto& weights = Quadrature::template weights<N>;
        double      worst   = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
        {
            double value = 0.0;
            for (size_t i = 0; i < N; ++i)
                value += weights[i] * pow_integer(nodes[i], degree);

            const double exact = 1.0 / static_cast<double>(degree + 1);
            const double error = math::abs(value - exact);
            if (error > worst)
                worst = error;
        }
        return worst;
    }

    template <typename Quadrature, size_t N>
    constexpr bool quadrature_shape_ok()
    {
        const auto& nodes   = Quadrature::template nodes<N>;
        const auto& weights = Quadrature::template weights<N>;
        if (!close(sum_values(weights), 1.0, 1.0e-12))
            return false;

        for (size_t i = 0; i < N; ++i)
        {
            if (!math::is_finite(nodes[i]) || !math::is_finite(weights[i]))
                return false;
            if (nodes[i] < 0.0 || nodes[i] > 1.0 || weights[i] <= 0.0)
                return false;
            if (i > 0 && nodes[i] <= nodes[i - 1])
                return false;
        }
        return true;
    }

    template <typename MatrixType, typename Nodes>
    constexpr double apply_to_power(const MatrixType& matrix, const Nodes& nodes, size_t row, size_t power)
    {
        const auto* values = matrix.data();
        double      total  = 0.0;
        for (size_t col = 0; col < Nodes::count; ++col)
            total += values[row * Nodes::count + col] * pow_integer(nodes[col], power);
        return total;
    }

    template <typename Calculus, typename Quadrature, size_t N>
    constexpr double max_differentiator_error(size_t max_degree)
    {
        const auto& nodes = Quadrature::template nodes<N>;
        const auto& diff  = Calculus::template differentiator<N, Quadrature>;
        double      worst = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
            for (size_t row = 0; row < N; ++row)
            {
                const double exact =
                    degree == 0 ? 0.0 : static_cast<double>(degree) * pow_integer(nodes[row], degree - 1);
                const double error = math::abs(apply_to_power(diff, nodes, row, degree) - exact);
                if (error > worst)
                    worst = error;
            }
        return worst;
    }

    template <typename Calculus, typename Quadrature, size_t N>
    constexpr double max_accumulator_error(size_t max_degree)
    {
        const auto& nodes = Quadrature::template nodes<N>;
        const auto& acc   = Calculus::template accumulator<N, Quadrature>;
        double      worst = 0.0;

        for (size_t degree = 0; degree <= max_degree; ++degree)
            for (size_t row = 0; row < N; ++row)
            {
                const double exact = pow_integer(nodes[row], degree + 1) / static_cast<double>(degree + 1);
                const double error = math::abs(apply_to_power(acc, nodes, row, degree) - exact);
                if (error > worst)
                    worst = error;
            }
        return worst;
    }

    constexpr Matrix<double, 2, 2> dense_matrix{3.0, 1.0, 1.0, 2.0};
    constexpr Matrix<double, 2, 1> dense_rhs{9.0, 8.0};
    constexpr Matrix<double, 3, 2> tall_matrix{1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    constexpr Matrix<double, 3, 1> tall_rhs{2.0, 3.0, 5.0};
    constexpr Matrix<double, 3, 4> thomas_band{0.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, 0.0};
    constexpr Matrix<double, 4, 2> thomas_rhs{1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0};

    constexpr bool linalg_constexpr_ok()
    {
        const auto product = matmul(dense_matrix, dense_matrix);
        if (!close(product[0], 10.0) || !close(product[1], 5.0) || !close(product[2], 5.0) || !close(product[3], 5.0))
            return false;

        Matrix<double, 2, 2> product_into{uninitialized};
        matmul_into(product_into, dense_matrix, dense_matrix);
        if (!close(product_into[0], product[0]) || !close(product_into[3], product[3]))
            return false;

        Matrix<double, 2, 2> transposed = transpose(dense_matrix);
        transpose_into(transposed, transposed);
        if (!close(transposed[1], dense_matrix[1]) || !close(transposed[2], dense_matrix[2]))
            return false;

        const auto doolittle = solve<Doolittle>(dense_matrix, dense_rhs);
        const auto cholesky  = solve<Cholesky>(dense_matrix, dense_rhs);
        const auto bunch     = solve<BunchKaufman>(dense_matrix, dense_rhs);
        const auto qr        = solve<Householder>(tall_matrix, tall_rhs);
        const auto thomas    = solve<Thomas>(thomas_band, thomas_rhs);

        Matrix<double, 2, 1> doolittle_into{uninitialized};
        solve_into<Doolittle>(doolittle_into, dense_matrix, dense_rhs);

        Context<Doolittle, 2, 2> context;
        factorize_into<Doolittle>(context, dense_matrix);
        auto context_rhs = dense_rhs;
        context.substitute_inplace<1>(context_rhs.data());

        const auto thomas_context = factorize<Thomas>(thomas_band);
        auto       thomas_work    = thomas_rhs;
        thomas_context.substitute_inplace<2>(thomas_work.data());

        return close(doolittle[0], 2.0) && close(doolittle[1], 3.0) && close(cholesky[0], 2.0) &&
               close(cholesky[1], 3.0) && close(bunch[0], 2.0) && close(bunch[1], 3.0) && close(qr[0], 2.0) &&
               close(qr[1], 3.0) && close(doolittle_into[0], 2.0) && close(doolittle_into[1], 3.0) &&
               close(context_rhs[0], 2.0) && close(context_rhs[1], 3.0) && close(thomas[0], 1.0) &&
               close(thomas[1], 2.0) && close(thomas[6], 1.0) && close(thomas[7], 2.0) && close(thomas_work[0], 1.0) &&
               close(thomas_work[1], 2.0) && close(thomas_work[6], 1.0) && close(thomas_work[7], 2.0);
    }

    constexpr bool tensor_math_constexpr_ok()
    {
        constexpr Vector<double, 3> values{1.0, 2.0, 3.0};
        constexpr auto              shifted = values + 1.0;
        constexpr auto              scaled  = 2.0 * values;
        constexpr auto              rooted  = math::sqrt(scaled + values);

        return close(math::sum(values), 6.0) && close(math::dot(values, values), 14.0) &&
               close(math::norm2(values), gcem::sqrt(14.0)) && close(shifted[2], 4.0) && close(scaled[1], 4.0) &&
               close(rooted[0], gcem::sqrt(3.0)) && math::is_finite(rooted);
    }

    constexpr bool finite_semantics_constexpr_ok()
    {
        constexpr double positive_inf = std::bit_cast<double>(0x7ff0'0000'0000'0000ULL);
        constexpr double quiet_nan    = std::bit_cast<double>(0x7ff8'0000'0000'0001ULL);
        constexpr double large_finite = std::numeric_limits<double>::max() / 2.0;

        constexpr Vector<double, 2> finite_values{1.0, -large_finite};
        constexpr Vector<double, 2> nan_values{1.0, quiet_nan};

        return math::is_finite(1.0) && math::is_finite(large_finite) && !math::is_finite(positive_inf) &&
               !math::is_finite(quiet_nan) && math::is_finite(finite_values) && !math::is_finite(nan_values);
    }

    constexpr bool grid_constexpr_ok()
    {
        constexpr double rho0              = ProbeGrid::nodes[0];
        constexpr double x0                = 2.0 * rho0 * rho0 - 1.0;
        constexpr double theta_step        = 2.0 * grid::detail::pi / static_cast<double>(ProbeGrid::theta_rows);
        constexpr bool   radial_tables_ok  = close(ProbeGrid::x[0], x0) &&
                                            close(ProbeGrid::y[0], 1.0 - rho0 * rho0) &&
                                            close(ProbeGrid::rhos(0, 0), rho0) &&
                                            close(ProbeGrid::rhos(1, 0), rho0 * rho0);
        constexpr bool theta_tables_ok = close(ProbeGrid::theta[0], 0.0, 0.0) &&
                                         close(ProbeGrid::theta[1], theta_step) &&
                                         close(ProbeGrid::cos_mtheta(0, 3), 1.0) &&
                                         close(ProbeGrid::sin_mtheta(0, 3), 0.0, 1.0e-15) &&
                                         close(ProbeGrid::m_cos_mtheta(0, 3), 0.0, 0.0) &&
                                         close(ProbeGrid::m2_sin_mtheta(0, 3), 0.0, 0.0);
        constexpr bool chebyshev_tables_ok = close(ProbeGrid::T(0, 0), ProbeGrid::x[0]) &&
                                             close(ProbeGrid::T_r(0, 0), 4.0 * ProbeGrid::nodes[0]) &&
                                             close(ProbeGrid::T_rr(0, 0), 4.0);
        constexpr bool harmonic_tables_ok = close(ProbeGrid::cos_mtheta(1, 2), math::cos(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::sin_mtheta(1, 2), math::sin(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::m_cos_mtheta(1, 2), math::cos(ProbeGrid::theta[2])) &&
                                            close(ProbeGrid::m_sin_mtheta(1, 2), math::sin(ProbeGrid::theta[2]));

        return ProbeGrid::nodes.count == ProbeGrid::radial_nodes &&
               ProbeGrid::weights.count == ProbeGrid::radial_nodes &&
               ProbeGrid::accumulator.shape[0] == ProbeGrid::radial_nodes &&
               ProbeGrid::differentiator.shape[1] == ProbeGrid::radial_nodes && radial_tables_ok && theta_tables_ok &&
               chebyshev_tables_ok && harmonic_tables_ok && quadrature_shape_ok<Chebyshev, 8>() &&
               quadrature_shape_ok<Legendre, 8>() && quadrature_shape_ok<Lobatto, 8>() &&
               quadrature_shape_ok<Radau, 8>() && close(Lobatto::nodes<8>[0], 0.0, 0.0) &&
               close(Lobatto::nodes<8>[7], 1.0, 0.0) && close(Radau::nodes<8>[7], 1.0, 0.0) &&
               max_moment_error<Chebyshev, 16>(7) < 1.0e-11 && max_moment_error<Legendre, 16>(15) < 1.0e-10 &&
               max_moment_error<Lobatto, 16>(13) < 1.0e-10 && max_moment_error<Radau, 16>(14) < 1.0e-10 &&
               max_differentiator_error<Spectral, Chebyshev, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Legendre, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Lobatto, 8>(3) < 1.0e-8 &&
               max_differentiator_error<Spectral, Radau, 8>(3) < 1.0e-8 &&
               max_accumulator_error<Spectral, Chebyshev, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Legendre, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Lobatto, 8>(2) < 1.0e-8 &&
               max_accumulator_error<Spectral, Radau, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD33, Lobatto, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD35, Lobatto, 8>(2) < 1.0e-8 &&
               max_differentiator_error<CFD55, Lobatto, 8>(2) < 1.0e-8 &&
               max_accumulator_error<CFD33, Lobatto, 8>(1) < 1.0e-8 &&
               max_accumulator_error<CFD35, Lobatto, 8>(1) < 1.0e-8 &&
               max_accumulator_error<CFD55, Lobatto, 8>(1) < 1.0e-8;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool h_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(0.12, -0.003);
            ProfileOps::update_h(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_enveloped_profile<ProbeGrid>(out, coeffs, 0) &&
                   check_enveloped_profile<ProbeGrid>(out, coeffs, ProbeGrid::nodes.count - 1);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool v_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(-0.08, 0.002);
            ProfileOps::update_v(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_enveloped_profile<ProbeGrid>(out, coeffs, 1);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool kappa_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto   coeffs = make_profile_coefficients<Count>(0.05, 0.001);
            constexpr double ka = 1.7;
            ProfileOps::update_kappa(
                out,
                coeffs,
                ProbeGrid::T,
                ProbeGrid::T_r,
                ProbeGrid::T_rr,
                ProbeGrid::rhos,
                ka
            );
            return math::is_finite(out) && check_kappa_profile<ProbeGrid>(out, coeffs, 2, ka);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool psin_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto coeffs = make_profile_coefficients<Count>(0.02, -0.0005);
            ProfileOps::update_psin(out, coeffs, ProbeGrid::T, ProbeGrid::T_r, ProbeGrid::T_rr, ProbeGrid::rhos);
            return math::is_finite(out) && check_psin_profile<ProbeGrid>(out, coeffs, 3);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Count>
    constexpr bool F_profile_grid_ok()
    {
        if constexpr (Count > 0)
        {
            Matrix<double, ProbeGrid::nodes.count, 3> out{};
            const auto   coeffs = make_profile_coefficients<Count>(0.015, -0.0004);
            constexpr double scale = 2.25;
            ProfileOps::update_F(
                out,
                coeffs,
                ProbeGrid::T,
                ProbeGrid::T_r,
                ProbeGrid::T_rr,
                ProbeGrid::rhos,
                scale
            );
            return math::is_finite(out) && check_F_profile<ProbeGrid>(out, coeffs, 4, scale);
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool c_profile_grid_ok()
    {
        if constexpr (Order < ProfileOps::c_family_size)
        {
            constexpr size_t count = ProfileOps::template c_count<Order>();
            if constexpr (count > 0)
            {
                constexpr size_t power = ProfileOps::template fourier_power<Order>();

                Matrix<double, ProbeGrid::nodes.count, 3> out{};
                const auto   coeffs = make_profile_coefficients<count>(0.07, 0.001);
                constexpr double offset = 0.25;
                ProfileOps::template update_c<Order>(
                    out,
                    coeffs,
                    ProbeGrid::T,
                    ProbeGrid::T_r,
                    ProbeGrid::T_rr,
                    ProbeGrid::rhos,
                    offset
                );
                return math::is_finite(out) && check_fourier_profile<power, ProbeGrid>(out, coeffs, 5, offset);
            }
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool s_profile_grid_ok()
    {
        static_assert(Order > 0, "s profile checks start at s1");

        if constexpr (Order <= ProfileOps::s_family_size)
        {
            constexpr size_t count = ProfileOps::template s_count<Order>();
            if constexpr (count > 0)
            {
                constexpr size_t power = ProfileOps::template fourier_power<Order>();

                Matrix<double, ProbeGrid::nodes.count, 3> out{};
                const auto   coeffs = make_profile_coefficients<count>(-0.06, 0.0015);
                constexpr double offset = -0.15;
                ProfileOps::template update_s<Order>(
                    out,
                    coeffs,
                    ProbeGrid::T,
                    ProbeGrid::T_r,
                    ProbeGrid::T_rr,
                    ProbeGrid::rhos,
                    offset
                );
                return math::is_finite(out) && check_fourier_profile<power, ProbeGrid>(out, coeffs, 6, offset);
            }
        }
        return true;
    }

    template <typename ProfileOps, typename ProbeGrid, size_t Order>
    constexpr bool highest_s_profile_grid_ok()
    {
        if constexpr (Order > 0)
            return s_profile_grid_ok<ProfileOps, ProbeGrid, Order>();
        else
            return true;
    }

    constexpr bool profiles_grid_constexpr_ok()
    {
        constexpr bool highest_c_ok = [] {
            if constexpr (Topology::C_max < ProbeProfiles::c_family_size)
                return c_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::C_max>();
            else
                return true;
        }();
        constexpr bool highest_s_ok = highest_s_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::S_max>();

        return h_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::h_count>() &&
               v_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::v_count>() &&
               kappa_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::kappa_count>() &&
               psin_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::psin_count>() &&
               F_profile_grid_ok<ProbeProfiles, ProbeGrid, Topology::F_count>() &&
               c_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() &&
               s_profile_grid_ok<ProbeProfiles, ProbeGrid, 1>() && highest_c_ok && highest_s_ok;
    }

    constexpr bool runtime_profiles_constexpr_ok()
    {
        using Shape   = MixedProfileShape;
        using Runtime = MixedRuntimeProfiles;

        constexpr size_t h_id  = Shape::h_profile_id;
        constexpr size_t c0_id = Shape::c_profile_id<0>();
        constexpr size_t c1_id = Shape::c_profile_id<1>();
        constexpr size_t c2_id = Shape::c_profile_id<2>();
        constexpr size_t s2_id = Shape::s_profile_id<2>();

        const auto h_coeffs  = make_profile_coefficients<2>(0.12, -0.003);
        const auto c0_coeffs = make_profile_coefficients<1>(0.07, 0.001);
        const auto c1_coeffs = make_profile_coefficients<2>(0.08, 0.001);
        const auto s2_coeffs = make_profile_coefficients<3>(-0.06, 0.0015);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[c0_id] = 0.25;
        params.offsets[c1_id] = 0.35;
        params.offsets[c2_id] = 4.5;
        params.offsets[s2_id] = -0.15;
        params.scales[c2_id]  = 2.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, h_id>(x, h_coeffs);
        write_profile_coefficients<Shape, c0_id>(x, c0_coeffs);
        write_profile_coefficients<Shape, c1_id>(x, c1_coeffs);
        write_profile_coefficients<Shape, s2_id>(x, s2_coeffs);

        Runtime runtime{};
        runtime.refresh_fixed(params);
        runtime.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        return check_enveloped_profile<RuntimeProbeGrid>(runtime.template profile_matrix<h_id>(), h_coeffs, 0) &&
               check_fourier_profile<0, RuntimeProbeGrid>(
                   runtime.template profile_matrix<c0_id>(),
                   c0_coeffs,
                   5,
                   params.offsets[c0_id]
               ) &&
               check_fourier_profile<1, RuntimeProbeGrid>(
                   runtime.template profile_matrix<c1_id>(),
                   c1_coeffs,
                   5,
                   params.offsets[c1_id]
               ) &&
               check_fourier_profile<2, RuntimeProbeGrid>(
                   runtime.template profile_matrix<s2_id>(),
                   s2_coeffs,
                   6,
                   params.offsets[s2_id]
               ) &&
               close(runtime.template profile_field<c2_id>(0, 0), 9.0) &&
               close(runtime.template profile_field<c2_id>(0, 1), 0.0) &&
               close(runtime.template profile_field<c2_id>(0, 2), 0.0) &&
               close(runtime.template c_family_field<2>(0, 0), 9.0) &&
               close(runtime.template s_family_field<1>(0, 0), 0.0) &&
               close(runtime.template s_family_field<2>(0, 0), runtime.template profile_field<s2_id>(0, 0));
    }

    constexpr bool runtime_profile_semantics_constexpr_ok()
    {
        using Shape   = RuntimeSemanticShape;
        using Runtime = RuntimeSemanticProfiles;

        constexpr size_t c0_id = Shape::c_profile_id<0>();
        constexpr size_t c4_id = Shape::c_profile_id<4>();
        constexpr size_t s4_id = Shape::s_profile_id<4>();
        constexpr size_t F_id  = Shape::F_profile_id;

        const auto c0_coeffs = make_profile_coefficients<1>(0.04, 0.0);
        const auto c4_coeffs = make_profile_coefficients<2>(0.05, 0.002);
        const auto s4_coeffs = make_profile_coefficients<2>(-0.03, 0.001);
        const auto F_coeffs  = make_profile_coefficients<2>(0.015, -0.0004);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[c0_id] = 0.2;
        params.offsets[c4_id] = 0.3;
        params.offsets[s4_id] = -0.1;
        params.scales[F_id]   = 2.25;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, c0_id>(x, c0_coeffs);
        write_profile_coefficients<Shape, c4_id>(x, c4_coeffs);
        write_profile_coefficients<Shape, s4_id>(x, s4_coeffs);
        write_profile_coefficients<Shape, F_id>(x, F_coeffs);

        Runtime runtime{};
        runtime.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        return check_fourier_profile<0, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<c0_id>(),
                   c0_coeffs,
                   0,
                   params.offsets[c0_id]
               ) &&
               check_fourier_profile<2, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<c4_id>(),
                   c4_coeffs,
                   1,
                   params.offsets[c4_id]
               ) &&
               check_fourier_profile<2, RuntimeSemanticGrid>(
                   runtime.template profile_matrix<s4_id>(),
                   s4_coeffs,
                   2,
                   params.offsets[s4_id]
               ) &&
               check_F_profile<RuntimeSemanticGrid>(
                   runtime.template profile_matrix<F_id>(),
                   F_coeffs,
                   3,
                   params.scales[F_id]
               ) &&
               close(runtime.template c_family_field<4>(0, 0), runtime.template profile_field<c4_id>(0, 0)) &&
               close(runtime.template s_family_field<4>(0, 0), runtime.template profile_field<s4_id>(0, 0));
    }

    constexpr bool geometry_circular_constexpr_ok()
    {
        using Shape   = CircularGeometryShape;
        using Grid    = CircularGeometryGrid;
        using Runtime = CircularGeometryProfiles;
        using Geometry = CircularGeometryRuntime;

        constexpr double a  = 0.42;
        constexpr double R0 = 1.8;
        constexpr double Z0 = -0.25;
        constexpr double ka = 1.55;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::kappa_profile_id] = ka;
        params.scales[Shape::kappa_profile_id]  = 1.0;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;
        params.scales[Shape::c_profile_id<0>()]  = 1.0;

        Runtime profiles{};
        profiles.refresh_fixed(params);

        Geometry geometry{};
        geometry.update(a, R0, Z0, profiles);

        if (!math::is_finite(geometry.surface_fields) || !math::is_finite(geometry.radial_fields))
            return false;

        for (size_t i = 0; i < Grid::radial_nodes; ++i)
        {
            const double rho_i      = Grid::nodes[i];
            const double expected_J = a * a * rho_i * ka;
            const double expected_S = 2.0 * grid::detail::pi * expected_J;
            const double expected_V = 4.0 * grid::detail::pi * grid::detail::pi * expected_J * R0;
            constexpr double geometry_smoke_tol = 1.0e-9;

            if (!close(geometry.radial_field(radial_S_r, i), expected_S, geometry_smoke_tol) ||
                !close(geometry.radial_field(radial_V_r, i), expected_V, geometry_smoke_tol))
                return false;
            if (geometry.radial_field(radial_Kn, i) <= 0.0 || geometry.radial_field(radial_Ln_r, i) <= 0.0)
                return false;

            for (size_t j = 0; j < Grid::theta_rows; ++j)
            {
                const double sin_t      = Grid::sin_mtheta(1, j);
                const double cos_t      = Grid::cos_mtheta(1, j);
                const double expected_R = R0 + a * rho_i * cos_t;

                if (!close(geometry.surface_field(surface_sin_tb, i, j), sin_t, geometry_smoke_tol) ||
                    !close(geometry.surface_field(surface_R, i, j), expected_R, geometry_smoke_tol) ||
                    !close(geometry.surface_field(surface_R_t, i, j), -a * rho_i * sin_t, geometry_smoke_tol) ||
                    !close(geometry.surface_field(surface_Z_t, i, j), -a * rho_i * ka * cos_t, geometry_smoke_tol) ||
                    !close(geometry.surface_field(surface_J, i, j), expected_J, geometry_smoke_tol) ||
                    !close(geometry.surface_field(surface_JdivR, i, j), expected_J / expected_R, geometry_smoke_tol))
                    return false;
            }
        }

        return true;
    }

    constexpr bool source_materialization_constexpr_ok()
    {
        using Shape   = SourceMaterializationShape;
        using Grid    = SourceMaterializationGrid;
        using Runtime = SourceMaterializationProfiles;
        using Source  = SourceMaterializationRuntime;

        constexpr size_t psin_id = Shape::psin_profile_id;
        const auto       psin_coeffs = make_profile_coefficients<3>(0.01, -0.0002);

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::h_profile_id] = 0.0;
        params.offsets[Shape::v_profile_id] = 0.0;
        params.offsets[Shape::kappa_profile_id] = 1.45;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, psin_id>(x, psin_coeffs);

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Source source{};
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{-1.0, -0.875, -0.75, -0.625, -0.5};
        source.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );

        source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0));

        if (!close(source.source_target_root_fields(root_psin, 0), 0.0, 0.0) ||
            !close(source.source_target_root_fields(root_psin, Grid::radial_nodes - 1), 1.0, 0.0))
            return false;

        for (size_t i = 0; i < Grid::radial_nodes; ++i)
        {
            const double q = source.source_psin_query[i];
            if (!close(source.source_parameter_query[i], q) ||
                !close(source.source_target_root_fields(root_psin, i), q) ||
                source.source_target_root_fields(root_psin_r, i) <= 0.0 ||
                !math::is_finite(source.source_target_root_fields(root_psin_rr, i)))
                return false;

            const double expected_heat = 2.0 + 3.0 * q;
            const double expected_current = -1.0 + 0.5 * q;
            if (!close(source.materialized_heat_input[i], expected_heat, 1.0e-10) ||
                !close(source.materialized_current_input[i], expected_current, 1.0e-10))
                return false;
        }

        return true;
    }

    constexpr bool pf_source_constexpr_ok()
    {
        using Shape   = SourceMaterializationShape;
        using Grid    = SourceMaterializationGrid;
        using Runtime = SourceMaterializationProfiles;
        using Source  = SourceMaterializationRuntime;
        using Geometry = GeometryRuntime<Grid>;

        constexpr size_t psin_id = Shape::psin_profile_id;
        const auto       psin_coeffs = make_profile_coefficients<3>(0.01, -0.0002);
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        constexpr double a  = 0.42;
        constexpr double R0 = 1.8;
        constexpr double Z0 = -0.25;
        constexpr double Ip = 0.75;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[Shape::h_profile_id] = 0.0;
        params.offsets[Shape::v_profile_id] = 0.0;
        params.offsets[Shape::kappa_profile_id] = 1.45;
        params.offsets[Shape::c_profile_id<0>()] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, psin_id>(x, psin_coeffs);

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Geometry geometry{};
        geometry.update(a, R0, Z0, profiles);

        auto make_source = [&heat, &current, &profiles](Source& source) constexpr {
            source.set_uniform_sources(
                std::span<const double, heat.size()>{heat.data(), heat.size()},
                std::span<const double, current.size()>{current.data(), current.size()}
            );
            source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0));
        };

        Source ip_source{};
        make_source(ip_source);
        ip_source.update_pf_psin_uniform_ip(geometry, Ip, source::axis_fix_count<Grid>(0.0));
        for (size_t i = 0; i < Grid::radial_nodes; ++i)
            if (!close(ip_source.Pn_psin[i], ip_source.materialized_heat_input[i]) ||
                !close(ip_source.FFn_psin[i], ip_source.materialized_current_input[i]))
                return false;

        return math::is_finite(ip_source.alpha1) && math::is_finite(ip_source.alpha2);
    }

    constexpr bool residual_pack_constexpr_ok()
    {
        using Shape    = ResidualProbeShape;
        using Grid     = ResidualProbeGrid;
        using Runtime  = ResidualProbeProfiles;
        using Source   = ResidualProbeSource;
        using Geometry = ResidualProbeGeometry;
        using Residual = ResidualProbeRuntime;

        constexpr size_t h_id     = Shape::h_profile_id;
        constexpr size_t v_id     = Shape::v_profile_id;
        constexpr size_t k_id     = Shape::kappa_profile_id;
        constexpr size_t c0_id    = Shape::c_profile_id<0>();
        constexpr size_t c1_id    = Shape::c_profile_id<1>();
        constexpr size_t s1_id    = Shape::s_profile_id<1>();
        constexpr size_t psin_id  = Shape::psin_profile_id;

        profiles::ProfileRuntimeParams<Shape> params{};
        params.offsets[k_id]  = 1.45;
        params.offsets[c0_id] = 0.0;
        params.offsets[c1_id] = 0.0;
        params.offsets[s1_id] = 0.0;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, h_id>(x, make_profile_coefficients<2>(0.020, -0.0010));
        write_profile_coefficients<Shape, v_id>(x, make_profile_coefficients<2>(0.010, 0.0010));
        write_profile_coefficients<Shape, k_id>(x, make_profile_coefficients<2>(0.015, -0.0007));
        write_profile_coefficients<Shape, c0_id>(x, make_profile_coefficients<2>(0.004, 0.0002));
        write_profile_coefficients<Shape, c1_id>(x, make_profile_coefficients<2>(0.003, 0.0002));
        write_profile_coefficients<Shape, s1_id>(x, make_profile_coefficients<2>(-0.002, 0.0001));
        write_profile_coefficients<Shape, psin_id>(x, make_profile_coefficients<2>(0.010, -0.0002));

        Runtime profiles{};
        profiles.refresh_fixed(params);
        profiles.refresh_active(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, params);

        Geometry geometry{};
        geometry.update(0.42, 1.8, -0.25, profiles);

        Source source{};
        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        source.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );
        source.materialize_profile_owned_psin(profiles, source::axis_fix_count<Grid>(0.0));
        source.update_pf_psin_uniform_ip(geometry, 0.75, source::axis_fix_count<Grid>(0.0));

        Residual residual{};
        residual.update_compact(source, geometry);
        const auto packed = residual.pack(0.42, 1.8, 2.1);

        if (!math::is_finite(residual.surface_fields) || !math::is_finite(packed))
            return false;

        double norm1 = 0.0;
        for (size_t i = 0; i < Shape::x_size; ++i)
            norm1 += math::abs(packed[i]);
        return norm1 > 1.0e-12;
    }

    constexpr bool pf_operator_constexpr_ok()
    {
        using Shape    = ResidualProbeShape;
        using Operator = ResidualProbeOperator;

        constexpr size_t h_id     = Shape::h_profile_id;
        constexpr size_t v_id     = Shape::v_profile_id;
        constexpr size_t k_id     = Shape::kappa_profile_id;
        constexpr size_t c0_id    = Shape::c_profile_id<0>();
        constexpr size_t c1_id    = Shape::c_profile_id<1>();
        constexpr size_t s1_id    = Shape::s_profile_id<1>();
        constexpr size_t psin_id  = Shape::psin_profile_id;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, h_id>(x, make_profile_coefficients<2>(0.020, -0.0010));
        write_profile_coefficients<Shape, v_id>(x, make_profile_coefficients<2>(0.010, 0.0010));
        write_profile_coefficients<Shape, k_id>(x, make_profile_coefficients<2>(0.015, -0.0007));
        write_profile_coefficients<Shape, c0_id>(x, make_profile_coefficients<2>(0.004, 0.0002));
        write_profile_coefficients<Shape, c1_id>(x, make_profile_coefficients<2>(0.003, 0.0002));
        write_profile_coefficients<Shape, s1_id>(x, make_profile_coefficients<2>(-0.002, 0.0001));
        write_profile_coefficients<Shape, psin_id>(x, make_profile_coefficients<2>(0.010, -0.0002));

        Operator op{};
        Operator::RuntimeParams params{};
        params.a = 0.42;
        params.R0 = 1.8;
        params.Z0 = -0.25;
        params.B0 = 2.1;
        params.Ip = 0.75;
        params.fix_rho = 0.0;
        params.profile_params.offsets[k_id] = 1.45;
        params.profile_params.offsets[c0_id] = 0.0;
        params.profile_params.offsets[c1_id] = 0.0;
        params.profile_params.offsets[s1_id] = 0.0;
        op.set_runtime_params(params);

        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        op.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );

        typename Operator::PackedVector packed{};
        op.evaluate(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, packed);

        double norm1 = 0.0;
        for (size_t i = 0; i < Shape::x_size; ++i)
            norm1 += math::abs(packed[i]);
        return math::is_finite(packed) && norm1 > 1.0e-12;
    }

    constexpr bool pf_operator_plan_invalidation_ok()
    {
        using Shape    = SourceMaterializationShape;
        using Grid     = SourceMaterializationGrid;
        using Operator = PfPsinUniformIpOperator<Shape, Grid, UniformSourceShape<5>>;

        constexpr size_t h_id    = Shape::h_profile_id;
        constexpr size_t v_id    = Shape::v_profile_id;
        constexpr size_t k_id    = Shape::kappa_profile_id;
        constexpr size_t c0_id   = Shape::c_profile_id<0>();
        constexpr size_t psin_id = Shape::psin_profile_id;

        Vector<double, Shape::x_size> x{};
        write_profile_coefficients<Shape, psin_id>(x, make_profile_coefficients<3>(0.010, -0.0002));

        Operator op{};
        Operator::RuntimeParams params{};
        params.a = 0.42;
        params.R0 = 1.8;
        params.Z0 = -0.25;
        params.B0 = 2.1;
        params.Ip = 0.75;
        params.fix_rho = 0.0;
        params.profile_params.offsets[h_id] = 0.02;
        params.profile_params.offsets[v_id] = -0.01;
        params.profile_params.offsets[k_id] = 1.45;
        params.profile_params.offsets[c0_id] = 0.0;
        op.set_runtime_params(params);

        constexpr std::array<double, 5> heat{2.0, 2.75, 3.5, 4.25, 5.0};
        constexpr std::array<double, 5> current{0.5, 0.625, 0.75, 0.875, 1.0};
        op.set_uniform_sources(
            std::span<const double, heat.size()>{heat.data(), heat.size()},
            std::span<const double, current.size()>{current.data(), current.size()}
        );

        typename Operator::PackedVector packed{};
        op.evaluate(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, packed);
        if (!op.plan.prepared || op.plan.n_axis_fix != source::axis_fix_count<Grid>(0.0))
            return false;

        auto next = op.runtime_params();
        next.fix_rho = 0.5;
        next.profile_params.offsets[k_id] = 1.80;
        op.set_runtime_params(next);
        if (op.plan.prepared)
            return false;

        op.evaluate(std::span<const double, Shape::x_size>{x.data(), Shape::x_size}, packed);
        return op.plan.prepared && op.plan.n_axis_fix == source::axis_fix_count<Grid>(0.5) &&
               close(op.plan.fixed_profiles.profile_field<k_id>(0, 0), 1.80);
    }

    static_assert(linalg_constexpr_ok());
    static_assert(tensor_math_constexpr_ok());
    static_assert(finite_semantics_constexpr_ok());
    static_assert(grid_constexpr_ok());
    static_assert(profiles_grid_constexpr_ok());
    static_assert(runtime_profiles_constexpr_ok());
    static_assert(runtime_profile_semantics_constexpr_ok());
    static_assert(source_materialization_constexpr_ok());

    int root_residual(void*, int n, const double* x, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != 1)
            return 0;
        fvec[0] = x[0] * x[0] - 9.0;
        return 0;
    }

    bool runtime_library_ok(nlohmann::json& report)
    {
        const auto svd_solution = solve<GolubReinsch>(dense_matrix, dense_rhs);

        double        root_x[1] = {4.0};
        double        root_f[1] = {0.0};
        constexpr int root_n    = 1;
        constexpr int root_lwa  = root_n * (3 * root_n + 13) / 2;
        double        root_work[root_lwa];
        const int     root_info = hybrd1(root_residual, nullptr, root_n, root_x, root_f, 1.0e-10, root_work, root_lwa);

        double           lapack_a[4] = {3.0, 1.0, 1.0, 2.0};
        double           lapack_b[2] = {9.0, 8.0};
        lapack_int       ipiv[2];
        const lapack_int lapack_info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, 2, 1, lapack_a, 2, ipiv, lapack_b, 1);

        report["runtime"] = {
            {"gcem_sqrt_25", gcem::sqrt(25.0)},
            {"golub_reinsch", {svd_solution[0], svd_solution[1]}},
            {"cminpack", {{"info", root_info}, {"x", root_x[0]}, {"f", root_f[0]}}},
            {"lapacke", {{"info", static_cast<int>(lapack_info)}, {"solution", {lapack_b[0], lapack_b[1]}}}},
        };

        return close(svd_solution[0], 2.0) && close(svd_solution[1], 3.0) && root_info > 0 &&
               close(root_x[0], 3.0, 1.0e-8) && lapack_info == 0 && close(lapack_b[0], 2.0) && close(lapack_b[1], 3.0);
    }
} // namespace

int run(int, char**)
{
    nlohmann::json report;

    report["constexpr"] = {
        {"linalg", linalg_constexpr_ok()},
        {"tensor_math", tensor_math_constexpr_ok()},
        {"finite_semantics", finite_semantics_constexpr_ok()},
        {"grid", grid_constexpr_ok()},
        {"profiles_grid", profiles_grid_constexpr_ok()},
        {"runtime_profiles", runtime_profiles_constexpr_ok()},
        {"runtime_profile_semantics", runtime_profile_semantics_constexpr_ok()},
        {"geometry_circular", geometry_circular_constexpr_ok()},
        {"source_materialization", source_materialization_constexpr_ok()},
        {"pf_source", pf_source_constexpr_ok()},
        {"residual_pack", residual_pack_constexpr_ok()},
        {"pf_operator", pf_operator_constexpr_ok()},
        {"pf_operator_plan_invalidation", pf_operator_plan_invalidation_ok()},
    };
    report["quadrature"] = {
        {"chebyshev_moment_error_n16_degree7", max_moment_error<Chebyshev, 16>(7)},
        {"legendre_moment_error_n16_degree15", max_moment_error<Legendre, 16>(15)},
        {"lobatto_moment_error_n16_degree13", max_moment_error<Lobatto, 16>(13)},
        {"radau_moment_error_n16_degree14", max_moment_error<Radau, 16>(14)},
    };
    report["calculus"] = {
        {"spectral_legendre_diff_error", max_differentiator_error<Spectral, Legendre, 8>(3)},
        {"spectral_legendre_acc_error", max_accumulator_error<Spectral, Legendre, 8>(2)},
        {"cfd33_lobatto_diff_error", max_differentiator_error<CFD33, Lobatto, 8>(2)},
        {"cfd35_lobatto_diff_error", max_differentiator_error<CFD35, Lobatto, 8>(2)},
        {"cfd55_lobatto_diff_error", max_differentiator_error<CFD55, Lobatto, 8>(2)},
        {"cfd33_lobatto_acc_error", max_accumulator_error<CFD33, Lobatto, 8>(1)},
        {"cfd35_lobatto_acc_error", max_accumulator_error<CFD35, Lobatto, 8>(1)},
        {"cfd55_lobatto_acc_error", max_accumulator_error<CFD55, Lobatto, 8>(1)},
    };

    const bool ok = linalg_constexpr_ok() && tensor_math_constexpr_ok() && finite_semantics_constexpr_ok() &&
                    grid_constexpr_ok() && profiles_grid_constexpr_ok() && runtime_profiles_constexpr_ok() &&
                    runtime_profile_semantics_constexpr_ok() && geometry_circular_constexpr_ok() &&
                    source_materialization_constexpr_ok() && pf_source_constexpr_ok() &&
                    residual_pack_constexpr_ok() && pf_operator_constexpr_ok() &&
                    pf_operator_plan_invalidation_ok() &&
                    runtime_library_ok(report);

    std::cout << report.dump(2) << '\n';
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

} // namespace veqlib_temp_validation_cli

