#pragma once

// Topology-derived fixed types and default case/result records for production kernels.

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string>

#include "abi_enums.h"
#include "config.h"
#include "grid.h"
#include "operators.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace cxx_kernel_api
{
    namespace
    {
        using grid::Legendre;
        using grid::Spectral;
        using config::Topology;
        using operators::SourceOperator;
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
                  typename CalculusScheme,
                  size_t BoundaryMmax = inferred_M_max<CFamilyCounts, SFamilyCounts>(),
                  bool   LayoutProfileFirst = false>
        struct SourceCompiledTopology
        {
            static constexpr size_t L_max =
                inferred_L_max<HCount, VCount, KappaCount, PsinCount, FCount, CFamilyCounts, SFamilyCounts>();
            static constexpr size_t M_max = BoundaryMmax;
            static constexpr size_t K_max = inferred_K_max<M_max>();
            static_assert(M_max >= inferred_M_max<CFamilyCounts, SFamilyCounts>());

            using Shape = profiles::OptimizedProfileShapeFromCountsWithMmaxT<L_max,
                                                                             K_max,
                                                                             M_max,
                                                                             HCount,
                                                                             VCount,
                                                                             KappaCount,
                                                                             PsinCount,
                                                                             FCount,
                                                                             CFamilyCounts,
                                                                             SFamilyCounts,
                                                                             LayoutProfileFirst>;
            using Grid = grid::Grid<Nr, Nt, Shape::L_max, Shape::M_max, Shape::K_max, QuadratureScheme, CalculusScheme>;
            using Source   = source::SampledSourceShape<SourceSamples>;
            using Operator = SourceOperator<Shape,
                                                     Grid,
                                                     Source,
                                                     Topology::source_route_code,
                                                     Topology::source_constraint_code,
                                                     Topology::source_coordinate_code,
                                                     Topology::source_nodes_code,
                                                     Topology::source_active_family_code,
                                                     Topology::source_parameterization_code>;
        };

        constexpr auto kernel_c_counts = Topology::c_family_counts;
        constexpr auto kernel_s_counts = Topology::s_family_counts;

        using CompiledTopology = SourceCompiledTopology<Topology::Nr,
                                                       Topology::Nt,
                                                       Topology::source_sample_count,
                                                       Topology::h_count,
                                                       Topology::v_count,
                                                       Topology::kappa_count,
                                                       Topology::psin_count,
                                                       Topology::F_count,
                                                       kernel_c_counts,
                                                       kernel_s_counts,
                                                       Legendre,
                                                       Spectral,
                                                       Topology::M_max,
                                                       Topology::layout_profile_first>;
        using CompiledShape    = CompiledTopology::Shape;
        using CompiledGrid     = CompiledTopology::Grid;
        using CompiledSource   = CompiledTopology::Source;
        using CompiledOperator = CompiledTopology::Operator;
        using PackedVector   = CompiledOperator::PackedVector;

        static_assert(CompiledShape::L_max == Topology::L_max);
        static_assert(CompiledShape::M_max == Topology::M_max);
        static_assert(CompiledShape::K_max == Topology::K_max);

        constexpr double default_max_residual              = 1.0e-6;
        constexpr int    default_requested_max_evaluations = 1000;
        constexpr int    default_maxfev   = default_requested_max_evaluations > 500 ? default_requested_max_evaluations : 500;
        constexpr double default_hybr_eps = 1.0e-6;
        constexpr double default_hybr_factor              = 1.0;
        constexpr double default_lm_eps                   = 0.0;
        constexpr double default_lm_factor                = 100.0;
        constexpr double default_accepted_residual_factor = 10.0;
        constexpr double default_accepted_residual_floor  = 1.0e-5;
        constexpr double default_x_scale_floor            = 1.0e-2;
        constexpr double default_core_profile_prior       = 1.5e-1;
        constexpr double default_fourier_profile_prior    = 5.0e-2;
        constexpr double default_F_profile_prior          = 2.5e-1;
        constexpr double default_kappa_profile_prior      = 1.0;

        enum class SolverKind
        {
            LevenbergMarquardt,
            NewtonKrylov,
            NewtonRaphson,
            Powell,
        };


        struct RuntimeCase
        {
            std::string                                    case_name = "PF_psin_uniform_Ip";
            std::array<double, CompiledSource::sample_count> heat{};
            std::array<double, CompiledSource::sample_count> current{};
            std::array<double, CompiledShape::x_size>        x0{};
            std::array<double, CompiledShape::x_size>        x_scale{};
            std::array<double, CompiledShape::x_size>        residual_scale{};
            double                                         a         = 1.05 / 1.85;
            double                                         R0        = 1.05;
            double                                         Z0        = 0.0;
            double                                         B0        = 3.0;
            double                                         ka        = 2.2;
            double                                         c0_offset = 0.0;
            double                                         s1_offset = 0.52359877559829887308;
            std::array<double, CompiledShape::M_max + 1>     c_offsets{};
            std::array<double, CompiledShape::M_max + 1>     s_offsets{};
            double                                         p0                       = 0.0;
            double                                         Ip                       = 3.7699111867885415;
            double                                         beta                     = std::numeric_limits<double>::quiet_NaN();
            double                                         max_residual             = default_max_residual;
            double                                         accepted_residual_factor = default_accepted_residual_factor;
            double                                         accepted_residual_floor  = default_accepted_residual_floor;
            double                                         residual_normalization_floor              = 1.0;
            double                                         residual_normalization_max_ratio          = 1.0e6;
            double                                         residual_normalization_huber_tau          = 3.0;
            double                                         residual_normalization_probe_step         = 1.0e-6;
            double                                         residual_normalization_sensitivity_lambda = 0.5;
            int                                            max_evaluations = default_requested_max_evaluations;
            int                                            residual_normalization_probe_count = 4;
            int                                            repeat                             = 10;
            int                                            warmup                             = 1;
            SolverKind                                     solver                             = SolverKind::Powell;
            int                                            initial_policy_code                = InitialPolicyCold;
            int                                            continue_policy_code               = ContinuePolicyWarm;
            int                                            residual_normalization_code = ResidualNormalizationFast;
        };

        struct SolveResult
        {
            std::array<double, CompiledShape::x_size> x{};
            PackedVector                            raw{};
            PackedVector                            scaled{};
            std::array<double, 2>                   alpha{};
            double                                  raw_norm                       = 0.0;
            double                                  scaled_norm                    = 0.0;
            int                                     info                           = 0;
            int                                     nfev                           = 0;
            int                                     njev                           = 0;
            int                                     callbacks                      = 0;
            int                                     solver_nfev                    = 0;
            int                                     jacobian_component_evaluations = 0;
            int                                     jvp_evaluations                = 0;
            int                                     linear_iterations              = 0;
            int                                     initial_residual_evaluations   = 0;
            int                                     certification_residual_evaluations = 0;
            int                                     total_raw_residual_evaluations = 0;
            double                                  residual_callback_ms           = 0.0;
            double                                  residual_kernel_ms             = 0.0;
            double                                  residual_scale_ms              = 0.0;
            double                                  final_residual_ms              = 0.0;
            double                                  jacobian_callback_ms           = 0.0;
            double                                  jvp_callback_ms                = 0.0;
            double                                  linear_solve_ms                = 0.0;
            double                                  cert_threshold                 = 0.0;
            double                                  initial_raw_norm               = 0.0;
            double                                  fast_path_raw_norm             = 0.0;
            std::string                             accepted_by                    = "solver";
            std::string                             fast_path                      = "none";
            std::string                             fallback_reason                = "";
            bool                                    fallback_used                  = false;
            bool                                    accepted                       = false;
        };

        double norm2(std::span<const double, CompiledShape::x_size> values) noexcept
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

        constexpr double default_acceptance_threshold() noexcept
        {
            constexpr double scaled = default_max_residual * default_accepted_residual_factor;
            return scaled > default_accepted_residual_floor ? scaled : default_accepted_residual_floor;
        }

        double acceptance_threshold(const RuntimeCase& input) noexcept
        {
            const double scaled = input.max_residual * input.accepted_residual_factor;
            return scaled > input.accepted_residual_floor ? scaled : input.accepted_residual_floor;
        }

        int max_solver_evaluations(const RuntimeCase& input) noexcept
        {
            return input.max_evaluations > 500 ? input.max_evaluations : 500;
        }

        constexpr std::array<double, 51> default_scaled_heat_values = {
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

        constexpr std::array<double, 51> default_scaled_current_values = {
            -0.2884247371510828,  -0.28903009704030685,  -0.28957011555052764,   -0.29005075790401863,
            -0.2904781886031161,  -0.2908575435658439,   -0.2911919683745232,    -0.2914838716684553,
            -0.2917344967330943,  -0.2919437907627883,   -0.29211086667346203,   -0.2922346427093772,
            -0.2923145136340828,  -0.29232232828599186,  -0.2923070658458569,    -0.2922075036101631,
            -0.29204221633819877, -0.2918026604476458,   -0.2914746207679046,    -0.29104564954616613,
            -0.2905098257293566,  -0.28984851066007944,  -0.2890431993822433,    -0.2880834160319397,
            -0.2869422749878048,  -0.28559761849429993,  -0.28403023480001177,   -0.2822015304104101,
            -0.2800887239343856,  -0.27765576104313033,  -0.274853815959196,     -0.27165492157456717,
            -0.26800248440388824, -0.2638385916755866,   -0.25912135836327627,   -0.25377334278513203,
            -0.2477205935589761,  -0.2409002037441097,   -0.23321180576616163,   -0.22454617787318162,
            -0.2148028033443947,  -0.20384443014756062,  -0.191504616045878,     -0.17758595047988296,
            -0.1618458187382766,  -0.143974507066849,    -0.12356213030456985,   -0.10004932271682981,
            -0.07266570051589127, -0.040268147498330784, -0.0011074929612556953,
        };

        template <typename Values>
        constexpr std::array<double, CompiledSource::sample_count> source_defaults_from(const Values& values) noexcept
        {
            std::array<double, CompiledSource::sample_count> out{};
            for (size_t i = 0; i < CompiledSource::sample_count; ++i)
            {
                const size_t source_index = i < values.size() ? i : values.size() - 1;
                out[i]                    = values[source_index];
            }
            return out;
        }

        constexpr auto default_scaled_heat    = source_defaults_from(default_scaled_heat_values);
        constexpr auto default_scaled_current = source_defaults_from(default_scaled_current_values);

    } // namespace

} // namespace cxx_kernel_api
