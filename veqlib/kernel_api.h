#pragma once

// Production PF/psin/uniform/Ip kernel API used by nanobind bindings.
// C++ validation/benchmark CLI code was removed after parity with the Numba backend was established.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>

#include <cminpack.h>
#ifdef ENABLE_ENZYME
    #include <enzyme/enzyme>
extern int enzyme_dupv;
extern int enzyme_width;
#endif
#include <nlohmann/json.hpp>

#include "config.h"
#include "grid.h"
#include "nonlinear.h"
#include "operators.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_kernel_api
{
    namespace
    {
        using grid::Legendre;
        using grid::Spectral;
        using config::Topology;
        using operators::PfPsinUniformIpOperator;
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
                  size_t BoundaryMmax = inferred_M_max<CFamilyCounts, SFamilyCounts>()>
        struct PfPsinUniformIpTopology
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
                                                                             SFamilyCounts>;
            using Grid = grid::Grid<Nr, Nt, Shape::L_max, Shape::M_max, Shape::K_max, QuadratureScheme, CalculusScheme>;
            using Source   = source::UniformSourceShape<SourceSamples>;
            using Operator = PfPsinUniformIpOperator<Shape, Grid, Source>;
        };

        constexpr auto bench_c_counts = Topology::c_family_counts;
        constexpr auto bench_s_counts = Topology::s_family_counts;

        using BenchTopology = PfPsinUniformIpTopology<Topology::Nr,
                                                      Topology::Nt,
                                                      Topology::source_sample_count,
                                                      Topology::h_count,
                                                      Topology::v_count,
                                                      Topology::kappa_count,
                                                      Topology::psin_count,
                                                      Topology::F_count,
                                                      bench_c_counts,
                                                      bench_s_counts,
                                                      Legendre,
                                                      Spectral,
                                                      Topology::M_max>;
        using BenchShape    = BenchTopology::Shape;
        using BenchGrid     = BenchTopology::Grid;
        using BenchSource   = BenchTopology::Source;
        using BenchOperator = BenchTopology::Operator;
        using PackedVector  = BenchOperator::PackedVector;

        static_assert(BenchShape::L_max == Topology::L_max);
        static_assert(BenchShape::M_max == Topology::M_max);
        static_assert(BenchShape::K_max == Topology::K_max);

        constexpr double veqpy_max_residual              = 1.0e-6;
        constexpr int    veqpy_requested_max_evaluations = 1000;
        constexpr int    veqpy_maxfev   = veqpy_requested_max_evaluations > 500 ? veqpy_requested_max_evaluations : 500;
        constexpr double veqpy_hybr_eps = 1.0e-6;
        constexpr double veqpy_hybr_factor              = 1.0;
        constexpr int    veqpy_hybr_mode                = 1;
        constexpr int    veqpy_hybr_nprint              = 0;
        constexpr double veqpy_lm_eps                   = 0.0;
        constexpr double veqpy_lm_factor                = 100.0;
        constexpr int    veqpy_lm_mode                  = 2;
        constexpr int    veqpy_lm_nprint                = 0;
        constexpr double veqpy_accepted_residual_factor = 10.0;
        constexpr double veqpy_accepted_residual_floor  = 1.0e-5;
        constexpr double veqpy_x_scale_floor            = 1.0e-2;
        constexpr double veqpy_core_profile_prior       = 1.5e-1;
        constexpr double veqpy_fourier_profile_prior    = 5.0e-2;
        constexpr double veqpy_F_profile_prior          = 2.5e-1;
        constexpr double veqpy_kappa_profile_prior      = 1.0;

        enum class SolverKind
        {
            LevenbergMarquardt,
            Newton,
            NewtonKrylov,
            NewtonRaphson,
            Powell,
        };

        enum RuntimeSolverMethodCode : int
        {
            SolverMethodPowell             = 1,
            SolverMethodLevenbergMarquardt = 2,
            SolverMethodNewton             = 3,
            SolverMethodNewtonKrylov       = 4,
            SolverMethodNewtonRaphson      = 5,
        };

        enum InitialPolicyCode : int
        {
            InitialPolicyColdZeros     = 1,
            InitialPolicyColdGeometric = 2,
            InitialPolicyCold          = 3,
            InitialPolicyWarmClone     = 4,
        };

        enum ResidualNormalizationCode : int
        {
            ResidualNormalizationNone     = 0,
            ResidualNormalizationFast     = 1,
            ResidualNormalizationBalanced = 2,
            ResidualNormalizationSafe     = 3,
        };

        struct CaseInput
        {
            std::string                                   case_name = "PF_psin_uniform_Ip";
            std::array<double, BenchSource::sample_count> heat{};
            std::array<double, BenchSource::sample_count> current{};
            std::array<double, BenchShape::x_size>        x0{};
            std::array<double, BenchShape::x_size>        x_scale{};
            std::array<double, BenchShape::x_size>        residual_scale{};
            double                                        a         = 1.05 / 1.85;
            double                                        R0        = 1.05;
            double                                        Z0        = 0.0;
            double                                        B0        = 3.0;
            double                                        ka        = 2.2;
            double                                        c0_offset = 0.0;
            double                                        s1_offset = 0.52359877559829887308;
            std::array<double, BenchShape::M_max + 1>     c_offsets{};
            std::array<double, BenchShape::M_max + 1>     s_offsets{};
            double                                        Ip                           = 3.7699111867885415;
            double                                        fix_rho                      = 0.05;
            double                                        max_residual                 = veqpy_max_residual;
            double                                        accepted_residual_factor     = veqpy_accepted_residual_factor;
            double                                        accepted_residual_floor      = veqpy_accepted_residual_floor;
            double                                        residual_normalization_floor = 1.0;
            double                                        residual_normalization_max_ratio          = 1.0e6;
            double                                        residual_normalization_huber_tau          = 3.0;
            double                                        residual_normalization_probe_step         = 1.0e-6;
            double                                        residual_normalization_sensitivity_lambda = 0.5;
            int                                           max_evaluations = veqpy_requested_max_evaluations;
            int                                           residual_normalization_probe_count = 4;
            int                                           repeat                             = 10;
            int                                           warmup                             = 1;
            int                                           enzyme_width                       = 1;
            SolverKind                                    solver                             = SolverKind::Powell;
            int                                           initial_policy_code                = InitialPolicyCold;
            int                                           residual_normalization_code         = ResidualNormalizationFast;
        };

        struct SolveResult
        {
            std::array<double, BenchShape::x_size> x{};
            PackedVector                           raw{};
            PackedVector                           scaled{};
            std::array<double, 2>                  alpha{};
            double                                 raw_norm                       = 0.0;
            double                                 scaled_norm                    = 0.0;
            int                                    info                           = 0;
            int                                    nfev                           = 0;
            int                                    njev                           = 0;
            int                                    callbacks                      = 0;
            int                                    jacobian_component_evaluations = 0;
            int                                    jvp_evaluations                = 0;
            int                                    linear_iterations              = 0;
            double                                 residual_callback_ms           = 0.0;
            double                                 residual_kernel_ms             = 0.0;
            double                                 residual_scale_ms              = 0.0;
            double                                 final_residual_ms              = 0.0;
            double                                 jacobian_callback_ms           = 0.0;
            double                                 jvp_callback_ms                = 0.0;
            double                                 linear_solve_ms                = 0.0;
            bool                                   accepted                       = false;
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

        double acceptance_threshold(const CaseInput& input) noexcept
        {
            const double scaled = input.max_residual * input.accepted_residual_factor;
            return scaled > input.accepted_residual_floor ? scaled : input.accepted_residual_floor;
        }

        int max_solver_evaluations(const CaseInput& input) noexcept
        {
            return input.max_evaluations > 500 ? input.max_evaluations : 500;
        }

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
            for (size_t order = 0; order <= BenchShape::M_max; ++order)
                params.offsets[BenchShape::c_profile_id(order)] = input.c_offsets[order];
            for (size_t order = 1; order <= BenchShape::M_max; ++order)
                params.offsets[BenchShape::s_profile_id(order)] = input.s_offsets[order];
            return params;
        }

        template <typename Shape>
        std::array<double, Shape::x_size>
        build_x_block_scale_vector(const std::array<double, Shape::x_size>&     x_guess,
                                   const profiles::ProfileRuntimeParams<Shape>& profile_params) noexcept
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

        constexpr double residual_scale_tiny() noexcept { return std::numeric_limits<double>::min(); }

        double finite_positive_or(double value, double fallback) noexcept
        {
            return std::isfinite(value) && value > 0.0 ? value : fallback;
        }

        double residual_scale_floor(double floor) noexcept
        {
            return finite_positive_or(floor, 1.0);
        }

        double residual_scale_max_ratio(double max_ratio) noexcept
        {
            return std::isfinite(max_ratio) && max_ratio >= 1.0 ? max_ratio : 1.0;
        }

        double median_sorted_prefix(std::array<double, BenchShape::x_size>& values, size_t count) noexcept
        {
            if (count == 0)
                return 0.0;
            std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
            const size_t mid = count / 2;
            if ((count % 2) == 1)
                return values[mid];
            return 0.5 * (values[mid - 1] + values[mid]);
        }

        double stable_rms_clipped(std::array<double, BenchShape::x_size>& values,
                                  size_t                                 count,
                                  double                                 cutoff) noexcept
        {
            if (count == 0)
                return 0.0;
            double max_abs = 0.0;
            for (size_t i = 0; i < count; ++i)
            {
                const double clipped = std::min(values[i], cutoff);
                if (clipped > max_abs)
                    max_abs = clipped;
                values[i] = clipped;
            }
            if (max_abs == 0.0)
                return 0.0;
            double total = 0.0;
            for (size_t i = 0; i < count; ++i)
            {
                const double scaled = values[i] / max_abs;
                total += scaled * scaled;
            }
            return max_abs * std::sqrt(total / static_cast<double>(count));
        }

        double stable_rms_block(const PackedVector& residual, size_t offset, size_t length) noexcept
        {
            if (length == 0)
                return 0.0;
            double max_abs = 0.0;
            for (size_t i = 0; i < length; ++i)
            {
                const double value = std::abs(residual[offset + i]);
                if (std::isfinite(value) && value > max_abs)
                    max_abs = value;
            }
            if (max_abs == 0.0)
                return 0.0;
            double total = 0.0;
            for (size_t i = 0; i < length; ++i)
            {
                const double value = std::abs(residual[offset + i]);
                if (!std::isfinite(value))
                    continue;
                const double scaled = value / max_abs;
                total += scaled * scaled;
            }
            return max_abs * std::sqrt(total / static_cast<double>(length));
        }

        double robust_rms_block(const PackedVector& residual, size_t offset, size_t length, double huber_tau) noexcept
        {
            std::array<double, BenchShape::x_size> finite{};
            size_t                                 count = 0;
            for (size_t i = 0; i < length; ++i)
            {
                const double value = std::abs(residual[offset + i]);
                if (std::isfinite(value))
                    finite[count++] = value;
            }
            if (count == 0)
                return 0.0;

            std::array<double, BenchShape::x_size> sorted = finite;
            const double                           center = median_sorted_prefix(sorted, count);
            std::array<double, BenchShape::x_size> deviations{};
            for (size_t i = 0; i < count; ++i)
                deviations[i] = std::abs(finite[i] - center);
            const double mad = median_sorted_prefix(deviations, count);
            double       cutoff =
                center + std::max(huber_tau, 0.0) * 1.4826 * mad;
            if (!std::isfinite(cutoff) || cutoff <= 0.0)
                cutoff = center;
            return stable_rms_clipped(finite, count, cutoff);
        }

        double balanced_residual_anchor(std::array<double, BenchShape::active_count>& values) noexcept
        {
            std::array<double, BenchShape::active_count> finite_positive{};
            size_t                                       count = 0;
            for (double value : values)
                if (std::isfinite(value) && value > 0.0)
                    finite_positive[count++] = value;
            if (count == 0)
                return 1.0;
            std::sort(finite_positive.begin(), finite_positive.begin() + static_cast<std::ptrdiff_t>(count));
            const size_t mid = count / 2;
            if ((count % 2) == 1)
                return finite_positive[mid];
            return 0.5 * (finite_positive[mid - 1] + finite_positive[mid]);
        }

        void clip_scale_by_anchor(std::array<double, BenchShape::active_count>& values,
                                  double                                        floor,
                                  double                                        max_ratio) noexcept
        {
            const double floor_eval = std::max(residual_scale_floor(floor), residual_scale_tiny());
            const double ratio_eval = residual_scale_max_ratio(max_ratio);
            for (double& value : values)
            {
                if (!std::isfinite(value) || value < floor_eval)
                    value = floor_eval;
            }
            double anchor = balanced_residual_anchor(values);
            if (!std::isfinite(anchor) || anchor < floor_eval)
                anchor = floor_eval;
            const double lower = std::max(floor_eval, anchor / ratio_eval);
            double       upper = anchor * ratio_eval;
            if (!std::isfinite(upper) || upper < floor_eval)
                upper = std::numeric_limits<double>::max();
            upper = std::max(floor_eval, upper);
            for (double& value : values)
                value = std::clamp(value, lower, upper);
        }

        std::array<double, BenchShape::x_size>
        expand_block_scale_values(const std::array<double, BenchShape::active_count>& block_values) noexcept
        {
            std::array<double, BenchShape::x_size> scale{};
            size_t                                 offset = 0;
            for (size_t block = 0; block < BenchShape::active_count; ++block)
            {
                const size_t length = BenchShape::active_lengths[block];
                for (size_t i = 0; i < length; ++i)
                    scale[offset + i] = block_values[block];
                offset += length;
            }
            return scale;
        }

        std::array<double, BenchShape::x_size> build_none_residual_scale() noexcept
        {
            std::array<double, BenchShape::x_size> scale{};
            scale.fill(1.0);
            return scale;
        }

        std::array<double, BenchShape::x_size> build_fast_residual_scale(const PackedVector& residual,
                                                                         double              floor = 1.0,
                                                                         double max_ratio = 1.0e6) noexcept
        {
            std::array<double, BenchShape::active_count> block_values{};
            size_t                                       offset         = 0;
            const double                                 safe_floor     = residual_scale_floor(floor);
            const double                                 safe_max_ratio = residual_scale_max_ratio(max_ratio);
            const double ceiling = std::isfinite(safe_floor * safe_max_ratio) ? safe_floor * safe_max_ratio
                                                                              : std::numeric_limits<double>::max();
            for (size_t block = 0; block < BenchShape::active_count; ++block)
            {
                const size_t length = BenchShape::active_lengths[block];
                const double rms         = stable_rms_block(residual, offset, length);
                const double floored     = rms > safe_floor ? rms : safe_floor;
                block_values[block]      = floored < ceiling ? floored : ceiling;
                offset += length;
            }
            return expand_block_scale_values(block_values);
        }

        std::array<double, BenchShape::x_size> build_balanced_residual_scale(const PackedVector& residual,
                                                                             double              floor,
                                                                             double              max_ratio,
                                                                             double huber_tau) noexcept
        {
            std::array<double, BenchShape::active_count> block_values{};
            size_t                                       offset = 0;
            for (size_t block = 0; block < BenchShape::active_count; ++block)
            {
                const size_t length = BenchShape::active_lengths[block];
                block_values[block] = robust_rms_block(residual, offset, length, huber_tau);
                offset += length;
            }
            clip_scale_by_anchor(block_values, floor, max_ratio);
            return expand_block_scale_values(block_values);
        }

        template <size_t N>
        double relative_abs_rms(const std::array<double, N>& values) noexcept
        {
            if constexpr (N == 0)
                return 0.0;
            double mean = 0.0;
            for (double value : values)
                mean += std::abs(value);
            mean /= static_cast<double>(N);

            double variance = 0.0;
            for (double value : values)
            {
                const double centered = std::abs(value) - mean;
                variance += centered * centered;
            }
            const double relative = std::sqrt(variance / static_cast<double>(N)) / (mean + 1.0e-16);
            return relative <= 1.0e-6 ? 0.0 : relative;
        }

        double estimate_axis_shift_h0(const CaseInput& input) noexcept
        {
            const double epsilon           = input.R0 != 0.0 ? input.a / input.R0 : 0.0;
            const double kappa             = std::abs(input.ka);
            const double elongation_factor = (1.0 + kappa * kappa) != 0.0 ? 2.0 * kappa / (1.0 + kappa * kappa) : 0.0;
            const double source_drive = std::hypot(relative_abs_rms(input.heat), 0.5 * relative_abs_rms(input.current));
            const double h0           = epsilon * elongation_factor * std::tanh(source_drive);
            return std::abs(h0) <= 1.0e-6 ? 0.0 : h0;
        }

        double boundary_curve_strain(const CaseInput& input) noexcept
        {
            constexpr size_t samples = 32;
            constexpr double two_pi  = 6.283185307179586476925286766559;
            const double     kappa   = std::abs(input.ka);

            bool has_c_shape = false;
            for (size_t order = 0; order <= BenchShape::M_max; ++order)
                has_c_shape = has_c_shape || input.c_offsets[order] != 0.0;
            bool has_s_shape = false;
            for (size_t order = 1; order <= BenchShape::M_max; ++order)
                has_s_shape = has_s_shape || input.s_offsets[order] != 0.0;
            if (!has_c_shape && !has_s_shape)
                return 0.0;

            double total = 0.0;
            for (size_t sample = 0; sample < samples; ++sample)
            {
                const double theta     = two_pi * static_cast<double>(sample) / static_cast<double>(samples);
                double       eta       = input.c_offsets[0];
                double       eta_prime = 0.0;
                for (size_t order = 1; order <= BenchShape::M_max; ++order)
                {
                    const double order_eval = static_cast<double>(order);
                    const double phase      = order_eval * theta;
                    eta += input.c_offsets[order] * std::cos(phase);
                    eta_prime -= order_eval * input.c_offsets[order] * std::sin(phase);
                    eta += input.s_offsets[order] * std::sin(phase);
                    eta_prime += order_eval * input.s_offsets[order] * std::cos(phase);
                }

                const double sin_theta      = std::sin(theta);
                const double cos_theta      = std::cos(theta);
                const double speed_boundary = std::sqrt(std::pow(std::sin(theta + eta) * (1.0 + eta_prime), 2.0) +
                                                        std::pow(kappa * cos_theta, 2.0));
                const double speed_ellipse  = std::sqrt(sin_theta * sin_theta + std::pow(kappa * cos_theta, 2.0));
                const double strain         = (speed_boundary - speed_ellipse) / std::max(speed_ellipse, 1.0e-12);
                total += strain * strain;
            }
            return std::sqrt(total / static_cast<double>(samples));
        }

        constexpr size_t fourier_radial_power_for_order(size_t order) noexcept
        {
            return order < BenchShape::K_max ? order : BenchShape::K_max;
        }

        constexpr size_t profile_radial_power(size_t profile_id) noexcept
        {
            if (profile_id >= BenchShape::c0_profile_id && profile_id <= BenchShape::c_profile_id(BenchShape::M_max))
            {
                const size_t order = profile_id - BenchShape::c0_profile_id;
                return order == 0 ? 0 : fourier_radial_power_for_order(order);
            }
            if (profile_id >= BenchShape::s_profile_id(1) && profile_id <= BenchShape::s_profile_id(BenchShape::M_max))
            {
                const size_t order = profile_id - BenchShape::c0_profile_id - BenchShape::M_max;
                return fourier_radial_power_for_order(order);
            }
            return 0;
        }

        double profile_offset_for_initial_seed(const CaseInput& input, size_t profile_id) noexcept
        {
            if (profile_id >= BenchShape::c0_profile_id && profile_id <= BenchShape::c_profile_id(BenchShape::M_max))
                return input.c_offsets[profile_id - BenchShape::c0_profile_id];
            if (profile_id >= BenchShape::s_profile_id(1) && profile_id <= BenchShape::s_profile_id(BenchShape::M_max))
                return input.s_offsets[profile_id - BenchShape::c0_profile_id - BenchShape::M_max];
            return 0.0;
        }

        void seed_geometric_initial_state(CaseInput& input) noexcept
        {
            input.x0.fill(0.0);
            const double h0_est = estimate_axis_shift_h0(input);
            for (size_t active_slot = 0; active_slot < BenchShape::active_count; ++active_slot)
            {
                const size_t profile_id = BenchShape::active_profile_ids[active_slot];
                if (BenchShape::active_lengths[active_slot] == 0)
                    continue;
                const int index = BenchShape::coeff_index[profile_id][0];
                if (index < 0)
                    continue;
                const size_t x_index = static_cast<size_t>(index);
                if (profile_id == BenchShape::h_profile_id)
                {
                    input.x0[x_index] = h0_est;
                }
                else if ((profile_id >= BenchShape::c0_profile_id &&
                          profile_id <= BenchShape::c_profile_id(BenchShape::M_max)) ||
                         (profile_id >= BenchShape::s_profile_id(1) &&
                          profile_id <= BenchShape::s_profile_id(BenchShape::M_max)))
                {
                    const double offset = profile_offset_for_initial_seed(input, profile_id);
                    const size_t power  = profile_radial_power(profile_id);
                    input.x0[x_index]   = -offset / static_cast<double>(2 * power + 1);
                }
            }
        }

        void apply_initial_policy(CaseInput& input)
        {
            switch (input.initial_policy_code)
            {
            case InitialPolicyColdZeros:
                input.x0.fill(0.0);
                return;
            case InitialPolicyColdGeometric:
                seed_geometric_initial_state(input);
                return;
            case InitialPolicyCold:
                if (boundary_curve_strain(input) >= 0.20)
                    seed_geometric_initial_state(input);
                else
                    input.x0.fill(0.0);
                return;
            case InitialPolicyWarmClone:
                return;
            default:
                throw std::runtime_error("invalid initial policy code");
            }
        }

        constexpr const char* solver_entrypoint(SolverKind solver) noexcept
        {
            switch (solver)
            {
            case SolverKind::LevenbergMarquardt:
                return "cminpack::lmdif";
            case SolverKind::Newton:
                return "nonlinear::Newton";
            case SolverKind::NewtonKrylov:
                return "nonlinear::NewtonKrylov";
            case SolverKind::NewtonRaphson:
                return "nonlinear::NewtonRaphson";
            case SolverKind::Powell:
                return "cminpack::hybrd";
            }
            return "unknown";
        }

        constexpr const char* solver_method(SolverKind solver) noexcept
        {
            switch (solver)
            {
            case SolverKind::LevenbergMarquardt:
                return "levenberg-marquardt";
            case SolverKind::Newton:
                return "newton";
            case SolverKind::NewtonKrylov:
                return "newton-krylov";
            case SolverKind::NewtonRaphson:
                return "newton-raphson";
            case SolverKind::Powell:
                return "powell";
            }
            return "unknown";
        }

        constexpr int solver_method_code(SolverKind solver) noexcept
        {
            switch (solver)
            {
            case SolverKind::Powell:
                return SolverMethodPowell;
            case SolverKind::LevenbergMarquardt:
                return SolverMethodLevenbergMarquardt;
            case SolverKind::Newton:
                return SolverMethodNewton;
            case SolverKind::NewtonKrylov:
                return SolverMethodNewtonKrylov;
            case SolverKind::NewtonRaphson:
                return SolverMethodNewtonRaphson;
            default:
                return 0;
            }
        }

        inline SolverKind solver_kind_from_runtime_method_code(int code)
        {
            switch (code)
            {
            case SolverMethodPowell:
                return SolverKind::Powell;
            case SolverMethodLevenbergMarquardt:
                return SolverKind::LevenbergMarquardt;
            case SolverMethodNewton:
                return SolverKind::Newton;
            case SolverMethodNewtonKrylov:
                return SolverKind::NewtonKrylov;
            case SolverMethodNewtonRaphson:
                return SolverKind::NewtonRaphson;
            default:
                throw std::runtime_error("solver.method_code must be 1 (powell), 2 (levenberg-marquardt), "
                                         "3 (newton), 4 (newton-krylov), or 5 (newton-raphson)");
            }
        }

        constexpr const char* initial_policy_name(int code) noexcept
        {
            switch (code)
            {
            case InitialPolicyColdZeros:
                return "cold-zeros";
            case InitialPolicyColdGeometric:
                return "cold-geometric";
            case InitialPolicyCold:
                return "cold";
            case InitialPolicyWarmClone:
                return "warm-clone";
            default:
                return "unknown";
            }
        }

        constexpr bool initial_policy_is_warm_clone(int code) noexcept { return code == InitialPolicyWarmClone; }

        inline void validate_initial_policy_code(int code)
        {
            switch (code)
            {
            case InitialPolicyColdZeros:
            case InitialPolicyColdGeometric:
            case InitialPolicyCold:
            case InitialPolicyWarmClone:
                return;
            default:
                throw std::runtime_error("solver.initial_policy_code must be 1 (cold-zeros), 2 (cold-geometric), "
                                         "3 (cold), or 4 (warm-clone)");
            }
        }

        constexpr const char* residual_normalization_name(int code) noexcept
        {
            switch (code)
            {
            case ResidualNormalizationNone:
                return "none";
            case ResidualNormalizationFast:
                return "fast";
            case ResidualNormalizationBalanced:
                return "balanced";
            case ResidualNormalizationSafe:
                return "safe";
            default:
                return "unknown";
            }
        }

        inline void validate_residual_normalization_code(int code)
        {
            switch (code)
            {
            case ResidualNormalizationNone:
            case ResidualNormalizationFast:
            case ResidualNormalizationBalanced:
            case ResidualNormalizationSafe:
                return;
            default:
                throw std::runtime_error(
                    "solver.residual_normalization_code must be 0 (none), 1 (fast), 2 (balanced), or 3 (safe)");
            }
        }

        constexpr bool solver_info_succeeded(SolverKind solver, int info) noexcept
        {
            switch (solver)
            {
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
            return width == 1 || width == 2 || width == 3 || width == 4 || width == 5 || width == 6 || width == 8 ||
                   width == 9 || width == 10 || width == 12 || width == 18;
        }

        constexpr const char* solver_jacobian(const CaseInput& input) noexcept
        {
            switch (input.solver)
            {
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
            }
            return "unknown";
        }

        CaseInput build_inline_case(int repeat, int warmup, SolverKind solver, int enzyme_jacobian_width)
        {
            CaseInput input{};
            input.heat         = benchmark_scaled_heat;
            input.current      = benchmark_scaled_current;
            input.repeat       = repeat;
            input.warmup       = warmup;
            input.solver       = solver;
            input.enzyme_width = enzyme_jacobian_width;
            input.c_offsets[0] = input.c0_offset;
            if constexpr (BenchShape::M_max >= 1)
                input.s_offsets[1] = input.s1_offset;
            apply_initial_policy(input);
            input.x_scale = build_x_block_scale_vector<BenchShape>(input.x0, profile_params_for_case(input));
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

        std::array<double, BenchShape::x_size>
        decode_z_to_x(std::span<const double, BenchShape::x_size>   z,
                      const std::array<double, BenchShape::x_size>& x_scale) noexcept
        {
            std::array<double, BenchShape::x_size> x;
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                x[i] = z[i] * x_scale[i];
            return x;
        }

        std::array<double, BenchShape::x_size>
        encode_x_to_z(const std::array<double, BenchShape::x_size>& x,
                      const std::array<double, BenchShape::x_size>& x_scale) noexcept
        {
            std::array<double, BenchShape::x_size> z;
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                z[i] = x[i] / x_scale[i];
            return z;
        }

        BenchOperator::Setup setup_for_case(const CaseInput& input) noexcept
        {
            BenchOperator::Setup setup{};
            setup.profile_params = profile_params_for_case(input);
            setup.fix_rho        = input.fix_rho;
            for (size_t i = 0; i < BenchSource::sample_count; ++i)
            {
                setup.heat[i]    = input.heat[i];
                setup.current[i] = input.current[i];
            }
            return setup;
        }

        BenchOperator::SolveParams solve_params_for_case(const CaseInput& input) noexcept
        {
            BenchOperator::SolveParams params{};
            params.a  = input.a;
            params.R0 = input.R0;
            params.Z0 = input.Z0;
            params.B0 = input.B0;
            params.Ip = input.Ip;
            return params;
        }

        BenchOperator make_operator_for_case(const CaseInput& input) noexcept
        {
            BenchOperator op{setup_for_case(input)};
            op.set_solve_params(solve_params_for_case(input));
            return op;
        }

        struct SolveContext
        {
            BenchOperator op;
            CaseInput     input{};
            int           evaluations                    = 0;
            int           jacobian_component_evaluations = 0;
            double        residual_callback_ms           = 0.0;
            double        residual_kernel_ms             = 0.0;
            double        residual_scale_ms              = 0.0;
            double        final_residual_ms              = 0.0;
            double        jacobian_callback_ms           = 0.0;
            double        jvp_callback_ms                = 0.0;
            double        linear_solve_ms                = 0.0;

            explicit SolveContext(const CaseInput& case_input)
                : op(make_operator_for_case(case_input)), input(case_input)
            {
            }

            void reset_solve_counters() noexcept
            {
                evaluations                    = 0;
                jacobian_component_evaluations = 0;
                residual_callback_ms           = 0.0;
                residual_kernel_ms             = 0.0;
                residual_scale_ms              = 0.0;
                final_residual_ms              = 0.0;
                jacobian_callback_ms           = 0.0;
                jvp_callback_ms                = 0.0;
                linear_solve_ms                = 0.0;
            }

            void raw_residual(std::span<const double, BenchShape::x_size> x,
                              std::span<double, BenchShape::x_size>       residual) noexcept
            {
                PackedVector raw{uninitialized};
                op.evaluate(x, raw);
                for (size_t i = 0; i < BenchShape::x_size; ++i)
                    residual[i] = raw[i];
            }
        };

        double deterministic_probe_sign(size_t probe, size_t index) noexcept
        {
            uint64_t value = (static_cast<uint64_t>(probe) + 1ULL) * 0x9e3779b97f4a7c15ULL;
            value ^= (static_cast<uint64_t>(index) + 1ULL) * 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 30U;
            value *= 0xbf58476d1ce4e5b9ULL;
            value ^= value >> 27U;
            value *= 0x94d049bb133111ebULL;
            value ^= value >> 31U;
            return (value & 1ULL) == 0ULL ? -1.0 : 1.0;
        }

        std::array<double, BenchShape::x_size>
        build_safe_residual_scale(SolveContext&       context,
                                  const PackedVector& initial_raw,
                                  double              floor,
                                  double              max_ratio,
                                  double              huber_tau,
                                  int                 probe_count,
                                  double              probe_step,
                                  double              sensitivity_lambda) noexcept
        {
            if (probe_count <= 0 || !std::isfinite(probe_step) || probe_step <= 0.0)
                return build_balanced_residual_scale(initial_raw, floor, max_ratio, huber_tau);

            std::array<double, BenchShape::active_count> amplitude_values{};
            size_t                                       offset = 0;
            for (size_t block = 0; block < BenchShape::active_count; ++block)
            {
                const size_t length  = BenchShape::active_lengths[block];
                amplitude_values[block] = robust_rms_block(initial_raw, offset, length, huber_tau);
                offset += length;
            }

            std::array<double, BenchShape::active_count> sensitivity_sq{};
            for (int probe = 0; probe < probe_count; ++probe)
            {
                std::array<double, BenchShape::x_size> probe_x{};
                for (size_t i = 0; i < BenchShape::x_size; ++i)
                    probe_x[i] = context.input.x0[i] +
                                 probe_step * context.input.x_scale[i] *
                                     deterministic_probe_sign(static_cast<size_t>(probe), i);

                PackedVector probe_raw{uninitialized};
                context.raw_residual(std::span<const double, BenchShape::x_size>{probe_x.data(), BenchShape::x_size},
                                     std::span<double, BenchShape::x_size>{probe_raw.data(), BenchShape::x_size});

                PackedVector diff{uninitialized};
                for (size_t i = 0; i < BenchShape::x_size; ++i)
                    diff[i] = (probe_raw[i] - initial_raw[i]) / probe_step;

                offset = 0;
                for (size_t block = 0; block < BenchShape::active_count; ++block)
                {
                    const size_t length = BenchShape::active_lengths[block];
                    const double value  = robust_rms_block(diff, offset, length, huber_tau);
                    sensitivity_sq[block] += value * value;
                    offset += length;
                }
            }

            const double lambda = std::isfinite(sensitivity_lambda) && sensitivity_lambda > 0.0
                                      ? sensitivity_lambda
                                      : 0.0;
            std::array<double, BenchShape::active_count> combined{};
            for (size_t block = 0; block < BenchShape::active_count; ++block)
            {
                const double sensitivity = std::sqrt(sensitivity_sq[block] / static_cast<double>(probe_count));
                combined[block]          = std::hypot(amplitude_values[block], lambda * sensitivity);
            }
            clip_scale_by_anchor(combined, floor, max_ratio);
            return expand_block_scale_values(combined);
        }

        std::array<double, BenchShape::x_size>
        build_residual_scale_for_context(SolveContext& context, const PackedVector& initial_raw) noexcept
        {
            switch (context.input.residual_normalization_code)
            {
            case ResidualNormalizationNone:
                return build_none_residual_scale();
            case ResidualNormalizationFast:
                return build_fast_residual_scale(initial_raw,
                                                 context.input.residual_normalization_floor,
                                                 context.input.residual_normalization_max_ratio);
            case ResidualNormalizationBalanced:
                return build_balanced_residual_scale(initial_raw,
                                                     context.input.residual_normalization_floor,
                                                     context.input.residual_normalization_max_ratio,
                                                     context.input.residual_normalization_huber_tau);
            case ResidualNormalizationSafe:
                return build_safe_residual_scale(context,
                                                 initial_raw,
                                                 context.input.residual_normalization_floor,
                                                 context.input.residual_normalization_max_ratio,
                                                 context.input.residual_normalization_huber_tau,
                                                 context.input.residual_normalization_probe_count,
                                                 context.input.residual_normalization_probe_step,
                                                 context.input.residual_normalization_sensitivity_lambda);
            default:
                return build_none_residual_scale();
            }
        }

        void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept;

#ifdef ENABLE_ENZYME
        struct EnzymeResidualContext
        {
            BenchOperator                          op;
            std::array<double, BenchShape::x_size> x_scale{};
            std::array<double, BenchShape::x_size> residual_scale{};

            EnzymeResidualContext() : op(make_operator_for_case(CaseInput{})) {}

            explicit EnzymeResidualContext(const CaseInput& input) : op(make_operator_for_case(input)) {}
        };

        EnzymeResidualContext enzyme_context_for_input(const CaseInput& input) noexcept
        {
            EnzymeResidualContext context{input};
            context.x_scale        = input.x_scale;
            context.residual_scale = input.residual_scale;
            return context;
        }

        double scaled_residual_vector_for_enzyme(double* z, double* fvec, void* context_value) noexcept
        {
            auto&      context = *static_cast<EnzymeResidualContext*>(context_value);
            const auto x =
                decode_z_to_x(std::span<const double, BenchShape::x_size>{z, BenchShape::x_size}, context.x_scale);

            PackedVector raw{uninitialized};
            context.op.evaluate(std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size}, raw);
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

        int pf_lm_residual_x(void* data, int m, int n, const double* x, double* fvec, int iflag)
        {
            if (iflag <= 0 || m != static_cast<int>(BenchShape::x_size) ||
                n != static_cast<int>(BenchShape::x_size))
                return 0;

            auto& context = *static_cast<SolveContext*>(data);
            ++context.evaluations;

            const auto callback_started = std::chrono::steady_clock::now();
            PackedVector raw{uninitialized};
            const auto   kernel_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, BenchShape::x_size>{x, BenchShape::x_size},
                                 std::span<double, BenchShape::x_size>{raw.data(), BenchShape::x_size});
            context.residual_kernel_ms += elapsed_ms_since(kernel_started);

            const auto scale_started = std::chrono::steady_clock::now();
            for (size_t i = 0; i < BenchShape::x_size; ++i)
            {
                const double scaled = raw[i] / context.input.residual_scale[i];
                fvec[i] = context.input.residual_normalization_code == ResidualNormalizationFast
                              ? std::asinh(scaled)
                              : scaled;
            }
            context.residual_scale_ms += elapsed_ms_since(scale_started);
            context.residual_callback_ms += elapsed_ms_since(callback_started);
            return 0;
        }

        void scaled_residual_z_no_count(SolveContext& context, const double* z, double* fvec) noexcept
        {
            const auto   callback_started = std::chrono::steady_clock::now();
            const auto   x = decode_z_to_x(std::span<const double, BenchShape::x_size>{z, BenchShape::x_size},
                                         context.input.x_scale);
            PackedVector raw{uninitialized};
            const auto   kernel_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size},
                                 std::span<double, BenchShape::x_size>{raw.data(), BenchShape::x_size});
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
                enzyme::Duplicated<void*>{static_cast<void*>(&jvp_context), static_cast<void*>(&jvp_context_dot)});
        }

        template <size_t Width>
        void fill_enzyme_jacobian_z_vector(SolveContext& context, const double* z, double* fjac, int ldfjac)
        {
            static_assert(Width > 0);
            constexpr size_t n                 = BenchShape::x_size;
            constexpr size_t lane_stride_bytes = n * sizeof(double);

            std::array<double, n> z_primal;
            for (size_t i = 0; i < n; ++i)
                z_primal[i] = z[i];

            for (size_t first_col = 0; first_col < n; first_col += Width)
            {
                std::array<double, Width * n>            z_dot{};
                std::array<double, n>                    f_primal{};
                std::array<double, Width * n>            f_dot{};
                EnzymeResidualContext                    chunk_context = enzyme_context_for_input(context.input);
                std::array<EnzymeResidualContext, Width> chunk_context_dot{};
                std::memset(chunk_context_dot.data(), 0, sizeof(chunk_context_dot));

                const size_t lane_count = std::min(Width, n - first_col);
                for (size_t lane = 0; lane < lane_count; ++lane)
                    z_dot[lane * n + first_col + lane] = 1.0;

                __enzyme_fwddiff<void>(reinterpret_cast<void*>(scaled_residual_vector_for_enzyme),
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
                                       static_cast<void*>(chunk_context_dot.data()));

                for (size_t lane = 0; lane < lane_count; ++lane)
                {
                    const size_t col = first_col + lane;
                    for (size_t row = 0; row < n; ++row)
                        fjac[row + static_cast<size_t>(ldfjac) * col] = f_dot[lane * n + row];
                }
            }
            context.jacobian_component_evaluations += static_cast<int>(n);
        }

        void fill_enzyme_jacobian_z_scalar(SolveContext& context, const double* z, double* fjac, int ldfjac)
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
                                              static_cast<void*>(&column_context_dot)});
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

        void scaled_residual_z_array_no_count(SolveContext&                                 context,
                                              const std::array<double, BenchShape::x_size>& z,
                                              std::array<double, BenchShape::x_size>&       fvec) noexcept
        {
            scaled_residual_z_no_count(context, z.data(), fvec.data());
        }

        int
        pf_residual_jacobian_z(void* data, int n, const double* z, double* fvec, double* fjac, int ldfjac, int iflag)
        {
            if (n != static_cast<int>(BenchShape::x_size))
                return 0;
            if (iflag == 1)
                return pf_residual_z(data, n, z, fvec, iflag);
            if (iflag == 2)
            {
                auto&      context = *static_cast<SolveContext*>(data);
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
                constexpr size_t          n = BenchShape::x_size;
                std::array<double, n * n> column_major{};
                fill_enzyme_jacobian_z(*context, z, column_major.data(), static_cast<int>(n));
                for (size_t row = 0; row < n; ++row)
                    for (size_t col = 0; col < n; ++col)
                        jacobian[row * n + col] = column_major[row + n * col];
            }

            void jvp(const double* z, const double* v, double* jv) const { fill_enzyme_jvp_z(*context, z, v, jv); }
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

        void fill_solve_result_from_z(
            SolveContext& context, SolveResult& result, const double* z, int info, int nfev, int njev, int callbacks)
        {
            result.info      = info;
            result.nfev      = nfev;
            result.njev      = njev;
            result.callbacks = callbacks;
            result.x         = decode_z_to_x(std::span<const double, BenchShape::x_size>{z, BenchShape::x_size},
                                     context.input.x_scale);
            const auto final_residual_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, BenchShape::x_size>{result.x.data(), BenchShape::x_size},
                                 std::span<double, BenchShape::x_size>{result.raw.data(), BenchShape::x_size});
            result.final_residual_ms = elapsed_ms_since(final_residual_started);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
            result.raw_norm             = norm2(std::span<const double, BenchShape::x_size>{
                result.raw.data(),
                BenchShape::x_size,
            });
            result.scaled_norm          = norm2(std::span<const double, BenchShape::x_size>{
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
            result.accepted = result.raw_norm <= acceptance_threshold(context.input);
        }

        void fill_solve_result_from_x(
            SolveContext& context, SolveResult& result, const double* x, int info, int nfev, int njev, int callbacks)
        {
            result.info      = info;
            result.nfev      = nfev;
            result.njev      = njev;
            result.callbacks = callbacks;
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                result.x[i] = x[i];
            const auto final_residual_started = std::chrono::steady_clock::now();
            context.raw_residual(std::span<const double, BenchShape::x_size>{result.x.data(), BenchShape::x_size},
                                 std::span<double, BenchShape::x_size>{result.raw.data(), BenchShape::x_size});
            result.final_residual_ms = elapsed_ms_since(final_residual_started);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
            result.raw_norm             = norm2(std::span<const double, BenchShape::x_size>{
                result.raw.data(),
                BenchShape::x_size,
            });
            result.scaled_norm          = norm2(std::span<const double, BenchShape::x_size>{
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
            result.accepted = result.raw_norm <= acceptance_threshold(context.input);
        }

        SolveResult run_hybrd_once(SolveContext& context)
        {
            context.reset_solve_counters();
            auto         z = encode_x_to_z(context.input.x0, context.input.x_scale);
            PackedVector fvec{uninitialized};

            constexpr int                                               n  = static_cast<int>(BenchShape::x_size);
            constexpr int                                               ml = n - 1;
            constexpr int                                               mu = n - 1;
            constexpr int                                               lr = n * (n + 1) / 2;
            std::array<double, BenchShape::x_size>                      diag;
            std::array<double, BenchShape::x_size * BenchShape::x_size> fjac;
            std::array<double, static_cast<size_t>(lr)>                 r;
            std::array<double, BenchShape::x_size>                      qtf;
            std::array<double, BenchShape::x_size>                      wa1;
            std::array<double, BenchShape::x_size>                      wa2;
            std::array<double, BenchShape::x_size>                      wa3;
            std::array<double, BenchShape::x_size>                      wa4;
            int                                                         nfev = 0;
            const int                                                   info = hybrd(pf_residual_z,
                                   &context,
                                   n,
                                   z.data(),
                                   fvec.data(),
                                   context.input.max_residual,
                                   max_solver_evaluations(context.input),
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
                                   wa4.data());

            SolveResult result{};
            fill_solve_result_from_z(context, result, z.data(), info, nfev, 0, context.evaluations);
            result.jacobian_component_evaluations = context.jacobian_component_evaluations;
            return result;
        }

        SolveResult run_lmdif_once(SolveContext& context)
        {
            context.reset_solve_counters();
            std::array<double, BenchShape::x_size> x{};
            std::copy(context.input.x0.begin(), context.input.x0.end(), x.begin());
            PackedVector fvec{uninitialized};

            constexpr int                                               m = static_cast<int>(BenchShape::x_size);
            constexpr int                                               n = static_cast<int>(BenchShape::x_size);
            std::array<double, BenchShape::x_size>                      diag{};
            std::array<double, BenchShape::x_size * BenchShape::x_size> fjac;
            std::array<int, BenchShape::x_size>                         ipvt;
            std::array<double, BenchShape::x_size>                      qtf;
            std::array<double, BenchShape::x_size>                      wa1;
            std::array<double, BenchShape::x_size>                      wa2;
            std::array<double, BenchShape::x_size>                      wa3;
            std::array<double, BenchShape::x_size>                      wa4;
            diag.fill(1.0);

            int       nfev = 0;
            const int info = lmdif(pf_lm_residual_x,
                                   &context,
                                   m,
                                   n,
                                   x.data(),
                                   fvec.data(),
                                   context.input.max_residual,
                                   context.input.max_residual,
                                   context.input.max_residual,
                                   max_solver_evaluations(context.input),
                                   veqpy_lm_eps,
                                   diag.data(),
                                   veqpy_lm_mode,
                                   veqpy_lm_factor,
                                   veqpy_lm_nprint,
                                   &nfev,
                                   fjac.data(),
                                   m,
                                   ipvt.data(),
                                   qtf.data(),
                                   wa1.data(),
                                   wa2.data(),
                                   wa3.data(),
                                   wa4.data());

            SolveResult result{};
            fill_solve_result_from_x(context, result, x.data(), info, nfev, 0, context.evaluations);
            result.jacobian_component_evaluations = context.jacobian_component_evaluations;
            return result;
        }

#ifdef ENABLE_ENZYME
        SolveResult run_hybrj_once(SolveContext& context)
        {
            context.reset_solve_counters();
            auto         z = encode_x_to_z(context.input.x0, context.input.x_scale);
            PackedVector fvec{uninitialized};

            constexpr int                                               n  = static_cast<int>(BenchShape::x_size);
            constexpr int                                               lr = n * (n + 1) / 2;
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
            const int                                                   info = hybrj(pf_residual_jacobian_z,
                                   &context,
                                   n,
                                   z.data(),
                                   fvec.data(),
                                   fjac.data(),
                                   n,
                                   context.input.max_residual,
                                   max_solver_evaluations(context.input),
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
                                   wa4.data());

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
            const auto                                 encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, BenchShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualProblem problem{&context};
            auto                  solver = nonlinear::make_solver<Policy>(problem);
            solver.context.tolerance     = context.input.max_residual;
            if constexpr (requires { solver.context.max_iterations; })
                solver.context.max_iterations = max_solver_evaluations(context.input);
            if constexpr (requires { solver.context.max_dimension; })
                solver.context.max_dimension = static_cast<int>(BenchShape::x_size);

            solver.optimize_inplace(z);

            SolveResult result{};
            fill_solve_result_from_z(context,
                                     result,
                                     z.data(),
                                     solver.context.info,
                                     solver.context.evaluations,
                                     jacobian_evaluation_count(solver.context),
                                     solver.context.evaluations);
            result.jacobian_component_evaluations = context.jacobian_component_evaluations;
            result.jvp_evaluations                = jvp_evaluation_count(solver.context);
            result.linear_iterations              = linear_iteration_count(solver.context);
            return result;
        }

        template <typename Policy>
        SolveResult run_nonlinear_residual_policy_once(SolveContext& context)
        {
            context.reset_solve_counters();
            const auto                                 encoded = encode_x_to_z(context.input.x0, context.input.x_scale);
            tensor::Vector<double, BenchShape::x_size> z{uninitialized};
            std::copy(encoded.begin(), encoded.end(), z.begin());

            ScaledResidualOnlyProblem problem{&context};
            auto                      solver = nonlinear::make_solver<Policy>(problem);
            solver.context.tolerance         = context.input.max_residual;
            if constexpr (requires { solver.context.max_evaluations; })
                solver.context.max_evaluations = max_solver_evaluations(context.input);
            if constexpr (requires { solver.context.max_iterations; })
                solver.context.max_iterations = max_solver_evaluations(context.input);

            solver.optimize_inplace(z);

            SolveResult result{};
            fill_solve_result_from_z(context,
                                     result,
                                     z.data(),
                                     solver.context.info,
                                     solver.context.evaluations,
                                     jacobian_evaluation_count(solver.context),
                                     solver.context.evaluations);
            return result;
        }

        SolveResult run_solver_once(SolveContext& context)
        {
            if (context.input.solver == SolverKind::LevenbergMarquardt)
                return run_lmdif_once(context);
            if (context.input.solver == SolverKind::Newton)
                return run_nonlinear_policy_once<nonlinear::Newton>(context);
            if (context.input.solver == SolverKind::NewtonKrylov)
                return run_nonlinear_policy_once<nonlinear::NewtonKrylov>(context);
            if (context.input.solver == SolverKind::NewtonRaphson)
                return run_nonlinear_policy_once<nonlinear::NewtonRaphson>(context);
            if (context.input.solver == SolverKind::Powell)
                return run_hybrd_once(context);
            throw std::runtime_error("unsupported solver kind");
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

    } // namespace

} // namespace veqlib_kernel_api
