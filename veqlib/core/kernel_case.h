#pragma once

// Runtime case defaults, scaling, normalization, and option-code helpers for VEQlib kernels.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "kernel_topology.h"
#include "math.h"

namespace veqlib_kernel_api
{
    namespace
    {
        using math::is_finite;
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

        template <typename Shape>
        constexpr double coefficient_space_profile_scale(size_t profile_id, double physical_scale) noexcept
        {
            if (profile_id == Shape::F_profile_id && physical_scale != 0.0)
                return physical_scale * physical_scale;
            return physical_scale;
        }

        profiles::ProfileRuntimeParams<KernelShape> profile_params_for_case(const CaseInput& input) noexcept
        {
            profiles::ProfileRuntimeParams<KernelShape> params{};
            params.offsets[KernelShape::kappa_profile_id] = input.ka;
            params.scales[KernelShape::F_profile_id]      = input.R0 * input.B0;
            for (size_t order = 0; order <= KernelShape::M_max; ++order)
                params.offsets[KernelShape::c_profile_id(order)] = input.c_offsets[order];
            for (size_t order = 1; order <= KernelShape::M_max; ++order)
                params.offsets[KernelShape::s_profile_id(order)] = input.s_offsets[order];
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
                profile_scale        = coefficient_space_profile_scale<Shape>(profile_id, profile_scale);
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
            return is_finite(value) && value > 0.0 ? value : fallback;
        }

        double residual_scale_floor(double floor) noexcept
        {
            return finite_positive_or(floor, 1.0);
        }

        double residual_scale_max_ratio(double max_ratio) noexcept
        {
            return is_finite(max_ratio) && max_ratio >= 1.0 ? max_ratio : 1.0;
        }

        double median_sorted_prefix(std::array<double, KernelShape::x_size>& values, size_t count) noexcept
        {
            if (count == 0)
                return 0.0;
            std::sort(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count));
            const size_t mid = count / 2;
            if ((count % 2) == 1)
                return values[mid];
            return 0.5 * (values[mid - 1] + values[mid]);
        }

        double stable_rms_clipped(std::array<double, KernelShape::x_size>& values,
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
                if (is_finite(value) && value > max_abs)
                    max_abs = value;
            }
            if (max_abs == 0.0)
                return 0.0;
            double total = 0.0;
            for (size_t i = 0; i < length; ++i)
            {
                const double value = std::abs(residual[offset + i]);
                if (!is_finite(value))
                    continue;
                const double scaled = value / max_abs;
                total += scaled * scaled;
            }
            return max_abs * std::sqrt(total / static_cast<double>(length));
        }

        double robust_rms_block(const PackedVector& residual, size_t offset, size_t length, double huber_tau) noexcept
        {
            std::array<double, KernelShape::x_size> finite{};
            size_t                                 count = 0;
            for (size_t i = 0; i < length; ++i)
            {
                const double value = std::abs(residual[offset + i]);
                if (is_finite(value))
                    finite[count++] = value;
            }
            if (count == 0)
                return 0.0;

            std::array<double, KernelShape::x_size> sorted = finite;
            const double                           center = median_sorted_prefix(sorted, count);
            std::array<double, KernelShape::x_size> deviations{};
            for (size_t i = 0; i < count; ++i)
                deviations[i] = std::abs(finite[i] - center);
            const double mad = median_sorted_prefix(deviations, count);
            double       cutoff =
                center + std::max(huber_tau, 0.0) * 1.4826 * mad;
            if (!is_finite(cutoff) || cutoff <= 0.0)
                cutoff = center;
            return stable_rms_clipped(finite, count, cutoff);
        }

        double balanced_residual_anchor(std::array<double, KernelShape::active_count>& values) noexcept
        {
            std::array<double, KernelShape::active_count> finite_positive{};
            size_t                                       count = 0;
            for (double value : values)
                if (is_finite(value) && value > 0.0)
                    finite_positive[count++] = value;
            if (count == 0)
                return 1.0;
            std::sort(finite_positive.begin(), finite_positive.begin() + static_cast<std::ptrdiff_t>(count));
            const size_t mid = count / 2;
            if ((count % 2) == 1)
                return finite_positive[mid];
            return 0.5 * (finite_positive[mid - 1] + finite_positive[mid]);
        }

        void clip_scale_by_anchor(std::array<double, KernelShape::active_count>& values,
                                  double                                        floor,
                                  double                                        max_ratio) noexcept
        {
            const double floor_eval = std::max(residual_scale_floor(floor), residual_scale_tiny());
            const double ratio_eval = residual_scale_max_ratio(max_ratio);
            for (double& value : values)
            {
                if (!is_finite(value) || value < floor_eval)
                    value = floor_eval;
            }
            double anchor = balanced_residual_anchor(values);
            if (!is_finite(anchor) || anchor < floor_eval)
                anchor = floor_eval;
            const double lower = std::max(floor_eval, anchor / ratio_eval);
            double       upper = anchor * ratio_eval;
            if (!is_finite(upper) || upper < floor_eval)
                upper = std::numeric_limits<double>::max();
            upper = std::max(floor_eval, upper);
            for (double& value : values)
                value = std::clamp(value, lower, upper);
        }

        std::array<double, KernelShape::x_size>
        expand_block_scale_values(const std::array<double, KernelShape::active_count>& block_values) noexcept
        {
            std::array<double, KernelShape::x_size> scale{};
            size_t                                 offset = 0;
            for (size_t block = 0; block < KernelShape::active_count; ++block)
            {
                const size_t length = KernelShape::active_lengths[block];
                for (size_t i = 0; i < length; ++i)
                    scale[offset + i] = block_values[block];
                offset += length;
            }
            return scale;
        }

        std::array<double, KernelShape::x_size> build_none_residual_scale() noexcept
        {
            std::array<double, KernelShape::x_size> scale{};
            scale.fill(1.0);
            return scale;
        }

        std::array<double, KernelShape::x_size> build_fast_residual_scale(const PackedVector& residual,
                                                                         double              floor = 1.0,
                                                                         double max_ratio = 1.0e6) noexcept
        {
            std::array<double, KernelShape::active_count> block_values{};
            size_t                                       offset         = 0;
            const double                                 safe_floor     = residual_scale_floor(floor);
            const double                                 safe_max_ratio = residual_scale_max_ratio(max_ratio);
            const double ceiling = is_finite(safe_floor * safe_max_ratio) ? safe_floor * safe_max_ratio
                                                                              : std::numeric_limits<double>::max();
            for (size_t block = 0; block < KernelShape::active_count; ++block)
            {
                const size_t length = KernelShape::active_lengths[block];
                const double rms         = stable_rms_block(residual, offset, length);
                const double floored     = rms > safe_floor ? rms : safe_floor;
                block_values[block]      = floored < ceiling ? floored : ceiling;
                offset += length;
            }
            return expand_block_scale_values(block_values);
        }

        std::array<double, KernelShape::x_size> build_balanced_residual_scale(const PackedVector& residual,
                                                                             double              floor,
                                                                             double              max_ratio,
                                                                             double huber_tau) noexcept
        {
            std::array<double, KernelShape::active_count> block_values{};
            size_t                                       offset = 0;
            for (size_t block = 0; block < KernelShape::active_count; ++block)
            {
                const size_t length = KernelShape::active_lengths[block];
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
            for (size_t order = 0; order <= KernelShape::M_max; ++order)
                has_c_shape = has_c_shape || input.c_offsets[order] != 0.0;
            bool has_s_shape = false;
            for (size_t order = 1; order <= KernelShape::M_max; ++order)
                has_s_shape = has_s_shape || input.s_offsets[order] != 0.0;
            if (!has_c_shape && !has_s_shape)
                return 0.0;

            double total = 0.0;
            for (size_t sample = 0; sample < samples; ++sample)
            {
                const double theta     = two_pi * static_cast<double>(sample) / static_cast<double>(samples);
                double       eta       = input.c_offsets[0];
                double       eta_prime = 0.0;
                for (size_t order = 1; order <= KernelShape::M_max; ++order)
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
            return order < KernelShape::K_max ? order : KernelShape::K_max;
        }

        constexpr size_t profile_radial_power(size_t profile_id) noexcept
        {
            if (profile_id >= KernelShape::c0_profile_id && profile_id <= KernelShape::c_profile_id(KernelShape::M_max))
            {
                const size_t order = profile_id - KernelShape::c0_profile_id;
                return order == 0 ? 0 : fourier_radial_power_for_order(order);
            }
            if (profile_id >= KernelShape::s_profile_id(1) && profile_id <= KernelShape::s_profile_id(KernelShape::M_max))
            {
                const size_t order = profile_id - KernelShape::c0_profile_id - KernelShape::M_max;
                return fourier_radial_power_for_order(order);
            }
            return 0;
        }

        double profile_offset_for_initial_seed(const CaseInput& input, size_t profile_id) noexcept
        {
            if (profile_id >= KernelShape::c0_profile_id && profile_id <= KernelShape::c_profile_id(KernelShape::M_max))
                return input.c_offsets[profile_id - KernelShape::c0_profile_id];
            if (profile_id >= KernelShape::s_profile_id(1) && profile_id <= KernelShape::s_profile_id(KernelShape::M_max))
                return input.s_offsets[profile_id - KernelShape::c0_profile_id - KernelShape::M_max];
            return 0.0;
        }

        void seed_geometric_initial_state(CaseInput& input) noexcept
        {
            input.x0.fill(0.0);
            const double h0_est = estimate_axis_shift_h0(input);
            for (size_t active_slot = 0; active_slot < KernelShape::active_count; ++active_slot)
            {
                const size_t profile_id = KernelShape::active_profile_ids[active_slot];
                if (KernelShape::active_lengths[active_slot] == 0)
                    continue;
                const int index = KernelShape::coeff_index[profile_id][0];
                if (index < 0)
                    continue;
                const size_t x_index = static_cast<size_t>(index);
                if (profile_id == KernelShape::h_profile_id)
                {
                    input.x0[x_index] = h0_est;
                }
                else if ((profile_id >= KernelShape::c0_profile_id &&
                          profile_id <= KernelShape::c_profile_id(KernelShape::M_max)) ||
                         (profile_id >= KernelShape::s_profile_id(1) &&
                          profile_id <= KernelShape::s_profile_id(KernelShape::M_max)))
                {
                    const double offset = profile_offset_for_initial_seed(input, profile_id);
                    const size_t power  = profile_radial_power(profile_id);
                    input.x0[x_index]   = -offset / static_cast<double>(2 * power + 1);
                }
            }
        }

        void apply_cold_policy(CaseInput& input, int policy_code)
        {
            switch (policy_code)
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
            default:
                throw std::runtime_error("invalid cold policy code");
            }
        }

        void apply_initial_policy(CaseInput& input) { apply_cold_policy(input, input.initial_policy_code); }

        constexpr const char* solver_entrypoint(SolverKind solver) noexcept
        {
            switch (solver)
            {
            case SolverKind::LevenbergMarquardt:
                return "nonlinear::LevenbergMarquardt";
            case SolverKind::NewtonKrylov:
                return "nonlinear::NewtonKrylov";
            case SolverKind::NewtonRaphson:
                return "nonlinear::NewtonRaphson";
            case SolverKind::Powell:
                return "nonlinear::Powell";
            }
            return "unknown";
        }

        constexpr const char* solver_method(SolverKind solver) noexcept
        {
            switch (solver)
            {
            case SolverKind::LevenbergMarquardt:
                return "levenberg-marquardt";
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
            case SolverMethodNewtonKrylov:
                return SolverKind::NewtonKrylov;
            case SolverMethodNewtonRaphson:
                return SolverKind::NewtonRaphson;
            default:
                throw std::runtime_error("solver.method_code must be 1 (powell), 2 (levenberg-marquardt), "
                                         "4 (newton-krylov), or 5 (newton-raphson)");
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
            default:
                return "unknown";
            }
        }

        constexpr const char* continue_policy_name(int code) noexcept
        {
            switch (code)
            {
            case ContinuePolicyColdZeros:
                return "cold-zeros";
            case ContinuePolicyColdGeometric:
                return "cold-geometric";
            case ContinuePolicyCold:
                return "cold";
            case ContinuePolicyWarmFixed:
                return "warm-fixed";
            case ContinuePolicyWarmPredict:
                return "warm-predict";
            case ContinuePolicyWarmChord:
                return "warm-chord";
            case ContinuePolicyWarm:
                return "warm";
            default:
                return "unknown";
            }
        }

        constexpr bool continue_policy_is_cold(int code) noexcept
        {
            return code == ContinuePolicyColdZeros || code == ContinuePolicyColdGeometric ||
                   code == ContinuePolicyCold;
        }

        constexpr bool continue_policy_uses_warm_state(int code) noexcept
        {
            return code == ContinuePolicyWarmFixed || code == ContinuePolicyWarmPredict ||
                   code == ContinuePolicyWarmChord || code == ContinuePolicyWarm;
        }

        constexpr bool continue_policy_uses_predictor(int code) noexcept
        {
            return code == ContinuePolicyWarmPredict || code == ContinuePolicyWarmChord ||
                   code == ContinuePolicyWarm;
        }

        constexpr bool continue_policy_uses_chord(int code) noexcept { return code == ContinuePolicyWarmChord; }

        constexpr int resolved_continue_policy(int code) noexcept
        {
            return code == ContinuePolicyWarm ? ContinuePolicyWarmPredict : code;
        }

        inline void validate_initial_policy_code(int code)
        {
            switch (code)
            {
            case InitialPolicyColdZeros:
            case InitialPolicyColdGeometric:
            case InitialPolicyCold:
                return;
            default:
                throw std::runtime_error("solver.initial_policy_code must be 1 (cold-zeros), 2 (cold-geometric), "
                                         "or 3 (cold)");
            }
        }

        inline void validate_continue_policy_code(int code)
        {
            switch (code)
            {
            case ContinuePolicyColdZeros:
            case ContinuePolicyColdGeometric:
            case ContinuePolicyCold:
            case ContinuePolicyWarmFixed:
            case ContinuePolicyWarmPredict:
            case ContinuePolicyWarmChord:
            case ContinuePolicyWarm:
                return;
            default:
                throw std::runtime_error(
                    "solver.continue_policy_code must be 1 (cold-zeros), 2 (cold-geometric), 3 (cold), "
                    "4 (warm-fixed), 5 (warm-predict), 6 (warm-chord), or 7 (warm)");
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
            case SolverKind::NewtonKrylov:
            case SolverKind::NewtonRaphson:
            case SolverKind::Powell:
                return info > 0;
            }
            return false;
        }

        constexpr const char* solver_jacobian(const CaseInput& input) noexcept
        {
            switch (input.solver)
            {
            case SolverKind::LevenbergMarquardt:
#ifdef ENABLE_ENZYME
                return "Enzyme batched dense Jacobian through nonlinear::LevenbergMarquardt";
#else
                return "CMINPACK forward difference through nonlinear::LevenbergMarquardt";
#endif
            case SolverKind::NewtonKrylov:
#ifdef ENABLE_ENZYME
                return "Enzyme Jacobian-vector product through GMRES";
#else
                return "finite-difference Jacobian-vector product through GMRES";
#endif
            case SolverKind::NewtonRaphson:
#ifdef ENABLE_ENZYME
                return "Enzyme batched dense Jacobian through dense Newton";
#else
                return "finite-difference dense Jacobian through dense Newton";
#endif
            case SolverKind::Powell:
#ifdef ENABLE_ENZYME
                return "Enzyme batched dense Jacobian through nonlinear::Powell";
#else
                return "CMINPACK forward difference through nonlinear::Powell";
#endif
            }
            return "unknown";
        }

        CaseInput build_inline_case(int repeat, int warmup, SolverKind solver)
        {
            CaseInput input{};
            input.heat         = default_scaled_heat;
            input.current      = default_scaled_current;
            input.repeat       = repeat;
            input.warmup       = warmup;
            input.solver       = solver;
            input.c_offsets[0] = input.c0_offset;
            if constexpr (KernelShape::M_max >= 1)
                input.s_offsets[1] = input.s1_offset;
            apply_initial_policy(input);
            input.x_scale = build_x_block_scale_vector<KernelShape>(input.x0, profile_params_for_case(input));
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


    } // namespace

} // namespace veqlib_kernel_api
