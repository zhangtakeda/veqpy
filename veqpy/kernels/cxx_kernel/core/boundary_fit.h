#pragma once

// Native boundary scatter-to-coefficient phase-QR fitter for benchmark comparison.

#include "math.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace cxx_boundary_fit
{
    using std::size_t;

    constexpr int    max_boundary_fourier_order = 20;
    constexpr double pi                         = math::detail::pi;
    constexpr double two_pi                     = 2.0 * math::detail::pi;

    struct BoundaryFitResult
    {
        double              R0;
        double              Z0;
        double              a;
        double              ka;
        std::vector<double> c_offsets;
        std::vector<double> s_offsets;
        double              rms;
        double              max_curve_error;
    };

    inline double wrap_pi(double value) noexcept
    {
        double wrapped = std::fmod(value + pi, two_pi);
        if (wrapped < 0.0)
            wrapped += two_pi;
        return wrapped - pi;
    }

    inline std::pair<std::vector<double>, std::vector<double>>
    ordered_boundary_variant(const std::vector<double>& R, const std::vector<double>& Z)
    {
        const size_t n     = R.size();
        size_t       start = 0;
        double       min_z = Z[0];
        for (size_t i = 1; i < n; ++i)
        {
            if (Z[i] < min_z)
            {
                min_z = Z[i];
                start = i;
            }
        }

        std::vector<double> r_ordered(n);
        std::vector<double> z_ordered(n);
        for (size_t offset = 0; offset < n; ++offset)
        {
            const size_t source = (start + offset) % n;
            r_ordered[offset]  = R[source];
            z_ordered[offset]  = Z[source];
        }

        const size_t direction_probe_count = std::min<size_t>(5, n - 1);
        if (direction_probe_count > 0)
        {
            double total = 0.0;
            for (size_t i = 1; i <= direction_probe_count; ++i)
                total += r_ordered[i] - r_ordered[0];
            if (total / static_cast<double>(direction_probe_count) > 0.0)
            {
                std::vector<double> reversed_r(n);
                std::vector<double> reversed_z(n);
                reversed_r[0] = r_ordered[0];
                reversed_z[0] = z_ordered[0];
                for (size_t i = 1; i < n; ++i)
                {
                    const size_t source = n - i;
                    reversed_r[i]       = r_ordered[source];
                    reversed_z[i]       = z_ordered[source];
                }
                r_ordered = std::move(reversed_r);
                z_ordered = std::move(reversed_z);
            }
        }
        return {std::move(r_ordered), std::move(z_ordered)};
    }

    inline double next_phase_candidate(double candidate, double previous) noexcept
    {
        while (candidate < previous - 1.0e-12)
            candidate += two_pi;
        return candidate;
    }

    inline std::vector<double> infer_theta(const std::vector<double>& z_points,
                                           double                     z0,
                                           double                     a,
                                           double                     ka)
    {
        const size_t        n = z_points.size();
        std::vector<double> theta(n);
        theta[0]        = 0.5 * pi;
        double previous = theta[0];
        const double step  = two_pi / static_cast<double>(std::max<size_t>(n, 1));
        const double scale = a * std::max(ka, 1.0e-6);

        for (size_t i = 1; i < n; ++i)
        {
            double sin_theta = -(z_points[i] - z0) / scale;
            sin_theta        = std::clamp(sin_theta, -1.0, 1.0);
            const double alpha      = std::asin(sin_theta);
            const double candidate0 = next_phase_candidate(alpha, previous);
            const double candidate1 = next_phase_candidate(pi - alpha, previous);
            const double candidate2 = candidate0 + two_pi;
            const double candidate3 = candidate1 + two_pi;
            const double target     = previous + step;

            double best          = candidate0;
            double best_distance = std::abs(candidate0 - target);
            double distance      = std::abs(candidate1 - target);
            if (distance < best_distance)
            {
                best          = candidate1;
                best_distance = distance;
            }
            distance = std::abs(candidate2 - target);
            if (distance < best_distance)
            {
                best          = candidate2;
                best_distance = distance;
            }
            distance = std::abs(candidate3 - target);
            if (distance < best_distance)
                best = candidate3;
            theta[i] = best;
            previous = best;
        }
        return theta;
    }

    inline std::vector<double> infer_theta_bar_target(const std::vector<double>& r_points,
                                                      const std::vector<double>& theta,
                                                      double                     R0,
                                                      double                     a)
    {
        if (a <= 0.0)
            throw std::runtime_error("Boundary width must be positive");

        const size_t        n = r_points.size();
        std::vector<double> target(n);
        double              max_excess = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double cosine_target = (r_points[i] - R0) / a;
            max_excess           = std::max(max_excess, std::abs(cosine_target) - 1.0);
            cosine_target        = std::clamp(cosine_target, -1.0, 1.0);
            const double principal = std::acos(cosine_target);
            const double positive =
                principal + two_pi * std::nearbyint((theta[i] - principal) / two_pi);
            const double negative =
                -principal + two_pi * std::nearbyint((theta[i] + principal) / two_pi);
            target[i] = std::abs(positive - theta[i]) <= std::abs(negative - theta[i]) ? positive
                                                                                       : negative;
        }
        if (max_excess > 1.0e-8)
            throw std::runtime_error("R_boundary values are inconsistent with R0/a for phase QR fitting");

        std::vector<double> unwrapped(n);
        unwrapped[0] = target[0];
        for (size_t i = 1; i < n; ++i)
        {
            double current  = target[i];
            double previous = unwrapped[i - 1];
            while (current - previous > pi)
                current -= two_pi;
            while (current - previous < -pi)
                current += two_pi;
            unwrapped[i] = current;
        }

        double mean_delta = 0.0;
        for (size_t i = 0; i < n; ++i)
            mean_delta += theta[i] - unwrapped[i];
        mean_delta /= static_cast<double>(std::max<size_t>(n, 1));
        const double shift = two_pi * std::nearbyint(mean_delta / two_pi);
        for (double& value : unwrapped)
            value += shift;
        return unwrapped;
    }

    inline std::vector<double> phase_design_matrix(const std::vector<double>& theta, int c_order, int s_order)
    {
        const size_t n_rows = theta.size();
        const size_t n_cols = static_cast<size_t>(c_order + 1 + s_order);
        std::vector<double> matrix(n_rows * n_cols);
        for (size_t row = 0; row < n_rows; ++row)
            matrix[row * n_cols] = 1.0;
        size_t col = 1;
        for (int order = 1; order <= c_order; ++order)
        {
            for (size_t row = 0; row < n_rows; ++row)
                matrix[row * n_cols + col] = math::relaxed_cos(static_cast<double>(order) * theta[row]);
            ++col;
        }
        for (int order = 1; order <= s_order; ++order)
        {
            for (size_t row = 0; row < n_rows; ++row)
                matrix[row * n_cols + col] = math::relaxed_sin(static_cast<double>(order) * theta[row]);
            ++col;
        }
        return matrix;
    }

    inline std::vector<double> solve_phase_projection_qr(const std::vector<double>& theta,
                                                         const std::vector<double>& delta,
                                                         int                        c_order,
                                                         int                        s_order)
    {
        const size_t n_rows = theta.size();
        const size_t n_cols = static_cast<size_t>(c_order + 1 + s_order);
        const auto   matrix = phase_design_matrix(theta, c_order, s_order);
        std::vector<double> q(n_rows * n_cols, 0.0);
        std::vector<double> r(n_cols * n_cols, 0.0);
        double              diagonal_max = 0.0;

        for (size_t col = 0; col < n_cols; ++col)
        {
            std::vector<double> work(n_rows);
            for (size_t row = 0; row < n_rows; ++row)
                work[row] = matrix[row * n_cols + col];
            for (size_t previous_col = 0; previous_col < col; ++previous_col)
            {
                double projection = 0.0;
                for (size_t row = 0; row < n_rows; ++row)
                    projection += q[row * n_cols + previous_col] * work[row];
                r[previous_col * n_cols + col] = projection;
                for (size_t row = 0; row < n_rows; ++row)
                    work[row] -= projection * q[row * n_cols + previous_col];
            }
            double norm = 0.0;
            for (double value : work)
                norm += value * value;
            norm                 = std::sqrt(norm);
            r[col * n_cols + col] = norm;
            diagonal_max          = std::max(diagonal_max, norm);
            if (norm > 0.0)
            {
                for (size_t row = 0; row < n_rows; ++row)
                    q[row * n_cols + col] = work[row] / norm;
            }
        }

        const double tolerance = std::numeric_limits<double>::epsilon() *
                                 static_cast<double>(std::max(n_rows, n_cols)) *
                                 std::max(diagonal_max, 1.0);
        for (size_t col = 0; col < n_cols; ++col)
        {
            if (std::abs(r[col * n_cols + col]) <= tolerance)
                throw std::runtime_error("R_boundary/Z_boundary do not provide full-rank phase QR fitting data");
        }

        std::vector<double> y(n_cols);
        for (size_t col = 0; col < n_cols; ++col)
        {
            double value = 0.0;
            for (size_t row = 0; row < n_rows; ++row)
                value += q[row * n_cols + col] * delta[row];
            y[col] = value;
        }

        std::vector<double> x(n_cols);
        for (size_t row = n_cols; row-- > 0;)
        {
            double value = y[row];
            for (size_t col = row + 1; col < n_cols; ++col)
                value -= r[row * n_cols + col] * x[col];
            x[row] = value / r[row * n_cols + row];
        }
        return x;
    }

    inline std::pair<std::vector<double>, std::vector<double>>
    coefficients_to_offsets(const std::vector<double>& coefficients, int c_order, int s_order)
    {
        std::vector<double> c_offsets(static_cast<size_t>(c_order + 1));
        std::vector<double> s_offsets(static_cast<size_t>(s_order + 1), 0.0);
        for (int i = 0; i <= c_order; ++i)
            c_offsets[static_cast<size_t>(i)] = coefficients[static_cast<size_t>(i)];
        for (int i = 1; i <= s_order; ++i)
            s_offsets[static_cast<size_t>(i)] = coefficients[static_cast<size_t>(c_order + i)];
        c_offsets[0] = wrap_pi(c_offsets[0]);
        s_offsets[0] = 0.0;
        return {std::move(c_offsets), std::move(s_offsets)};
    }

    inline std::vector<double> build_boundary(double                     R0,
                                              double                     Z0,
                                              double                     a,
                                              double                     ka,
                                              const std::vector<double>& c_offsets,
                                              const std::vector<double>& s_offsets,
                                              const std::vector<double>& theta)
    {
        const size_t        n = theta.size();
        std::vector<double> boundary(n * 2);
        for (size_t row = 0; row < n; ++row)
        {
            double theta_bar = theta[row] + c_offsets[0];
            for (size_t order = 1; order < c_offsets.size(); ++order)
                theta_bar += c_offsets[order] * math::relaxed_cos(static_cast<double>(order) * theta[row]);
            for (size_t order = 1; order < s_offsets.size(); ++order)
                theta_bar += s_offsets[order] * math::relaxed_sin(static_cast<double>(order) * theta[row]);
            boundary[row * 2]     = R0 + a * math::relaxed_cos(theta_bar);
            boundary[row * 2 + 1] = Z0 - a * ka * math::relaxed_sin(theta[row]);
        }
        return boundary;
    }

    inline double rms_r_error(const std::vector<double>& r_points, const std::vector<double>& fitted_boundary)
    {
        double total = 0.0;
        for (size_t row = 0; row < r_points.size(); ++row)
        {
            const double diff = r_points[row] - fitted_boundary[row * 2];
            total += diff * diff;
        }
        return std::sqrt(total / static_cast<double>(std::max<size_t>(r_points.size(), 1)));
    }

    inline double max_bidirectional_distance(const std::vector<double>& r_points,
                                             const std::vector<double>& z_points,
                                             const std::vector<double>& fitted_boundary)
    {
        const size_t n            = r_points.size();
        double       max_distance = 0.0;
        for (size_t row = 0; row < n; ++row)
        {
            double best = std::numeric_limits<double>::infinity();
            for (size_t col = 0; col < n; ++col)
            {
                const double dr        = r_points[row] - fitted_boundary[col * 2];
                const double dz        = z_points[row] - fitted_boundary[col * 2 + 1];
                const double distance2 = dr * dr + dz * dz;
                if (distance2 < best)
                    best = distance2;
            }
            max_distance = std::max(max_distance, std::sqrt(best));
        }
        for (size_t row = 0; row < n; ++row)
        {
            double best = std::numeric_limits<double>::infinity();
            for (size_t col = 0; col < n; ++col)
            {
                const double dr        = fitted_boundary[row * 2] - r_points[col];
                const double dz        = fitted_boundary[row * 2 + 1] - z_points[col];
                const double distance2 = dr * dr + dz * dz;
                if (distance2 < best)
                    best = distance2;
            }
            max_distance = std::max(max_distance, std::sqrt(best));
        }
        return max_distance;
    }

    inline BoundaryFitResult fit_boundary_qr_impl(std::vector<double> R,
                                                  std::vector<double> Z,
                                                  int                 c_order,
                                                  int                 s_order)
    {
        if (R.size() != Z.size())
            throw std::runtime_error("R_boundary and Z_boundary must have the same shape");
        if (R.size() < 4)
            throw std::runtime_error("R_boundary and Z_boundary must contain at least four points");
        if (c_order < 0 || s_order < 0)
            throw std::runtime_error("c_order and s_order must be non-negative");
        if (c_order > max_boundary_fourier_order || s_order > max_boundary_fourier_order)
            throw std::runtime_error("c_order and s_order exceed the native boundary fitter maximum order");
        const size_t fit_variables = static_cast<size_t>(c_order + 1 + s_order);
        if (R.size() < fit_variables)
            throw std::runtime_error("R_boundary/Z_boundary do not contain enough points for QR boundary fitting");
        for (size_t i = 0; i < R.size(); ++i)
        {
            if (!std::isfinite(R[i]) || !std::isfinite(Z[i]))
                throw std::runtime_error("R_boundary and Z_boundary must contain only finite values");
        }

        const auto [r_min_it, r_max_it] = std::minmax_element(R.begin(), R.end());
        const auto [z_min_it, z_max_it] = std::minmax_element(Z.begin(), Z.end());
        const double R0                 = 0.5 * (*r_max_it + *r_min_it);
        const double Z0                 = 0.5 * (*z_max_it + *z_min_it);
        const double a                  = 0.5 * (*r_max_it - *r_min_it);
        if (a <= 0.0)
            throw std::runtime_error("Boundary width must be positive");
        const double ka = std::max(0.5 * (*z_max_it - *z_min_it) / a, 1.0e-6);

        auto [r_points, z_points] = ordered_boundary_variant(R, Z);
        const auto theta          = infer_theta(z_points, Z0, a, ka);
        const auto theta_bar      = infer_theta_bar_target(r_points, theta, R0, a);
        std::vector<double> delta(theta.size());
        for (size_t i = 0; i < theta.size(); ++i)
            delta[i] = theta_bar[i] - theta[i];
        const auto coefficients              = solve_phase_projection_qr(theta, delta, c_order, s_order);
        auto [c_offsets, s_offsets]          = coefficients_to_offsets(coefficients, c_order, s_order);
        const auto fitted_boundary           = build_boundary(R0, Z0, a, ka, c_offsets, s_offsets, theta);
        const double rms                     = rms_r_error(r_points, fitted_boundary);
        const double max_curve_error         = max_bidirectional_distance(r_points, z_points, fitted_boundary);
        return {R0, Z0, a, ka, std::move(c_offsets), std::move(s_offsets), rms, max_curve_error};
    }

    inline std::vector<double> vector_from_array(nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> values,
                                                 const char* name)
    {
        const size_t        n = static_cast<size_t>(values.shape(0));
        std::vector<double> out(n);
        const double*       ptr = values.data();
        if (ptr == nullptr)
            throw std::runtime_error(std::string{name} + " data pointer is null");
        for (size_t i = 0; i < n; ++i)
            out[i] = ptr[i];
        return out;
    }

    inline nb::list list_from_vector(const std::vector<double>& values)
    {
        nb::list out;
        for (double value : values)
            out.append(value);
        return out;
    }
}

namespace cxx_python
{
    inline nb::dict fit_boundary_qr(nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> R_boundary,
                                    nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> Z_boundary,
                                    int c_order,
                                    int s_order)
    {
        auto R      = cxx_boundary_fit::vector_from_array(R_boundary, "R_boundary");
        auto Z      = cxx_boundary_fit::vector_from_array(Z_boundary, "Z_boundary");
        auto result = cxx_boundary_fit::fit_boundary_qr_impl(
            std::move(R),
            std::move(Z),
            c_order,
            s_order);

        nb::dict payload;
        payload["R0"]              = result.R0;
        payload["Z0"]              = result.Z0;
        payload["a"]               = result.a;
        payload["ka"]              = result.ka;
        payload["c_offsets"]       = cxx_boundary_fit::list_from_vector(result.c_offsets);
        payload["s_offsets"]       = cxx_boundary_fit::list_from_vector(result.s_offsets);
        payload["rms"]             = result.rms;
        payload["max_curve_error"] = result.max_curve_error;
        payload["c_order"]         = c_order;
        payload["s_order"]         = s_order;
        return payload;
    }
}
