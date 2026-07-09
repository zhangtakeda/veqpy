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
    constexpr double weighted_gnqr_weight_floor = 1.0e-2;

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

    struct BoundaryFitSetup
    {
        double              R0;
        double              Z0;
        double              a;
        double              ka;
        std::vector<double> r_points;
        std::vector<double> z_points;
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
                std::reverse(r_ordered.begin() + 1, r_ordered.end());
                std::reverse(z_ordered.begin() + 1, z_ordered.end());
            }
        }
        return {std::move(r_ordered), std::move(z_ordered)};
    }

    inline BoundaryFitSetup prepare_boundary_fit(const double* R,
                                                 const double* Z,
                                                 size_t        r_size,
                                                 size_t        z_size,
                                                 int           c_order,
                                                 int           s_order)
    {
        if (r_size != z_size)
            throw std::runtime_error("R_boundary and Z_boundary must have the same shape");
        if (r_size < 4)
            throw std::runtime_error("R_boundary and Z_boundary must contain at least four points");
        if (c_order < 0 || s_order < 0)
            throw std::runtime_error("c_order and s_order must be non-negative");
        if (c_order > max_boundary_fourier_order || s_order > max_boundary_fourier_order)
            throw std::runtime_error("c_order and s_order exceed the native boundary fitter maximum order");
        const size_t fit_variables = static_cast<size_t>(c_order + 1 + s_order);
        if (r_size < fit_variables)
            throw std::runtime_error("R_boundary/Z_boundary do not contain enough points for QR boundary fitting");
        if (R == nullptr)
            throw std::runtime_error("R_boundary data pointer is null");
        if (Z == nullptr)
            throw std::runtime_error("Z_boundary data pointer is null");
        if (!std::isfinite(R[0]) || !std::isfinite(Z[0]))
            throw std::runtime_error("R_boundary and Z_boundary must contain only finite values");

        double r_min = R[0];
        double r_max = R[0];
        double z_min = Z[0];
        double z_max = Z[0];
        size_t start = 0;
        for (size_t i = 1; i < r_size; ++i)
        {
            if (!std::isfinite(R[i]) || !std::isfinite(Z[i]))
                throw std::runtime_error("R_boundary and Z_boundary must contain only finite values");
            r_min = std::min(r_min, R[i]);
            r_max = std::max(r_max, R[i]);
            if (Z[i] < z_min)
            {
                z_min = Z[i];
                start = i;
            }
            z_max = std::max(z_max, Z[i]);
        }

        const double R0 = 0.5 * (r_max + r_min);
        const double Z0 = 0.5 * (z_max + z_min);
        const double a  = 0.5 * (r_max - r_min);
        if (a <= 0.0)
            throw std::runtime_error("Boundary width must be positive");
        const double ka = std::max(0.5 * (z_max - z_min) / a, 1.0e-6);

        std::vector<double> r_ordered(r_size);
        std::vector<double> z_ordered(r_size);
        for (size_t offset = 0; offset < r_size; ++offset)
        {
            const size_t source = (start + offset) % r_size;
            r_ordered[offset]  = R[source];
            z_ordered[offset]  = Z[source];
        }

        const size_t direction_probe_count = std::min<size_t>(5, r_size - 1);
        double       total                 = 0.0;
        for (size_t i = 1; i <= direction_probe_count; ++i)
            total += r_ordered[i] - r_ordered[0];
        if (total / static_cast<double>(direction_probe_count) > 0.0)
        {
            std::reverse(r_ordered.begin() + 1, r_ordered.end());
            std::reverse(z_ordered.begin() + 1, z_ordered.end());
        }

        return {R0, Z0, a, ka, std::move(r_ordered), std::move(z_ordered)};
    }

    inline double next_phase_candidate(double candidate, double previous) noexcept
    {
        while (candidate < previous - 1.0e-12)
            candidate += two_pi;
        return candidate;
    }

    inline void infer_theta_into(const std::vector<double>& z_points,
                                 double                     z0,
                                 double                     a,
                                 double                     ka,
                                 std::vector<double>&       theta)
    {
        const size_t n = z_points.size();
        theta.resize(n);
        if (n == 0)
            return;
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
    }

    inline std::vector<double> infer_theta(const std::vector<double>& z_points,
                                           double                     z0,
                                           double                     a,
                                           double                     ka)
    {
        std::vector<double> theta;
        infer_theta_into(z_points, z0, a, ka, theta);
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
        const int max_order = std::max(c_order, s_order);
        for (size_t row = 0; row < n_rows; ++row)
        {
            matrix[row * n_cols] = 1.0;
            double sin_base = 0.0;
            double cos_base = 1.0;
            math::relaxed_sincos(theta[row], sin_base, cos_base);
            double sin_order = sin_base;
            double cos_order = cos_base;
            for (int order = 1; order <= max_order; ++order)
            {
                if (order <= c_order)
                    matrix[row * n_cols + static_cast<size_t>(order)] = cos_order;
                if (order <= s_order)
                    matrix[row * n_cols + static_cast<size_t>(c_order + order)] = sin_order;
                const double next_sin = sin_order * cos_base + cos_order * sin_base;
                const double next_cos = cos_order * cos_base - sin_order * sin_base;
                sin_order             = next_sin;
                cos_order             = next_cos;
            }
        }
        return matrix;
    }

    template <typename MatrixValue>
    inline std::vector<double> solve_matrix_qr_from(size_t                     n_rows,
                                                    size_t                     n_cols,
                                                    const std::vector<double>& rhs,
                                                    MatrixValue                matrix_value)
    {
        std::vector<double> q(n_cols * n_rows, 0.0);
        std::vector<double> r(n_cols * n_cols, 0.0);
        std::vector<double> work(n_rows);
        double              diagonal_max = 0.0;

        for (size_t col = 0; col < n_cols; ++col)
        {
            for (size_t row = 0; row < n_rows; ++row)
                work[row] = matrix_value(row, col);
            for (size_t previous_col = 0; previous_col < col; ++previous_col)
            {
                double projection = 0.0;
                for (size_t row = 0; row < n_rows; ++row)
                    projection += q[previous_col * n_rows + row] * work[row];
                r[previous_col * n_cols + col] = projection;
                for (size_t row = 0; row < n_rows; ++row)
                    work[row] -= projection * q[previous_col * n_rows + row];
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
                    q[col * n_rows + row] = work[row] / norm;
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
                value += q[col * n_rows + row] * rhs[row];
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

    inline std::vector<double> solve_matrix_qr(const std::vector<double>& matrix,
                                               const std::vector<double>& rhs,
                                               size_t                     n_rows,
                                               size_t                     n_cols)
    {
        return solve_matrix_qr_from(n_rows, n_cols, rhs, [&](size_t row, size_t col) {
            return matrix[row * n_cols + col];
        });
    }

    inline std::vector<double> solve_row_scaled_matrix_qr(const std::vector<double>& matrix,
                                                          const std::vector<double>& rhs,
                                                          const std::vector<double>& row_scale,
                                                          size_t                     n_rows,
                                                          size_t                     n_cols)
    {
        return solve_matrix_qr_from(n_rows, n_cols, rhs, [&](size_t row, size_t col) {
            return row_scale[row] * matrix[row * n_cols + col];
        });
    }

    inline std::vector<double> solve_phase_projection_qr(const std::vector<double>& theta,
                                                         const std::vector<double>& delta,
                                                         int                        c_order,
                                                         int                        s_order)
    {
        const size_t n_rows = theta.size();
        const size_t n_cols = static_cast<size_t>(c_order + 1 + s_order);
        const auto   matrix = phase_design_matrix(theta, c_order, s_order);
        return solve_matrix_qr(matrix, delta, n_rows, n_cols);
    }

    inline std::vector<double> solve_weighted_phase_projection_qr(const std::vector<double>& theta,
                                                                  const std::vector<double>& theta_bar,
                                                                  const std::vector<double>& matrix,
                                                                  double                     a,
                                                                  size_t                     n_cols)
    {
        const size_t n_rows = theta.size();
        std::vector<double> row_weight(n_rows);
        std::vector<double> weighted_delta(n_rows);
        const double        floor = std::max(a * weighted_gnqr_weight_floor, 0.0);

        for (size_t row = 0; row < n_rows; ++row)
        {
            const double weight = std::max(std::abs(a * math::relaxed_sin(theta_bar[row])), floor);
            row_weight[row]     = weight;
            weighted_delta[row] = (theta_bar[row] - theta[row]) * weight;
        }
        return solve_row_scaled_matrix_qr(matrix, weighted_delta, row_weight, n_rows, n_cols);
    }

    inline double r_objective(const std::vector<double>& r_points,
                              const std::vector<double>& theta,
                              const std::vector<double>& matrix,
                              const std::vector<double>& coefficients,
                              double                     R0,
                              double                     a)
    {
        const size_t n_rows = theta.size();
        const size_t n_cols = coefficients.size();
        double       total  = 0.0;
        for (size_t row = 0; row < n_rows; ++row)
        {
            double theta_bar = theta[row];
            for (size_t col = 0; col < n_cols; ++col)
                theta_bar += matrix[row * n_cols + col] * coefficients[col];
            const double residual = r_points[row] - (R0 + a * math::relaxed_cos(theta_bar));
            total += residual * residual;
        }
        return total / static_cast<double>(std::max<size_t>(n_rows, 1));
    }

    inline bool apply_gnqr_step(std::vector<double>&       coefficients,
                                const std::vector<double>& r_points,
                                const std::vector<double>& theta,
                                const std::vector<double>& matrix,
                                double                     R0,
                                double                     a)
    {
        const size_t n_rows = theta.size();
        const size_t n_cols = coefficients.size();
        std::vector<double> row_scale(n_rows);
        std::vector<double> rhs(n_rows);
        double              residual_total = 0.0;

        for (size_t row = 0; row < n_rows; ++row)
        {
            double theta_bar = theta[row];
            for (size_t col = 0; col < n_cols; ++col)
                theta_bar += matrix[row * n_cols + col] * coefficients[col];

            double sin_theta_bar = 0.0;
            double cos_theta_bar = 1.0;
            math::relaxed_sincos(theta_bar, sin_theta_bar, cos_theta_bar);
            const double residual = r_points[row] - (R0 + a * cos_theta_bar);
            const double scale    = a * sin_theta_bar;
            row_scale[row]        = scale;
            rhs[row]              = -residual;
            residual_total += residual * residual;
        }

        const auto   step              = solve_row_scaled_matrix_qr(matrix, rhs, row_scale, n_rows, n_cols);
        const double current_objective = residual_total / static_cast<double>(std::max<size_t>(n_rows, 1));
        std::vector<double> candidate(n_cols);
        for (double damping : {1.0, 0.5, 0.25, 0.125, 0.0625})
        {
            for (size_t col = 0; col < n_cols; ++col)
                candidate[col] = coefficients[col] + damping * step[col];
            const double candidate_objective = r_objective(r_points, theta, matrix, candidate, R0, a);
            if (candidate_objective < current_objective)
            {
                coefficients = std::move(candidate);
                return true;
            }
        }
        return false;
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
        const int           max_order =
            std::max(static_cast<int>(c_offsets.size()), static_cast<int>(s_offsets.size())) - 1;
        for (size_t row = 0; row < n; ++row)
        {
            double sin_base = 0.0;
            double cos_base = 1.0;
            math::relaxed_sincos(theta[row], sin_base, cos_base);
            double sin_order = sin_base;
            double cos_order = cos_base;
            double theta_bar = theta[row] + c_offsets[0];
            for (int order = 1; order <= max_order; ++order)
            {
                if (static_cast<size_t>(order) < c_offsets.size())
                    theta_bar += c_offsets[static_cast<size_t>(order)] * cos_order;
                if (static_cast<size_t>(order) < s_offsets.size())
                    theta_bar += s_offsets[static_cast<size_t>(order)] * sin_order;
                const double next_sin = sin_order * cos_base + cos_order * sin_base;
                const double next_cos = cos_order * cos_base - sin_order * sin_base;
                sin_order             = next_sin;
                cos_order             = next_cos;
            }
            boundary[row * 2]     = R0 + a * math::relaxed_cos(theta_bar);
            boundary[row * 2 + 1] = Z0 - a * ka * sin_base;
        }
        return boundary;
    }

    inline std::vector<double> build_boundary_from_phase_matrix(double                     R0,
                                                                double                     Z0,
                                                                double                     a,
                                                                double                     ka,
                                                                const std::vector<double>& coefficients,
                                                                const std::vector<double>& theta,
                                                                const std::vector<double>& matrix,
                                                                int                        c_order,
                                                                int                        s_order)
    {
        const size_t n      = theta.size();
        const size_t n_cols = coefficients.size();
        std::vector<double> boundary(n * 2);
        for (size_t row = 0; row < n; ++row)
        {
            double theta_bar = theta[row];
            for (size_t col = 0; col < n_cols; ++col)
                theta_bar += matrix[row * n_cols + col] * coefficients[col];
            double sin_theta_bar = 0.0;
            double cos_theta_bar = 1.0;
            math::relaxed_sincos(theta_bar, sin_theta_bar, cos_theta_bar);
            const double sin_theta =
                s_order > 0 ? matrix[row * n_cols + static_cast<size_t>(c_order + 1)]
                            : math::relaxed_sin(theta[row]);
            boundary[row * 2]     = R0 + a * cos_theta_bar;
            boundary[row * 2 + 1] = Z0 - a * ka * sin_theta;
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
        const size_t n             = r_points.size();
        double       max_distance2 = 0.0;
        for (size_t row = 0; row < n; ++row)
        {
            double best = std::numeric_limits<double>::infinity();
            for (size_t step = 0; step < n; ++step)
            {
                const size_t col       = (row + step) % n;
                const double dr        = r_points[row] - fitted_boundary[col * 2];
                const double dz        = z_points[row] - fitted_boundary[col * 2 + 1];
                const double distance2 = dr * dr + dz * dz;
                if (distance2 < best)
                {
                    best = distance2;
                    if (best <= max_distance2)
                        break;
                }
            }
            max_distance2 = std::max(max_distance2, best);
        }
        for (size_t row = 0; row < n; ++row)
        {
            double best = std::numeric_limits<double>::infinity();
            for (size_t step = 0; step < n; ++step)
            {
                const size_t col       = (row + step) % n;
                const double dr        = fitted_boundary[row * 2] - r_points[col];
                const double dz        = fitted_boundary[row * 2 + 1] - z_points[col];
                const double distance2 = dr * dr + dz * dz;
                if (distance2 < best)
                {
                    best = distance2;
                    if (best <= max_distance2)
                        break;
                }
            }
            max_distance2 = std::max(max_distance2, best);
        }
        return std::sqrt(max_distance2);
    }

    inline BoundaryFitResult fit_boundary_weighted_qr_prepared(BoundaryFitSetup setup,
                                                               int              c_order,
                                                               int              s_order,
                                                               int              gn_steps)
    {
        const auto   theta       = infer_theta(setup.z_points, setup.Z0, setup.a, setup.ka);
        const auto   theta_bar   = infer_theta_bar_target(setup.r_points, theta, setup.R0, setup.a);
        const auto   matrix      = phase_design_matrix(theta, c_order, s_order);
        const size_t coefficient_count = static_cast<size_t>(c_order + 1 + s_order);
        auto         coefficients      =
            solve_weighted_phase_projection_qr(theta, theta_bar, matrix, setup.a, coefficient_count);
        for (int step = 0; step < gn_steps; ++step)
        {
            if (!apply_gnqr_step(coefficients, setup.r_points, theta, matrix, setup.R0, setup.a))
                break;
        }

        auto [c_offsets, s_offsets] = coefficients_to_offsets(coefficients, c_order, s_order);
        const auto fitted_boundary = build_boundary_from_phase_matrix(
            setup.R0,
            setup.Z0,
            setup.a,
            setup.ka,
            coefficients,
            theta,
            matrix,
            c_order,
            s_order);
        const double rms             = rms_r_error(setup.r_points, fitted_boundary);
        const double max_curve_error = max_bidirectional_distance(setup.r_points, setup.z_points, fitted_boundary);
        return {
            setup.R0,
            setup.Z0,
            setup.a,
            setup.ka,
            std::move(c_offsets),
            std::move(s_offsets),
            rms,
            max_curve_error};
    }

    inline BoundaryFitResult fit_boundary_qr_impl(const double* R,
                                                  const double* Z,
                                                  size_t        r_size,
                                                  size_t        z_size,
                                                  int           c_order,
                                                  int           s_order)
    {
        return fit_boundary_weighted_qr_prepared(prepare_boundary_fit(R, Z, r_size, z_size, c_order, s_order),
                                                 c_order,
                                                 s_order,
                                                 0);
    }

    inline BoundaryFitResult fit_boundary_qr_impl(std::vector<double> R,
                                                  std::vector<double> Z,
                                                  int                 c_order,
                                                  int                 s_order)
    {
        return fit_boundary_qr_impl(R.data(), Z.data(), R.size(), Z.size(), c_order, s_order);
    }

    inline BoundaryFitResult fit_boundary_weighted_gnqr_impl(const double* R,
                                                             const double* Z,
                                                             size_t        r_size,
                                                             size_t        z_size,
                                                             int           c_order,
                                                             int           s_order)
    {
        return fit_boundary_weighted_qr_prepared(prepare_boundary_fit(R, Z, r_size, z_size, c_order, s_order),
                                                 c_order,
                                                 s_order,
                                                 2);
    }

    inline BoundaryFitResult fit_boundary_weighted_gnqr_impl(std::vector<double> R,
                                                             std::vector<double> Z,
                                                             int                 c_order,
                                                             int                 s_order)
    {
        return fit_boundary_weighted_gnqr_impl(R.data(), Z.data(), R.size(), Z.size(), c_order, s_order);
    }

    struct LeastSquareState
    {
        double              R0;
        double              Z0;
        double              a;
        double              ka;
        std::vector<double> c_offsets;
        std::vector<double> s_offsets;
    };

    inline std::pair<std::vector<double>, std::vector<double>>
    least_square_bounds(const std::vector<double>& r_points,
                        const std::vector<double>& z_points,
                        double                     initial_a,
                        int                        c_order,
                        int                        s_order)
    {
        const auto [r_min_it, r_max_it] = std::minmax_element(r_points.begin(), r_points.end());
        const auto [z_min_it, z_max_it] = std::minmax_element(z_points.begin(), z_points.end());
        const double r_min              = *r_min_it;
        const double r_max              = *r_max_it;
        const double z_min              = *z_min_it;
        const double z_max              = *z_max_it;
        const double span_r             = r_max - r_min;
        const double span_z             = z_max - z_min;
        const size_t variable_count     = static_cast<size_t>(4 + c_order + 1 + s_order);
        std::vector<double> lower(variable_count, -10.0);
        std::vector<double> upper(variable_count, 10.0);
        lower[0] = r_min - 0.25 * span_r;
        upper[0] = r_max + 0.25 * span_r;
        lower[1] = z_min - 0.25 * span_z;
        upper[1] = z_max + 0.25 * span_z;
        lower[2] = std::max(1.0e-6, 0.25 * initial_a);
        upper[2] = std::max({4.0 * initial_a, span_z, 1.0});
        lower[3] = 1.0e-6;
        upper[3] = 10.0;
        return {std::move(lower), std::move(upper)};
    }

    inline std::vector<double> clip_vector(const std::vector<double>& vector,
                                           const std::vector<double>& lower,
                                           const std::vector<double>& upper)
    {
        std::vector<double> out(vector.size());
        for (size_t i = 0; i < vector.size(); ++i)
            out[i] = std::clamp(vector[i], lower[i], upper[i]);
        return out;
    }

    inline LeastSquareState unpack_least_square_vector(const std::vector<double>& vector,
                                                       int                        c_order,
                                                       int                        s_order,
                                                       bool                       normalize_offsets)
    {
        LeastSquareState state;
        state.R0 = vector[0];
        state.Z0 = vector[1];
        state.a  = vector[2];
        state.ka = vector[3];
        state.c_offsets.resize(static_cast<size_t>(c_order + 1));
        for (int i = 0; i <= c_order; ++i)
            state.c_offsets[static_cast<size_t>(i)] = vector[static_cast<size_t>(4 + i)];
        state.s_offsets.assign(static_cast<size_t>(s_order + 1), 0.0);
        for (int i = 1; i <= s_order; ++i)
            state.s_offsets[static_cast<size_t>(i)] = vector[static_cast<size_t>(4 + c_order + i)];
        if (normalize_offsets)
        {
            state.c_offsets[0] = wrap_pi(state.c_offsets[0]);
            state.s_offsets[0] = 0.0;
        }
        return state;
    }

    inline void least_square_residual_into(const std::vector<double>& vector,
                                           const std::vector<double>& r_points,
                                           const std::vector<double>& z_points,
                                           int                        c_order,
                                           int                        s_order,
                                           std::vector<double>&       residual,
                                           std::vector<double>&       theta)
    {
        const double R0 = vector[0];
        const double Z0 = vector[1];
        const double a  = vector[2];
        const double ka = vector[3];
        infer_theta_into(z_points, Z0, a, ka, theta);
        const size_t n         = r_points.size();
        const int    max_order = std::max(c_order, s_order);
        residual.resize(2 * n);
        for (size_t row = 0; row < n; ++row)
        {
            double sin_base = 0.0;
            double cos_base = 1.0;
            math::relaxed_sincos(theta[row], sin_base, cos_base);
            double sin_order = sin_base;
            double cos_order = cos_base;
            double theta_bar = theta[row] + vector[4];
            for (int order = 1; order <= max_order; ++order)
            {
                if (order <= c_order)
                    theta_bar += vector[static_cast<size_t>(4 + order)] * cos_order;
                if (order <= s_order)
                    theta_bar += vector[static_cast<size_t>(4 + c_order + order)] * sin_order;
                const double next_sin = sin_order * cos_base + cos_order * sin_base;
                const double next_cos = cos_order * cos_base - sin_order * sin_base;
                sin_order             = next_sin;
                cos_order             = next_cos;
            }
            double sin_theta_bar = 0.0;
            double cos_theta_bar = 1.0;
            math::relaxed_sincos(theta_bar, sin_theta_bar, cos_theta_bar);
            const double fitted_r = R0 + a * cos_theta_bar;
            const double fitted_z = Z0 - a * ka * sin_base;
            residual[row]     = r_points[row] - fitted_r;
            residual[n + row] = z_points[row] - fitted_z;
        }
    }

    inline std::vector<double> least_square_residual(const std::vector<double>& vector,
                                                     const std::vector<double>& r_points,
                                                     const std::vector<double>& z_points,
                                                     int                        c_order,
                                                     int                        s_order)
    {
        std::vector<double> residual;
        std::vector<double> theta;
        least_square_residual_into(vector, r_points, z_points, c_order, s_order, residual, theta);
        return residual;
    }

    inline double residual_objective(const std::vector<double>& residual)
    {
        double total = 0.0;
        for (double value : residual)
            total += value * value;
        return total / static_cast<double>(std::max<size_t>(residual.size(), 1));
    }

    inline double residual_rms(const std::vector<double>& residual)
    {
        return std::sqrt(residual_objective(residual));
    }

    inline std::vector<double> bounded_least_square_lm(std::vector<double>       x,
                                                       const std::vector<double>& lower,
                                                       const std::vector<double>& upper,
                                                       const std::vector<double>& r_points,
                                                       const std::vector<double>& z_points,
                                                       int                        c_order,
                                                       int                        s_order,
                                                       int                        max_iterations)
    {
        x                   = clip_vector(x, lower, upper);
        const size_t p      = x.size();
        double       lambda = 1.0e-4;
        std::vector<double> residual;
        std::vector<double> trial_residual;
        std::vector<double> candidate_residual;
        std::vector<double> theta;
        std::vector<double> trial_theta;
        std::vector<double> candidate_theta;
        std::vector<double> trial(x.size());
        std::vector<double> candidate(x.size());
        std::vector<double> augmented_rhs;
        for (int iteration = 0; iteration < max_iterations; ++iteration)
        {
            least_square_residual_into(x, r_points, z_points, c_order, s_order, residual, theta);
            const double current_objective = residual_objective(residual);
            const size_t rows              = residual.size();
            std::vector<double> jacobian(rows * p, 0.0);
            trial = x;
            for (size_t col = 0; col < p; ++col)
            {
                const double step_size = 1.0e-6 * std::max(std::abs(x[col]), 1.0);
                const double previous  = x[col];
                trial[col]             = std::min(upper[col], x[col] + step_size);
                double actual_step     = trial[col] - x[col];
                if (std::abs(actual_step) < 1.0e-14)
                {
                    trial[col]  = std::max(lower[col], x[col] - step_size);
                    actual_step = trial[col] - x[col];
                }
                if (std::abs(actual_step) < 1.0e-14)
                {
                    trial[col] = previous;
                    continue;
                }
                least_square_residual_into(
                    trial,
                    r_points,
                    z_points,
                    c_order,
                    s_order,
                    trial_residual,
                    trial_theta);
                for (size_t row = 0; row < rows; ++row)
                    jacobian[row * p + col] = (trial_residual[row] - residual[row]) / actual_step;
                trial[col] = previous;
            }

            bool accepted = false;
            const size_t augmented_rows = rows + p;
            augmented_rhs.assign(augmented_rows, 0.0);
            for (size_t row = 0; row < rows; ++row)
                augmented_rhs[row] = -residual[row];
            for (int attempt = 0; attempt < 8; ++attempt)
            {
                const double damping = std::sqrt(lambda);
                const auto   step    = solve_matrix_qr_from(
                    augmented_rows,
                    p,
                    augmented_rhs,
                    [&](size_t row, size_t col) {
                        if (row < rows)
                            return jacobian[row * p + col];
                        return (row - rows) == col ? damping : 0.0;
                    });
                double              step_norm = 0.0;
                double              x_norm    = 0.0;
                for (size_t col = 0; col < p; ++col)
                {
                    candidate[col] = std::clamp(x[col] + step[col], lower[col], upper[col]);
                    const double actual = candidate[col] - x[col];
                    step_norm += actual * actual;
                    x_norm += x[col] * x[col];
                }
                least_square_residual_into(
                    candidate,
                    r_points,
                    z_points,
                    c_order,
                    s_order,
                    candidate_residual,
                    candidate_theta);
                const double candidate_objective = residual_objective(candidate_residual);
                if (candidate_objective < current_objective)
                {
                    x        = candidate;
                    lambda   = std::max(lambda * 0.3, 1.0e-12);
                    accepted = true;
                    if (std::sqrt(step_norm) <= 1.0e-9 * (std::sqrt(x_norm) + 1.0))
                        return x;
                    if (current_objective - candidate_objective <= 1.0e-14 * std::max(current_objective, 1.0))
                        return x;
                    break;
                }
                lambda = std::min(lambda * 10.0, 1.0e12);
            }
            if (!accepted)
                return x;
        }
        return x;
    }

    inline BoundaryFitResult fit_boundary_least_square_impl(const double* R,
                                                            const double* Z,
                                                            size_t        r_size,
                                                            size_t        z_size,
                                                            int           c_order,
                                                            int           s_order)
    {
        auto setup                = prepare_boundary_fit(R, Z, r_size, z_size, c_order, s_order);
        auto [lower, upper]       = least_square_bounds(setup.r_points, setup.z_points, setup.a, c_order, s_order);
        std::vector<double> vector(static_cast<size_t>(4 + c_order + 1 + s_order), 0.0);
        vector[0] = setup.R0;
        vector[1] = setup.Z0;
        vector[2] = setup.a;
        vector[3] = setup.ka;
        vector    = bounded_least_square_lm(
            std::move(vector),
            lower,
            upper,
            setup.r_points,
            setup.z_points,
            c_order,
            s_order,
            40);
        auto state = unpack_least_square_vector(vector, c_order, s_order, true);
        const auto theta           = infer_theta(setup.z_points, state.Z0, state.a, state.ka);
        const auto fitted_boundary = build_boundary(
            state.R0,
            state.Z0,
            state.a,
            state.ka,
            state.c_offsets,
            state.s_offsets,
            theta);
        const auto residual        = least_square_residual(vector, setup.r_points, setup.z_points, c_order, s_order);
        const double rms           = residual_rms(residual);
        const double curve_error   = max_bidirectional_distance(setup.r_points, setup.z_points, fitted_boundary);
        return {
            state.R0,
            state.Z0,
            state.a,
            state.ka,
            std::move(state.c_offsets),
            std::move(state.s_offsets),
            rms,
            curve_error};
    }

    inline BoundaryFitResult fit_boundary_least_square_impl(std::vector<double> R,
                                                            std::vector<double> Z,
                                                            int                 c_order,
                                                            int                 s_order)
    {
        return fit_boundary_least_square_impl(R.data(), Z.data(), R.size(), Z.size(), c_order, s_order);
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
        auto result = cxx_boundary_fit::fit_boundary_qr_impl(
            R_boundary.data(),
            Z_boundary.data(),
            static_cast<cxx_boundary_fit::size_t>(R_boundary.shape(0)),
            static_cast<cxx_boundary_fit::size_t>(Z_boundary.shape(0)),
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

    inline nb::dict fit_boundary_weighted_gnqr(nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> R_boundary,
                                               nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> Z_boundary,
                                               int c_order,
                                               int s_order)
    {
        auto result = cxx_boundary_fit::fit_boundary_weighted_gnqr_impl(
            R_boundary.data(),
            Z_boundary.data(),
            static_cast<cxx_boundary_fit::size_t>(R_boundary.shape(0)),
            static_cast<cxx_boundary_fit::size_t>(Z_boundary.shape(0)),
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

    inline nb::dict fit_boundary_least_square(nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> R_boundary,
                                              nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> Z_boundary,
                                              int c_order,
                                              int s_order)
    {
        auto result = cxx_boundary_fit::fit_boundary_least_square_impl(
            R_boundary.data(),
            Z_boundary.data(),
            static_cast<cxx_boundary_fit::size_t>(R_boundary.shape(0)),
            static_cast<cxx_boundary_fit::size_t>(Z_boundary.shape(0)),
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
