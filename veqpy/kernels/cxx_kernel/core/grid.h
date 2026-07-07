#pragma once

// Grid tables and radial/poloidal helper views for generated Cxx Kernel artifacts.

#include "linalg.h"
#include "math.h"
#include "tensor.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>

namespace grid::detail
{
    using linalg::Thomas;
    using linalg::solve;
    using linalg::transpose;
    using math::clamp;
    using math::cos;
    using math::sin;
    using std::array;
    using std::min;
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

    inline constexpr double pi = 3.141592653589793238462643383279502884;

    // GST theta-correction coefficients. The expansion shape follows
    // A. Gil, J. Segura, and N. M. Temme, "Noniterative Computation of
    // Gauss--Jacobi Quadrature", SIAM J. Sci. Comput. 41(1), A668-A693, 2019.
    // The Chebyshev coefficients of P_1..P_6 are fitted offline for
    // the only Jacobi families we instantiate: Legendre, Lobatto interior,
    // and right-Radau interior. The generator is tools/gst_jacobi_fit_scan.py.
    struct GST6
    {
        Vector<double, 2>  theta1;
        Vector<double, 4>  theta2;
        Vector<double, 6>  theta3;
        Vector<double, 8>  theta4;
        Vector<double, 10> theta5;
        Vector<double, 12> theta6;

        template <size_t Order>
        constexpr const Vector<double, 2 * Order>& theta() const
        {
            static_assert(Order >= 1 && Order <= 6, "GST order must be between 1 and 6");

            if constexpr (Order == 1)
                return theta1;
            else if constexpr (Order == 2)
                return theta2;
            else if constexpr (Order == 3)
                return theta3;
            else if constexpr (Order == 4)
                return theta4;
            else if constexpr (Order == 5)
                return theta5;
            else // if constexpr (Order == 6)
                return theta6;
        }
    };

    template <int Alpha, int Beta>
    extern const GST6 gst6;

    struct LegendreValueDerivative
    {
        double value;
        double derivative;
    };

    // Legendre (Alpha=0, Beta=0) GST coefficients
    template <>
    inline constexpr GST6 gst6<0, 0> = {
        {
            +2.62922115540628349e-13,
            +1.24999998077783003e-01,
        },
        {
            -1.67880333571052457e-10,
            -8.20287576268427659e-02,
            +1.18485871693128521e-10,
            +1.30239272345437137e-03,
        },
        {
            +5.13264520942869408e-08,
            +2.25683227667133834e-01,
            -5.72112160540380265e-08,
            +1.91889940323391808e-02,
            +1.60957924557175103e-08,
            +1.39002549583454722e-04,
        },
        {
            -7.53883065094421954e-06,
            -1.40138875223484183e+00,
            +1.05505120793253059e-05,
            -2.47335310178969842e-01,
            -4.41688914031522416e-06,
            -1.27371652890125378e-02,
            +7.28143805138574424e-07,
            +1.52206840762467740e-03,
        },
        {
            +5.23812836155413699e-04,
            +1.02794642128320959e+01,
            -8.21876020647837745e-04,
            +1.90749420881598652e+00,
            +4.16068380964062557e-04,
            +7.43791247353873475e-01,
            -1.18268556675459703e-04,
            -1.98989672465992029e-01,
            +1.47630924768917304e-05,
            +2.78601208360669296e-02,
        },
        {
            -1.36468002415538556e-02,
            -3.72726369412698801e+01,
            +2.26714749200839644e-02,
            +8.30174852645884798e+00,
            -1.29695502449661305e-02,
            -1.63082668771281440e+01,
            +4.85211985189163125e-03,
            +7.04100111692478325e+00,
            -1.07578623459624175e-03,
            -1.80597842091388761e+00,
            +1.07211117821163972e-04,
            +1.99795219826384485e-01,
        },
    };

    // Lobatto interior (Alpha=1, Beta=1) GST coefficients
    template <>
    inline constexpr GST6 gst6<1, 1> = {
        {
            +3.81807368165373049e-13,
            -3.74999999591663746e-01,
        },
        {
            -2.34408643619658878e-10,
            +1.17180968060242724e-02,
            +8.14683866805553685e-11,
            +1.17187176317809887e-02,
        },
        {
            +6.08327303714616795e-08,
            -2.10624838596251668e-01,
            -2.80032365560578054e-08,
            -1.90028997277063849e-02,
            +1.38796477591955736e-08,
            -2.97890253817478790e-04,
        },
        {
            -6.96215454816339424e-06,
            +1.35615828483294543e+00,
            +4.64980113466689532e-06,
            +2.62468491233217416e-01,
            -3.21026670303320667e-06,
            +5.45359073682480811e-03,
            +4.40833941179735852e-07,
            -1.78675740841767428e-04,
        },
        {
            +3.54196285566095925e-04,
            -1.25466962828794575e+01,
            -3.49670722388500531e-04,
            -3.39624218765980679e+00,
            +2.42530077340860194e-04,
            -2.61788693298515507e-01,
            -6.07114363582356525e-05,
            +2.16308245213416711e-02,
            +7.81832085427816627e-06,
            -2.78223156658923259e-03,
        },
        {
            -6.65278041288836549e-03,
            +6.84435981555447626e+01,
            +8.96386677053716795e-03,
            +2.21569424401738821e+01,
            -5.91352616705825319e-03,
            +3.78579114622241342e+00,
            +2.07625747144089693e-03,
            -5.56167547586908895e-01,
            -4.68994086474730455e-04,
            +1.45951062553165867e-01,
            +4.66211519026683068e-05,
            -1.60034943710405560e-02,
        },
    };

    // Right-Radau interior (Alpha=1, Beta=0) GST coefficients
    template <>
    inline constexpr GST6 gst6<1, 0> = {
        {
            -2.49999999543023294e-01,
            -1.25000000815720769e-01,
        },
        {
            +3.64574872104807476e-02,
            -2.73427768127048228e-02,
            +1.56243506127982535e-02,
            -1.30195536287086707e-03,
        },
        {
            -1.41060876640776878e-01,
            +5.88338185752547860e-03,
            -9.50434991934850232e-02,
            +1.72393642740906509e-03,
            -1.40046935526629088e-03,
            -8.25132606888526806e-05,
        },
        {
            +8.44842663373256419e-01,
            -1.29341806965954306e-02,
            +7.44113901030439284e-01,
            -6.66322820578807494e-04,
            +5.66451988788038707e-02,
            -2.65972309612134928e-03,
            -5.44010861123311266e-04,
            +4.86522335054253251e-04,
        },
        {
            -6.99630070949836291e+00,
            -1.28716946547434885e+00,
            -6.37566860144910308e+00,
            -6.24853114982223512e-01,
            -1.26920972198683324e+00,
            +1.59861527210749144e-01,
            +6.99244237098492483e-02,
            -6.15493113174675130e-02,
            -1.08147854141870062e-02,
            +8.74487573099161884e-03,
        },
        {
            +3.71276054566577827e+01,
            +1.78656423242128817e+01,
            +1.77453599687873300e+01,
            +1.21457377460204743e+01,
            +1.53469321895356590e+01,
            -3.52631966094934368e+00,
            -2.93102801254098422e+00,
            +1.99034293705840515e+00,
            +7.25516172014770588e-01,
            -5.06412492502739786e-01,
            -7.29231769807240499e-02,
            +5.56097413327007115e-02,
        },
    };

    template <size_t Order>
    constexpr double gst_term(const GST6& coefficients, const double* chebyshev, double kpow, double spow)
    {
        static_assert(Order >= 1 && Order <= 6, "GST theta coefficient order is out of range");

        const auto& order_coefficients = coefficients.template theta<Order>();
        double      numerator          = 0.0;
        for (size_t i = 0; i < order_coefficients.count; ++i)
            numerator += order_coefficients[i] * chebyshev[i];
        return numerator / (kpow * spow);
    }

    template <int Alpha, int Beta>
    constexpr double gst_jacobi_root(size_t degree, size_t index, const GST6& coefficients)
    {
        const double nd          = static_cast<double>(degree);
        const double alpha       = static_cast<double>(Alpha);
        const double beta        = static_cast<double>(Beta);
        const double root_number = static_cast<double>(degree - index);
        const double kappa       = nd + 0.5 * (alpha + beta + 1.0);
        const double theta0      = pi * (root_number + 0.5 * alpha - 0.25) / kappa;
        const double x0          = cos(theta0);
        const double sin0        = sin(theta0);

        // GST-style explicit roots:
        // theta = theta0 + sum_j P_j(x0)/(kappa^(2j) * sin(theta0)^(2j-1)).
        // P_j is evaluated from the instantiated Chebyshev coefficient vectors above.
        double chebyshev[12];
        chebyshev[0] = 1.0;
        chebyshev[1] = x0;
        for (size_t i = 2; i < 12; ++i)
            chebyshev[i] = 2.0 * x0 * chebyshev[i - 1] - chebyshev[i - 2];

        const double kappa2 = kappa * kappa;
        const double sin2   = sin0 * sin0;
        double       kpow   = kappa2;
        double       spow   = sin0;
        double       delta  = gst_term<1>(coefficients, chebyshev, kpow, spow);

        kpow *= kappa2;
        spow *= sin2;
        delta += gst_term<2>(coefficients, chebyshev, kpow, spow);

        kpow *= kappa2;
        spow *= sin2;
        delta += gst_term<3>(coefficients, chebyshev, kpow, spow);

        kpow *= kappa2;
        spow *= sin2;
        delta += gst_term<4>(coefficients, chebyshev, kpow, spow);

        kpow *= kappa2;
        spow *= sin2;
        delta += gst_term<5>(coefficients, chebyshev, kpow, spow);

        kpow *= kappa2;
        spow *= sin2;
        delta += gst_term<6>(coefficients, chebyshev, kpow, spow);

        return clamp(cos(theta0 + delta), -1.0, 1.0);
    }

    template <size_t Degree, int Alpha, int Beta>
    constexpr Vector<double, Degree> gst_jacobi_roots()
    {
        const auto&            coefficients = gst6<Alpha, Beta>;
        Vector<double, Degree> roots{uninitialized};
        for (size_t i = 0; i < Degree; ++i)
            roots[i] = gst_jacobi_root<Alpha, Beta>(Degree, i, coefficients);

        return roots;
    }

    template <size_t Degree, int Alpha, int Beta>
    constexpr Vector<double, Degree> symmetric_gst_jacobi_roots()
    {
        const auto&            coefficients = gst6<Alpha, Beta>;
        Vector<double, Degree> roots{uninitialized};

        constexpr size_t half = Degree / 2;
        for (size_t i = 0; i < half; ++i)
        {
            const double root     = gst_jacobi_root<Alpha, Beta>(Degree, i, coefficients);
            roots[i]              = root;
            roots[Degree - 1 - i] = -root;
        }

        if constexpr (Degree % 2 == 1)
            roots[half] = 0.0;

        return roots;
    }

    constexpr double unit_interval(double x) noexcept { return 0.5 * (x + 1.0); }

    template <size_t N>
    constexpr void normalize_unit_weights(Vector<double, N>& weights)
    {
        double total = 0.0;
        for (size_t i = 0; i < N; ++i)
            total += weights[i];

        for (size_t i = 0; i < N; ++i)
            weights[i] /= total;
    }

    constexpr double legendre_value(size_t degree, double x)
    {
        if (degree == 0)
            return 1.0;

        double previous = 1.0;
        double current  = x;
        for (size_t k = 2; k <= degree; ++k)
        {
            const double kd   = static_cast<double>(k);
            const double next = ((2.0 * kd - 1.0) * x * current - (kd - 1.0) * previous) / kd;
            previous          = current;
            current           = next;
        }
        return current;
    }

    constexpr LegendreValueDerivative legendre_value_derivative(size_t degree, double x)
    {
        if (degree == 0)
            return {1.0, 0.0};

        double previous = 1.0;
        double current  = x;
        for (size_t k = 2; k <= degree; ++k)
        {
            const double kd   = static_cast<double>(k);
            const double next = ((2.0 * kd - 1.0) * x * current - (kd - 1.0) * previous) / kd;
            previous          = current;
            current           = next;
        }

        const double nd         = static_cast<double>(degree);
        const double derivative = nd * (previous - x * current) / (1.0 - x * x);
        return {current, derivative};
    }

    // Fejer-I weights integrate ordinary functions on [-1, 1] over the open
    // Chebyshev points used by the native "chebyshev" quadrature option.
    constexpr double fejer1_unit_weight(size_t point_index, size_t count)
    {
        const double theta = pi * (2.0 * static_cast<double>(point_index) + 1.0) / (2.0 * static_cast<double>(count));
        double       correction = 0.0;
        for (size_t j = 1; j <= count / 2; ++j)
        {
            const double jd = static_cast<double>(j);
            correction += cos(2.0 * jd * theta) / (4.0 * jd * jd - 1.0);
        }
        return (1.0 - 2.0 * correction) / static_cast<double>(count);
    }

    template <size_t Degree>
    constexpr Vector<double, Degree> legendre_roots()
    {
        return symmetric_gst_jacobi_roots<Degree, 0, 0>();
    }

    template <size_t N>
    constexpr Vector<double, N - 2> lobatto_interior_roots()
    {
        static_assert(N >= 3, "lobatto interior roots require at least three points");
        return symmetric_gst_jacobi_roots<N - 2, 1, 1>();
    }

    template <size_t N>
    constexpr Vector<double, N - 1> radau_right_roots()
    {
        static_assert(N >= 2, "radau right roots require at least two points");

        return gst_jacobi_roots<N - 1, 1, 0>();
    }

    struct ValueDerivative
    {
        double value;
        double derivative;
    };

    constexpr double alternating_sign(size_t index) noexcept { return index % 2 == 0 ? 1.0 : -1.0; }

    constexpr double jacobi(size_t degree, double alpha, double beta, double x)
    {
        if (degree == 0)
            return 1.0;

        double previous = 1.0;
        double current  = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);

        for (size_t k = 2; k <= degree; ++k)
        {
            const double kd       = static_cast<double>(k);
            const double two_k_ab = 2.0 * kd + alpha + beta;
            const double denom    = 2.0 * kd * (kd + alpha + beta) * (two_k_ab - 2.0);
            const double x_coeff  = (two_k_ab - 1.0) * (two_k_ab * (two_k_ab - 2.0) * x + alpha * alpha - beta * beta);
            const double previous_coeff = 2.0 * (kd + alpha - 1.0) * (kd + beta - 1.0) * two_k_ab;
            const double next           = (x_coeff * current - previous_coeff * previous) / denom;
            previous                    = current;
            current                     = next;
        }

        return current;
    }

    constexpr double jacobi_derivative(size_t degree, double alpha, double beta, double x)
    {
        if (degree == 0)
            return 0.0;

        return 0.5 * (static_cast<double>(degree) + alpha + beta + 1.0) *
               jacobi(degree - 1, alpha + 1.0, beta + 1.0, x);
    }

    constexpr ValueDerivative jacobi_value_derivative(size_t degree, double alpha, double beta, double x)
    {
        return {jacobi(degree, alpha, beta, x), jacobi_derivative(degree, alpha, beta, x)};
    }

    constexpr double jacobi_at_one(size_t degree, double alpha)
    {
        double value = 1.0;
        for (size_t k = 1; k <= degree; ++k)
            value *= (alpha + static_cast<double>(k)) / static_cast<double>(k);
        return value;
    }

    constexpr double legendre(size_t degree, double x) { return jacobi(degree, 0.0, 0.0, x); }

    constexpr double legendre_derivative(size_t degree, double x) { return jacobi_derivative(degree, 0.0, 0.0, x); }

    constexpr double unit_to_legendre_x(double node) noexcept { return 2.0 * node - 1.0; }

    template <size_t N>
    constexpr Matrix<double, N, N + 1> legendre_table(const Vector<double, N>& nodes)
    {
        Matrix<double, N, N + 1> table{uninitialized};
        auto                     table_data = table.data();
        for (size_t row = 0; row < N; ++row)
        {
            const double x         = unit_to_legendre_x(nodes[row]);
            double*      table_row = table_data + row * (N + 1);
            table_row[0]           = 1.0;
            table_row[1]           = x;

            for (size_t degree = 2; degree <= N; ++degree)
            {
                const double nd = static_cast<double>(degree);
                table_row[degree] =
                    ((2.0 * nd - 1.0) * x * table_row[degree - 1] - (nd - 1.0) * table_row[degree - 2]) / nd;
            }
        }
        return table;
    }

    template <size_t N>
    constexpr Matrix<double, N, N> legendre_vandermonde(const Vector<double, N>& nodes)
    {
        const auto           table      = legendre_table(nodes);
        const auto           table_data = table.data();
        Matrix<double, N, N> out{uninitialized};
        auto                 out_data = out.data();
        for (size_t row = 0; row < N; ++row)
            for (size_t col = 0; col < N; ++col)
                out_data[row * N + col] = table_data[row * (N + 1) + col];
        return out;
    }

    template <size_t N>
    constexpr Matrix<double, N, N> integrated_legendre_basis(const Vector<double, N>& nodes)
    {
        const auto           table      = legendre_table(nodes);
        const auto           table_data = table.data();
        Matrix<double, N, N> out{uninitialized};
        auto                 out_data = out.data();

        for (size_t row = 0; row < N; ++row)
        {
            const double  x         = unit_to_legendre_x(nodes[row]);
            double*       out_row   = out_data + row * N;
            const double* table_row = table_data + row * (N + 1);
            out_row[0]              = 0.5 * (x + 1.0);

            for (size_t degree = 1; degree < N; ++degree)
            {
                const double denom = 2.0 * static_cast<double>(degree) + 1.0;
                out_row[degree]    = 0.5 * (table_row[degree + 1] - table_row[degree - 1]) / denom;
            }
        }

        return out;
    }

    template <size_t N, bool UseLobattoTopNorm>
    constexpr double orthogonal_accumulator_entry(
        const double* table_data, const double* integrated_data, size_t row, size_t col, double quadrature_weight)
    {
        double total = 0.0;
        for (size_t degree = 0; degree < N; ++degree)
        {
            double norm = 1.0 / (2.0 * static_cast<double>(degree) + 1.0);
            if constexpr (UseLobattoTopNorm)
            {
                if (degree == N - 1)
                    norm = 1.0 / static_cast<double>(N - 1);
            }
            total += integrated_data[row * N + degree] * table_data[col * (N + 1) + degree] / norm;
        }
        return quadrature_weight * total;
    }

    template <size_t N, bool UseLobattoTopNorm>
    constexpr Matrix<double, N, N> orthogonal_accumulator(const Vector<double, N>& nodes,
                                                          const Vector<double, N>& quadrature_weights)
    {
        const auto           table           = legendre_table(nodes);
        const auto           integrated      = integrated_legendre_basis(nodes);
        const auto           table_data      = table.data();
        const auto           integrated_data = integrated.data();
        Matrix<double, N, N> out{uninitialized};
        auto                 out_data = out.data();
        for (size_t row = 0; row < N; ++row)
            for (size_t col = 0; col < N; ++col)
                out_data[row * N + col] = orthogonal_accumulator_entry<N, UseLobattoTopNorm>(
                    table_data, integrated_data, row, col, quadrature_weights[col]);

        return out;
    }

    // The accumulator entries are first-order Birkhoff/PSIM basis values:
    // A[i,j] = integral_0^x_i l_j(t) dt. Symmetric Legendre and Lobatto
    // nodes satisfy A[i,j] = w[j] - A[N-1-i,N-1-j], so only half the rows
    // need the modal sum. PSIM reference: Wang, Samson, Zhao, SISC 2014.
    template <size_t N, bool UseLobattoTopNorm>
    constexpr Matrix<double, N, N> symmetric_orthogonal_accumulator(const Vector<double, N>& nodes,
                                                                    const Vector<double, N>& quadrature_weights)
    {
        const auto           table           = legendre_table(nodes);
        const auto           integrated      = integrated_legendre_basis(nodes);
        const auto           table_data      = table.data();
        const auto           integrated_data = integrated.data();
        Matrix<double, N, N> out{uninitialized};
        auto                 out_data = out.data();

        constexpr size_t row_count = (N + 1) / 2;
        for (size_t row = 0; row < row_count; ++row)
        {
            const size_t mirror_row = N - 1 - row;
            for (size_t col = 0; col < N; ++col)
            {
                const double value = orthogonal_accumulator_entry<N, UseLobattoTopNorm>(
                    table_data, integrated_data, row, col, quadrature_weights[col]);
                const size_t mirror_col               = N - 1 - col;
                out_data[row * N + col]               = value;
                out_data[mirror_row * N + mirror_col] = quadrature_weights[col] - value;
            }
        }

        return out;
    }

    template <size_t N>
    constexpr Vector<double, N> chebyshev_barycentric_weights()
    {
        Vector<double, N> weights{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            const double theta = pi * (2.0 * static_cast<double>(i) + 1.0) / (2.0 * static_cast<double>(N));
            weights[i]         = alternating_sign(i) * sin(theta);
        }
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> legendre_barycentric_weights(const Vector<double, N>& nodes)
    {
        Vector<double, N> weights{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            const double x = unit_to_legendre_x(nodes[i]);
            weights[i]     = 1.0 / legendre_derivative(N, x);
        }
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> lobatto_barycentric_weights(const Vector<double, N>& nodes)
    {
        Vector<double, N> weights{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            const double x = unit_to_legendre_x(nodes[i]);
            weights[i]     = 1.0 / legendre(N - 1, x);
        }
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> radau_barycentric_weights(const Vector<double, N>& nodes)
    {
        Vector<double, N> weights{uninitialized};

        if constexpr (N > 1)
        {
            for (size_t i = 0; i < N - 1; ++i)
            {
                const double x          = unit_to_legendre_x(nodes[i]);
                const auto   polynomial = jacobi_value_derivative(N - 1, 1.0, 0.0, x);
                weights[i]              = 1.0 / ((x - 1.0) * polynomial.derivative);
            }
        }

        weights[N - 1] = 1.0 / jacobi_at_one(N - 1, 1.0);
        return weights;
    }

    template <size_t N>
    constexpr Matrix<double, N, N> spectral_differentiator(const Vector<double, N>& nodes,
                                                           const Vector<double, N>& barycentric_weights)
    {
        Matrix<double, N, N> out{uninitialized};
        auto                 out_data = out.data();
        for (size_t row = 0; row < N; ++row)
        {
            double diagonal = 0.0;
            for (size_t col = 0; col < N; ++col)
            {
                if (row == col)
                    continue;

                const double value = barycentric_weights[col] / (barycentric_weights[row] * (nodes[row] - nodes[col]));
                out_data[row * N + col] = value;
                diagonal -= value;
            }
            out_data[row * N + row] = diagonal;
        }
        return out;
    }

    template <size_t N>
    constexpr Matrix<double, N, N> spectral_accumulator(const Vector<double, N>& nodes)
    {
        static_assert(N <= 256, "constexpr spectral accumulators require at most 256 points");

        const auto basis      = legendre_vandermonde(nodes);
        const auto integrated = integrated_legendre_basis(nodes);
        const auto solution_t = solve(transpose(basis), transpose(integrated));
        return transpose(solution_t);
    }

    constexpr double integer_power(double base, size_t exponent)
    {
        double value = 1.0;
        for (size_t i = 0; i < exponent; ++i)
            value *= base;
        return value;
    }

    template <size_t Bandwidth>
    constexpr size_t band_radius = Bandwidth / 2;

    template <size_t Bandwidth, size_t N>
    constexpr bool in_band(size_t row, size_t col) noexcept
    {
        return row + band_radius<Bandwidth> >= col && col + band_radius<Bandwidth> >= row;
    }

    template <size_t Bandwidth>
    constexpr size_t band_index(size_t row, size_t col) noexcept
    {
        return band_radius<Bandwidth> + row - col;
    }

    template <size_t Bandwidth, size_t N>
    constexpr void set_banded(Matrix<double, Bandwidth, N>& matrix, size_t row, size_t col, double value) noexcept
    {
        matrix[band_index<Bandwidth>(row, col) * N + col] = value;
    }

    template <size_t Bandwidth, size_t N>
    constexpr Matrix<double, N, N> dense_from_banded(const Matrix<double, Bandwidth, N>& matrix)
    {
        Matrix<double, N, N> dense;
        for (size_t row = 0; row < N; ++row)
            for (size_t col = 0; col < N; ++col)
                if (in_band<Bandwidth, N>(row, col))
                    dense[row * N + col] = matrix[band_index<Bandwidth>(row, col) * N + col];
        return dense;
    }

    template <size_t Width, size_t N>
    constexpr Vector<double, Width>
    finite_difference_weights(const Vector<double, N>& nodes, size_t start, double target)
    {
        Matrix<double, Width, Width> system{uninitialized};
        Matrix<double, Width, 1>     rhs;

        for (size_t degree = 0; degree < Width; ++degree)
            for (size_t col = 0; col < Width; ++col)
                system[degree * Width + col] = integer_power(nodes[start + col] - target, degree);

        rhs[1] = 1.0;
        return solve(system, rhs);
    }

    template <size_t ImplicitWidth, size_t ExplicitWidth>
    struct CompactRow
    {
        Vector<double, ImplicitWidth> a;
        Vector<double, ExplicitWidth> b;
    };

    template <size_t ImplicitWidth, size_t ExplicitWidth, size_t N>
    constexpr CompactRow<ImplicitWidth, ExplicitWidth>
    compact_row_coefficients(const Vector<double, N>&             nodes,
                             size_t                               row,
                             const Vector<size_t, ImplicitWidth>& implicit_indices,
                             const Vector<size_t, ExplicitWidth>& explicit_indices)
    {
        constexpr size_t implicit_unknown_count = ImplicitWidth - 1;
        constexpr size_t unknown_count          = implicit_unknown_count + ExplicitWidth;
        constexpr size_t center                 = ImplicitWidth / 2;

        Matrix<double, unknown_count, unknown_count> system{uninitialized};
        Matrix<double, unknown_count, 1>             rhs;

        for (size_t degree = 0; degree < unknown_count; ++degree)
        {
            size_t column = 0;
            for (size_t index = 0; index < ImplicitWidth; ++index)
            {
                if (index == center)
                    continue;

                const double offset = nodes[implicit_indices[index]] - nodes[row];
                system[degree * unknown_count + column] =
                    degree > 0 ? static_cast<double>(degree) * integer_power(offset, degree - 1) : 0.0;
                ++column;
            }

            for (size_t index = 0; index < ExplicitWidth; ++index)
            {
                const double offset                             = nodes[explicit_indices[index]] - nodes[row];
                system[degree * unknown_count + column + index] = -integer_power(offset, degree);
            }

            rhs[degree] = degree == 1 ? -1.0 : 0.0;
        }

        const auto solution = solve(system, rhs);

        CompactRow<ImplicitWidth, ExplicitWidth> row_coefficients;
        row_coefficients.a.fill(0.0);
        row_coefficients.a[center] = 1.0;
        for (size_t index = 0, column = 0; index < ImplicitWidth; ++index)
        {
            if (index == center)
                continue;
            row_coefficients.a[index] = solution[column];
            ++column;
        }

        for (size_t index = 0; index < ExplicitWidth; ++index)
            row_coefficients.b[index] = solution[implicit_unknown_count + index];

        return row_coefficients;
    }

    template <size_t N, size_t ImplicitWidth, size_t ExplicitWidth>
    struct CompactMatrices
    {
        Matrix<double, ImplicitWidth, N> a;
        Matrix<double, N, N>             b;
    };

    template <size_t N, size_t ImplicitWidth, size_t ExplicitWidth>
    constexpr CompactMatrices<N, ImplicitWidth, ExplicitWidth> compact_matrices(const Vector<double, N>& nodes)
    {
        static_assert(ImplicitWidth % 2 == 1, "compact implicit width must be odd");
        static_assert(ExplicitWidth % 2 == 1, "compact explicit width must be odd");
        static_assert(N >= 4, "compact CFD calculus requires at least four nodes");

        constexpr size_t implicit_radius = ImplicitWidth / 2;
        constexpr size_t explicit_radius = ExplicitWidth / 2;
        constexpr size_t wide_boundary   = ImplicitWidth + ExplicitWidth - 1;
        constexpr size_t boundary_width  = N < wide_boundary ? N : wide_boundary;

        CompactMatrices<N, ImplicitWidth, ExplicitWidth> matrices;
        matrices.a.fill(0.0);
        matrices.b.fill(0.0);

        for (size_t row = 0; row < N; ++row)
        {
            const bool has_implicit = row >= implicit_radius && row + implicit_radius < N;
            const bool has_explicit = row >= explicit_radius && row + explicit_radius < N;
            if (has_implicit && has_explicit)
            {
                Vector<size_t, ImplicitWidth> implicit_indices{uninitialized};
                Vector<size_t, ExplicitWidth> explicit_indices{uninitialized};
                for (size_t i = 0; i < ImplicitWidth; ++i)
                    implicit_indices[i] = row - implicit_radius + i;
                for (size_t i = 0; i < ExplicitWidth; ++i)
                    explicit_indices[i] = row - explicit_radius + i;

                const auto coefficients = compact_row_coefficients(nodes, row, implicit_indices, explicit_indices);
                for (size_t i = 0; i < ImplicitWidth; ++i)
                    set_banded(matrices.a, row, implicit_indices[i], coefficients.a[i]);
                for (size_t i = 0; i < ExplicitWidth; ++i)
                    matrices.b[row * N + explicit_indices[i]] = coefficients.b[i];
                continue;
            }

            size_t start = row > boundary_width / 2 ? row - boundary_width / 2 : 0;
            if (start + boundary_width > N)
                start = N - boundary_width;

            set_banded(matrices.a, row, row, 1.0);
            const auto weights = finite_difference_weights<boundary_width>(nodes, start, nodes[row]);
            for (size_t i = 0; i < boundary_width; ++i)
                matrices.b[row * N + start + i] = weights[i];
        }

        return matrices;
    }

    template <size_t N>
    constexpr Vector<double, N> interpolation_weights(const Vector<double, N>& nodes, double target)
    {
        Vector<double, N> weights{uninitialized};
        for (size_t col = 0; col < N; ++col)
        {
            double value = 1.0;
            for (size_t other = 0; other < N; ++other)
            {
                if (other == col)
                    continue;
                value *= (target - nodes[other]) / (nodes[col] - nodes[other]);
            }
            weights[col] = value;
        }
        return weights;
    }

    template <size_t N, size_t ImplicitWidth, size_t ExplicitWidth>
    constexpr Matrix<double, N, N> compact_differentiator(const Vector<double, N>& nodes)
    {
        const auto matrices = compact_matrices<N, ImplicitWidth, ExplicitWidth>(nodes);
        return solve<Thomas>(matrices.a, matrices.b);
    }

    template <size_t N, size_t ImplicitWidth, size_t ExplicitWidth>
    constexpr Matrix<double, N, N> compact_accumulator(const Vector<double, N>& nodes)
    {
        const auto matrices = compact_matrices<N, ImplicitWidth, ExplicitWidth>(nodes);
        auto       system   = matrices.b;
        auto       rhs      = dense_from_banded(matrices.a);

        size_t constraint_row = 0;
        double best_abs       = math::abs(nodes[0]);
        for (size_t row = 1; row < N; ++row)
        {
            const double candidate = math::abs(nodes[row]);
            if (candidate < best_abs)
            {
                best_abs       = candidate;
                constraint_row = row;
            }
        }

        const auto interpolation = interpolation_weights(nodes, 0.0);
        for (size_t col = 0; col < N; ++col)
        {
            system[constraint_row * N + col] = interpolation[col];
            rhs[constraint_row * N + col]    = 0.0;
        }

        return solve(system, rhs);
    }

    template <size_t N>
    constexpr Matrix<double, N, N> uniform_accumulator()
    {
        static_assert(N >= 2, "uniform accumulator requires at least two nodes");

        constexpr double     h = 1.0 / static_cast<double>(N - 1);
        Matrix<double, N, N> out;
        for (size_t row = 1; row < N; ++row)
        {
            out[row * N]       = 0.5 * h;
            out[row * N + row] = 0.5 * h;
            for (size_t col = 1; col < row; ++col)
                out[row * N + col] = h;
        }
        return out;
    }

    template <size_t N, typename Quadrature, size_t ImplicitWidth, size_t ExplicitWidth>
    constexpr Matrix<double, N, N> make_cfd_differentiator()
    {
        const auto& nodes = Quadrature::template nodes<N>;
        return compact_differentiator<N, ImplicitWidth, ExplicitWidth>(nodes);
    }

    template <size_t N, typename Quadrature, size_t ImplicitWidth, size_t ExplicitWidth>
    constexpr Matrix<double, N, N> make_cfd_accumulator()
    {
        const auto& nodes = Quadrature::template nodes<N>;
        return compact_accumulator<N, ImplicitWidth, ExplicitWidth>(nodes);
    }

    template <size_t N>
    constexpr Vector<double, N> make_chebyshev_nodes()
    {
        static_assert(N >= 1, "chebyshev nodes require at least one point");
        static_assert(N <= 256, "chebyshev nodes require at most 256 points");

        Vector<double, N> nodes{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            const double theta = pi * (2.0 * static_cast<double>(i) + 1.0) / (2.0 * static_cast<double>(N));
            nodes[i]           = unit_interval(-cos(theta));
        }
        return nodes;
    }

    template <size_t N>
    constexpr Vector<double, N> make_chebyshev_weights()
    {
        static_assert(N >= 1, "chebyshev weights require at least one point");
        static_assert(N <= 256, "chebyshev weights require at most 256 points");

        Vector<double, N> weights{uninitialized};
        for (size_t i = 0; i < N; ++i)
            weights[i] = fejer1_unit_weight(i, N);
        normalize_unit_weights(weights);
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> make_legendre_nodes()
    {
        static_assert(N >= 1, "legendre nodes require at least one point");
        static_assert(N <= 256, "legendre nodes require at most 256 points");

        const auto        roots = legendre_roots<N>();
        Vector<double, N> nodes{uninitialized};
        for (size_t i = 0; i < N; ++i)
            nodes[i] = unit_interval(roots[i]);
        return nodes;
    }

    template <size_t N>
    constexpr Vector<double, N> make_legendre_weights()
    {
        static_assert(N >= 1, "legendre weights require at least one point");
        static_assert(N <= 256, "legendre weights require at most 256 points");

        const auto        nodes = make_legendre_nodes<N>();
        Vector<double, N> weights{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            const double x          = 2.0 * nodes[i] - 1.0;
            const auto   polynomial = legendre_value_derivative(N, x);
            weights[i]              = 1.0 / ((1.0 - x * x) * polynomial.derivative * polynomial.derivative);
        }
        normalize_unit_weights(weights);
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> make_lobatto_nodes()
    {
        static_assert(N >= 2, "lobatto nodes require at least two points");
        static_assert(N <= 256, "lobatto nodes require at most 256 points");

        Vector<double, N> nodes{uninitialized};
        nodes[0]     = 0.0;
        nodes[N - 1] = 1.0;

        if constexpr (N > 2)
        {
            const auto interior = lobatto_interior_roots<N>();
            for (size_t i = 0; i < N - 2; ++i)
                nodes[i + 1] = unit_interval(interior[i]);
        }

        return nodes;
    }

    template <size_t N>
    constexpr Vector<double, N> make_lobatto_weights()
    {
        static_assert(N >= 2, "lobatto weights require at least two points");
        static_assert(N <= 256, "lobatto weights require at most 256 points");

        const double      scale = 1.0 / (static_cast<double>(N) * static_cast<double>(N - 1));
        const auto        nodes = make_lobatto_nodes<N>();
        Vector<double, N> weights{uninitialized};

        weights[0]     = scale;
        weights[N - 1] = scale;

        if constexpr (N > 2)
        {
            for (size_t i = 1; i < N - 1; ++i)
            {
                const double x     = 2.0 * nodes[i] - 1.0;
                const double value = legendre_value(N - 1, x);
                weights[i]         = scale / (value * value);
            }
        }
        normalize_unit_weights(weights);
        return weights;
    }

    template <size_t N>
    constexpr Vector<double, N> make_radau_nodes()
    {
        static_assert(N >= 1, "radau nodes require at least one point");
        static_assert(N <= 256, "radau nodes require at most 256 points");

        Vector<double, N> nodes{uninitialized};

        if constexpr (N > 1)
        {
            const auto interior = radau_right_roots<N>();
            for (size_t i = 0; i < N - 1; ++i)
                nodes[i] = unit_interval(interior[i]);
        }
        nodes[N - 1] = 1.0;

        return nodes;
    }

    template <size_t N>
    constexpr Vector<double, N> make_radau_weights()
    {
        static_assert(N >= 1, "radau weights require at least one point");
        static_assert(N <= 256, "radau weights require at most 256 points");

        const double      scale = 1.0 / (static_cast<double>(N) * static_cast<double>(N));
        const auto        nodes = make_radau_nodes<N>();
        Vector<double, N> weights{uninitialized};

        if constexpr (N > 1)
        {
            for (size_t i = 0; i < N - 1; ++i)
            {
                const double x     = 2.0 * nodes[i] - 1.0;
                const double value = legendre_value(N - 1, x);
                weights[i]         = 0.5 * (1.0 + x) * scale / (value * value);
            }
        }
        weights[N - 1] = scale;
        normalize_unit_weights(weights);
        return weights;
    }

    // Weight formulas are the classical interpolatory/Gauss family rules on
    // [-1, 1], rescaled to the native unit interval [0, 1].

    struct Chebyshev
    {
        template <size_t N>
        static constexpr Vector<double, N> nodes = make_chebyshev_nodes<N>();

        template <size_t N>
        static constexpr Vector<double, N> weights = make_chebyshev_weights<N>();
    };

    struct Legendre
    {
        template <size_t N>
        static constexpr Vector<double, N> nodes = make_legendre_nodes<N>();

        template <size_t N>
        static constexpr Vector<double, N> weights = make_legendre_weights<N>();
    };

    struct Lobatto
    {
        template <size_t N>
        static constexpr Vector<double, N> nodes = make_lobatto_nodes<N>();

        template <size_t N>
        static constexpr Vector<double, N> weights = make_lobatto_weights<N>();
    };

    struct Radau
    {
        template <size_t N>
        static constexpr Vector<double, N> nodes = make_radau_nodes<N>();

        template <size_t N>
        static constexpr Vector<double, N> weights = make_radau_weights<N>();
    };

    template <size_t N, typename Quadrature>
    constexpr Matrix<double, N, N> make_spectral_differentiator()
    {
        const auto& nodes = Quadrature::template nodes<N>;

        if constexpr (std::is_same_v<Quadrature, Chebyshev>)
        {
            const auto weights = chebyshev_barycentric_weights<N>();
            return spectral_differentiator(nodes, weights);
        }
        else if constexpr (std::is_same_v<Quadrature, Legendre>)
        {
            const auto weights = legendre_barycentric_weights(nodes);
            return spectral_differentiator(nodes, weights);
        }
        else if constexpr (std::is_same_v<Quadrature, Lobatto>)
        {
            const auto weights = lobatto_barycentric_weights(nodes);
            return spectral_differentiator(nodes, weights);
        }
        else // if constexpr (std::is_same_v<Quadrature, Radau>)
        {
            const auto weights = radau_barycentric_weights(nodes);
            return spectral_differentiator(nodes, weights);
        }
    }

    template <size_t N, typename Quadrature>
    constexpr Matrix<double, N, N> make_spectral_accumulator()
    {
        static_assert(N <= 256, "constexpr spectral accumulators require at most 256 points");

        const auto& nodes = Quadrature::template nodes<N>;

        if constexpr (std::is_same_v<Quadrature, Chebyshev>)
        {
            return spectral_accumulator(nodes);
        }
        else if constexpr (std::is_same_v<Quadrature, Legendre>)
        {
            const auto& weights = Quadrature::template weights<N>;
            return symmetric_orthogonal_accumulator<N, false>(nodes, weights);
        }
        else if constexpr (std::is_same_v<Quadrature, Lobatto>)
        {
            const auto& weights = Quadrature::template weights<N>;
            return symmetric_orthogonal_accumulator<N, true>(nodes, weights);
        }
        else // if constexpr (std::is_same_v<Quadrature, Radau>)
        {
            const auto& weights = Quadrature::template weights<N>;
            return orthogonal_accumulator<N, false>(nodes, weights);
        }
    }

    struct Spectral
    {
        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> differentiator = make_spectral_differentiator<N, Quadrature>();

        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> accumulator = make_spectral_accumulator<N, Quadrature>();
    };

    struct CFD33
    {
        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> differentiator = make_cfd_differentiator<N, Quadrature, 3, 3>();

        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> accumulator = make_cfd_accumulator<N, Quadrature, 3, 3>();
    };

    struct CFD35
    {
        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> differentiator = make_cfd_differentiator<N, Quadrature, 3, 5>();

        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> accumulator = make_cfd_accumulator<N, Quadrature, 3, 5>();
    };

    struct CFD55
    {
        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> differentiator = make_cfd_differentiator<N, Quadrature, 5, 5>();

        template <size_t N, typename Quadrature>
        static constexpr Matrix<double, N, N> accumulator = make_cfd_accumulator<N, Quadrature, 5, 5>();
    };

    template <size_t N>
    constexpr Vector<double, N> make_x(const Vector<double, N>& rho)
    {
        Vector<double, N> out{uninitialized};
        for (size_t i = 0; i < N; ++i)
            out[i] = 2.0 * rho[i] * rho[i] - 1.0;
        return out;
    }

    template <size_t N>
    constexpr Vector<double, N> make_y(const Vector<double, N>& rho)
    {
        Vector<double, N> out{uninitialized};
        for (size_t i = 0; i < N; ++i)
            out[i] = 1.0 - rho[i] * rho[i];
        return out;
    }

    template <size_t Kmax, size_t N>
    constexpr Matrix<double, Kmax, N> make_rhos(const Vector<double, N>& rho)
    {
        static_assert(Kmax >= 2, "rho table requires at least rho and rho^2");

        Matrix<double, Kmax, N> out{uninitialized};
        for (size_t i = 0; i < N; ++i)
        {
            double value = rho[i];
            for (size_t row = 0; row < Kmax; ++row)
            {
                out(row, i) = value;
                value *= rho[i];
            }
        }
        return out;
    }

    enum class ChebyshevField
    {
        value,
        radial,
        radial2,
    };

    template <size_t Lmax, ChebyshevField Field, size_t N>
    constexpr Matrix<double, Lmax, N> make_chebyshev_table(const Vector<double, N>& rho, const Vector<double, N>& x)
    {
        static_assert(Lmax >= 1, "Chebyshev table requires at least one stored row");

        Matrix<double, Lmax, N> out{};
        for (size_t i = 0; i < N; ++i)
        {
            const double xi       = x[i];
            const double dx_dr    = 4.0 * rho[i];
            const double d2x_dr2  = 4.0;
            double       T_prev   = 1.0;
            double       Tx_prev  = 0.0;
            double       Txx_prev = 0.0;
            double       T_curr   = xi;
            double       Tx_curr  = 1.0;
            double       Txx_curr = 0.0;

            if constexpr (Field == ChebyshevField::value)
                out(0, i) = T_curr;
            else if constexpr (Field == ChebyshevField::radial)
                out(0, i) = Tx_curr * dx_dr;
            else
                out(0, i) = Txx_curr * dx_dr * dx_dr + Tx_curr * d2x_dr2;

            for (size_t degree = 1; degree < Lmax; ++degree)
            {
                const double T_next   = 2.0 * xi * T_curr - T_prev;
                const double Tx_next  = 2.0 * T_curr + 2.0 * xi * Tx_curr - Tx_prev;
                const double Txx_next = 4.0 * Tx_curr + 2.0 * xi * Txx_curr - Txx_prev;

                if constexpr (Field == ChebyshevField::value)
                    out(degree, i) = T_next;
                else if constexpr (Field == ChebyshevField::radial)
                    out(degree, i) = Tx_next * dx_dr;
                else
                    out(degree, i) = Txx_next * dx_dr * dx_dr + Tx_next * d2x_dr2;

                T_prev   = T_curr;
                Tx_prev  = Tx_curr;
                Txx_prev = Txx_curr;
                T_curr   = T_next;
                Tx_curr  = Tx_next;
                Txx_curr = Txx_next;
            }
        }
        return out;
    }

    template <size_t N>
    constexpr Vector<double, N> make_theta()
    {
        Vector<double, N> out{uninitialized};
        const double      step = 2.0 * pi / static_cast<double>(N);
        for (size_t i = 0; i < N; ++i)
            out[i] = step * static_cast<double>(i);
        return out;
    }

    enum class TrigField
    {
        cos,
        sin,
        m_cos,
        m_sin,
        m2_cos,
        m2_sin,
    };

    template <size_t Mmax, TrigField Field, size_t N>
    constexpr Matrix<double, Mmax + 1, N> make_trig_table(const Vector<double, N>& theta)
    {
        static_assert(Mmax >= 1, "trig table requires at least one positive harmonic");

        Matrix<double, Mmax + 1, N> out{uninitialized};
        for (size_t order = 0; order <= Mmax; ++order)
        {
            const double m  = static_cast<double>(order);
            const double m2 = m * m;
            for (size_t i = 0; i < N; ++i)
            {
                const double angle = m * theta[i];
                const double c     = cos(angle);
                const double s     = sin(angle);

                if constexpr (Field == TrigField::cos)
                    out(order, i) = c;
                else if constexpr (Field == TrigField::sin)
                    out(order, i) = s;
                else if constexpr (Field == TrigField::m_cos)
                    out(order, i) = m * c;
                else if constexpr (Field == TrigField::m_sin)
                    out(order, i) = m * s;
                else if constexpr (Field == TrigField::m2_cos)
                    out(order, i) = m2 * c;
                else
                    out(order, i) = m2 * s;
            }
        }
        return out;
    }

    template <size_t Nr, size_t Nt, size_t Lmax, size_t Mmax, size_t Kmax, typename Quadrature, typename Calculus>
    struct Grid
    {
        static_assert(Nr >= 4, "Grid requires at least four radial nodes");
        static_assert(Nt >= 4, "Grid requires at least four poloidal nodes");
        static_assert(Lmax >= 1, "Grid requires at least one Chebyshev row");
        static_assert(Mmax >= 1, "Grid requires at least one positive harmonic");
        static_assert(Kmax >= 2, "Grid requires rho and rho^2 rows");

        static constexpr size_t basis_rows     = Lmax;
        static constexpr size_t rho_power_rows = Kmax;
        static constexpr size_t harmonic_rows  = Mmax + 1;
        static constexpr size_t radial_nodes   = Nr;
        static constexpr size_t theta_rows     = Nt;
        static constexpr auto   nodes          = Quadrature::template nodes<Nr>;
        static constexpr auto   weights        = Quadrature::template weights<Nr>;
        static constexpr auto   accumulator    = Calculus::template accumulator<Nr, Quadrature>;
        static constexpr auto   differentiator = Calculus::template differentiator<Nr, Quadrature>;
        static constexpr auto   x              = make_x(nodes);
        static constexpr auto   y              = make_y(nodes);
        static constexpr auto   rhos           = make_rhos<Kmax>(nodes);
        static constexpr auto   T              = make_chebyshev_table<Lmax, ChebyshevField::value>(nodes, x);
        static constexpr auto   T_r            = make_chebyshev_table<Lmax, ChebyshevField::radial>(nodes, x);
        static constexpr auto   T_rr           = make_chebyshev_table<Lmax, ChebyshevField::radial2>(nodes, x);
        static constexpr auto   theta          = make_theta<Nt>();
        static constexpr auto   cos_mtheta     = make_trig_table<Mmax, TrigField::cos>(theta);
        static constexpr auto   sin_mtheta     = make_trig_table<Mmax, TrigField::sin>(theta);
        static constexpr auto   m_cos_mtheta   = make_trig_table<Mmax, TrigField::m_cos>(theta);
        static constexpr auto   m_sin_mtheta   = make_trig_table<Mmax, TrigField::m_sin>(theta);
        static constexpr auto   m2_cos_mtheta  = make_trig_table<Mmax, TrigField::m2_cos>(theta);
        static constexpr auto   m2_sin_mtheta  = make_trig_table<Mmax, TrigField::m2_sin>(theta);
    };
} // namespace grid::detail

namespace grid
{
    using detail::Chebyshev;
    using detail::Legendre;
    using detail::Lobatto;
    using detail::Radau;

    using detail::Spectral;
    using detail::CFD33;
    using detail::CFD35;
    using detail::CFD55;

    using detail::Grid;
} // namespace grid
