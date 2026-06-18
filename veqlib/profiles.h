#pragma once

#include "config.h"
#include "math.h"
#include "tensor.h"

namespace profiles::detail
{
    using math::max;
    using math::sqrt;
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;

    inline constexpr double F_squared_amplitude_floor = 1.0e-10;

    struct ProfileValues
    {
        double value;
        double diff;
        double diff2;
    };

    template <size_t Order>
    consteval size_t c_count()
    {
        static_assert(Order < config::c_family_counts.size(), "c profile order exceeds configured counts");
        return config::c_family_counts[Order];
    }

    template <size_t Order>
    consteval size_t s_count()
    {
        static_assert(Order > 0, "s0 is not a physical sine profile");
        static_assert(Order <= config::s_family_counts.size(), "s profile order exceeds configured counts");
        return config::s_family_counts[Order - 1];
    }

    template <size_t Order>
    consteval size_t fourier_power()
    {
        return Order < config::K_max ? Order : config::K_max;
    }

    template <size_t Count, size_t Nr>
        requires(Count > 0)
    constexpr void update_polys(Matrix<double, Nr, 3>&                   polys,
                                const Vector<double, Count>&             coeffs,
                                const Matrix<double, config::L_max, Nr>& T,
                                const Matrix<double, config::L_max, Nr>& T_r,
                                const Matrix<double, config::L_max, Nr>& T_rr) noexcept
    {
        static_assert(Count <= config::L_max + 1, "profile count exceeds shared basis rows plus T0");

        for (size_t i = 0; i < Nr; ++i)
        {
            double value = coeffs[0];
            double diff  = 0.0;
            double diff2 = 0.0;

            for (size_t k = 1; k < Count; ++k)
            {
                const size_t row   = k - 1;
                const double coeff = coeffs[k];
                value += coeff * T(row, i);
                diff += coeff * T_r(row, i);
                diff2 += coeff * T_rr(row, i);
            }

            polys(i, 0) = value;
            polys(i, 1) = diff;
            polys(i, 2) = diff2;
        }
    }

    template <size_t Nr>
    constexpr double rho_at(const Matrix<double, config::K_max, Nr>& rhos, size_t i) noexcept
    {
        static_assert(config::K_max >= 2, "rho table must contain rho and rho^2 rows");
        return rhos(0, i);
    }

    template <size_t Nr>
    constexpr double rho2_at(const Matrix<double, config::K_max, Nr>& rhos, size_t i) noexcept
    {
        static_assert(config::K_max >= 2, "rho table must contain rho and rho^2 rows");
        return rhos(1, i);
    }

    template <size_t Nr>
    constexpr double y_at(const Matrix<double, config::K_max, Nr>& rhos, size_t i) noexcept
    {
        return 1.0 - rho2_at(rhos, i);
    }

    template <size_t Power, size_t Nr>
    constexpr ProfileValues rho_power_rows(const Matrix<double, config::K_max, Nr>& rhos, size_t i) noexcept
    {
        static_assert(Power <= config::K_max, "rho power exceeds shared rho table rows");

        if constexpr (Power == 0)
            return {1.0, 0.0, 0.0};
        else if constexpr (Power == 1)
            return {rho_at(rhos, i), 1.0, 0.0};
        else
        {
            const double rho_pm2 = Power == 2 ? 1.0 : rhos(Power - 3, i);
            const double rho_pm1 = rhos(Power - 2, i);
            const double rho_p   = rhos(Power - 1, i);
            return {
                rho_p,
                static_cast<double>(Power) * rho_pm1,
                static_cast<double>(Power * (Power - 1)) * rho_pm2,
            };
        }
    }

    template <size_t Nr>
    constexpr void update_enveloped_profiles(Matrix<double, Nr, 3>&                   profiles,
                                             const Matrix<double, Nr, 3>&             polys,
                                             const Matrix<double, config::K_max, Nr>& rhos) noexcept
    {
        for (size_t i = 0; i < Nr; ++i)
        {
            const double rho   = rho_at(rhos, i);
            const double y     = y_at(rhos, i);
            const double value = polys(i, 0);
            const double diff  = polys(i, 1);
            const double diff2 = polys(i, 2);

            profiles(i, 0) = y * value;
            profiles(i, 1) = -2.0 * rho * value + y * diff;
            profiles(i, 2) = -2.0 * value - 4.0 * rho * diff + y * diff2;
        }
    }

    template <size_t Nr>
        requires(config::kappa_count > 0)
    constexpr void update_kappa_from_polys(Matrix<double, Nr, 3>&                   profiles,
                                           const Matrix<double, Nr, 3>&             polys,
                                           const Matrix<double, config::K_max, Nr>& rhos,
                                           double                                   ka) noexcept
    {
        for (size_t i = 0; i < Nr; ++i)
        {
            const double rho     = rho_at(rhos, i);
            const double y       = y_at(rhos, i);
            const double value   = polys(i, 0);
            const double diff    = polys(i, 1);
            const double diff2   = polys(i, 2);
            const double base    = y * value;
            const double base_r  = -2.0 * rho * value + y * diff;
            const double base_rr = -2.0 * value - 4.0 * rho * diff + y * diff2;

            profiles(i, 0) = ka + base;
            profiles(i, 1) = base_r;
            profiles(i, 2) = base_rr;
        }
    }

    template <size_t Nr>
        requires(config::psin_count > 0)
    constexpr void update_psin_from_polys(Matrix<double, Nr, 3>&                   profiles,
                                          const Matrix<double, Nr, 3>&             polys,
                                          const Matrix<double, config::K_max, Nr>& rhos) noexcept
    {
        for (size_t i = 0; i < Nr; ++i)
        {
            const double rho     = rho_at(rhos, i);
            const double y       = y_at(rhos, i);
            const double value   = polys(i, 0);
            const double diff    = polys(i, 1);
            const double diff2   = polys(i, 2);
            const double base    = y * value;
            const double base_r  = -2.0 * rho * value + y * diff;
            const double base_rr = -2.0 * value - 4.0 * rho * diff + y * diff2;
            const double amp     = 1.0 + base;
            const double rp      = rho2_at(rhos, i);
            const double rp_r    = 2.0 * rho;
            const double rp_rr   = 2.0;

            profiles(i, 0) = rp * amp;
            profiles(i, 1) = rp_r * amp + rp * base_r;
            profiles(i, 2) = rp_rr * amp + 2.0 * rp_r * base_r + rp * base_rr;
        }
    }

    template <size_t Nr>
        requires(config::F_count > 0)
    constexpr void update_F_from_polys(Matrix<double, Nr, 3>&                   profiles,
                                       const Matrix<double, Nr, 3>&             polys,
                                       const Matrix<double, config::K_max, Nr>& rhos,
                                       double                                   scale) noexcept
    {
        for (size_t i = 0; i < Nr; ++i)
        {
            const double rho               = rho_at(rhos, i);
            const double y                 = y_at(rhos, i);
            const double value             = polys(i, 0);
            const double diff              = polys(i, 1);
            const double diff2             = polys(i, 2);
            const double base              = y * value;
            const double base_r            = -2.0 * rho * value + y * diff;
            const double base_rr           = -2.0 * value - 4.0 * rho * diff + y * diff2;
            const double amp_raw_unclamped = 1.0 + base;
            const double amp_raw           = max(amp_raw_unclamped, F_squared_amplitude_floor);
            const double amp               = sqrt(amp_raw);
            const double inv_amp           = 1.0 / amp;
            const double inv_amp3          = inv_amp / amp_raw;
            const double amp_r             = 0.5 * base_r * inv_amp;
            const double amp_rr            = 0.5 * base_rr * inv_amp - 0.25 * base_r * base_r * inv_amp3;

            profiles(i, 0) = scale * amp;
            profiles(i, 1) = scale * amp_r;
            profiles(i, 2) = scale * amp_rr;
        }
    }

    template <size_t Order, size_t Nr>
    constexpr void update_fourier_from_polys(Matrix<double, Nr, 3>&                   profiles,
                                             const Matrix<double, Nr, 3>&             polys,
                                             const Matrix<double, config::K_max, Nr>& rhos,
                                             double                                   offset) noexcept
    {
        constexpr size_t Power = fourier_power<Order>();
        for (size_t i = 0; i < Nr; ++i)
        {
            const ProfileValues rp      = rho_power_rows<Power>(rhos, i);
            const double        rho     = rho_at(rhos, i);
            const double        y       = y_at(rhos, i);
            const double        value   = polys(i, 0);
            const double        diff    = polys(i, 1);
            const double        diff2   = polys(i, 2);
            const double        base    = y * value;
            const double        base_r  = -2.0 * rho * value + y * diff;
            const double        base_rr = -2.0 * value - 4.0 * rho * diff + y * diff2;
            const double        amp     = offset + base;

            profiles(i, 0) = rp.value * amp;
            profiles(i, 1) = rp.diff * amp + rp.value * base_r;
            profiles(i, 2) = rp.diff2 * amp + 2.0 * rp.diff * base_r + rp.value * base_rr;
        }
    }

    template <size_t Nr>
        requires(config::h_count > 0 && config::L_max + 1 >= config::h_count && config::K_max >= 2)
    constexpr void update_h_profiles(Matrix<double, Nr, 3>&                   profiles,
                                     const Vector<double, config::h_count>&   coeffs,
                                     const Matrix<double, config::L_max, Nr>& T,
                                     const Matrix<double, config::L_max, Nr>& T_r,
                                     const Matrix<double, config::L_max, Nr>& T_rr,
                                     const Matrix<double, config::K_max, Nr>& rhos) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<config::h_count>(polys, coeffs, T, T_r, T_rr);
        update_enveloped_profiles(profiles, polys, rhos);
    }

    template <size_t Nr>
        requires(config::v_count > 0 && config::L_max + 1 >= config::v_count && config::K_max >= 2)
    constexpr void update_v_profiles(Matrix<double, Nr, 3>&                   profiles,
                                     const Vector<double, config::v_count>&   coeffs,
                                     const Matrix<double, config::L_max, Nr>& T,
                                     const Matrix<double, config::L_max, Nr>& T_r,
                                     const Matrix<double, config::L_max, Nr>& T_rr,
                                     const Matrix<double, config::K_max, Nr>& rhos) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<config::v_count>(polys, coeffs, T, T_r, T_rr);
        update_enveloped_profiles(profiles, polys, rhos);
    }

    template <size_t Nr>
        requires(config::kappa_count > 0 && config::L_max + 1 >= config::kappa_count && config::K_max >= 2)
    constexpr void update_kappa_profiles(Matrix<double, Nr, 3>&                     profiles,
                                         const Vector<double, config::kappa_count>& coeffs,
                                         const Matrix<double, config::L_max, Nr>&   T,
                                         const Matrix<double, config::L_max, Nr>&   T_r,
                                         const Matrix<double, config::L_max, Nr>&   T_rr,
                                         const Matrix<double, config::K_max, Nr>&   rhos,
                                         double                                     ka) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<config::kappa_count>(polys, coeffs, T, T_r, T_rr);
        update_kappa_from_polys(profiles, polys, rhos, ka);
    }

    template <size_t Nr>
        requires(config::psin_count > 0 && config::L_max + 1 >= config::psin_count && config::K_max >= 2)
    constexpr void update_psin_profiles(Matrix<double, Nr, 3>&                    profiles,
                                        const Vector<double, config::psin_count>& coeffs,
                                        const Matrix<double, config::L_max, Nr>&  T,
                                        const Matrix<double, config::L_max, Nr>&  T_r,
                                        const Matrix<double, config::L_max, Nr>&  T_rr,
                                        const Matrix<double, config::K_max, Nr>&  rhos) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<config::psin_count>(polys, coeffs, T, T_r, T_rr);
        update_psin_from_polys(profiles, polys, rhos);
    }

    template <size_t Nr>
        requires(config::F_count > 0 && config::L_max + 1 >= config::F_count && config::K_max >= 2)
    constexpr void update_F_profiles(Matrix<double, Nr, 3>&                   profiles,
                                     const Vector<double, config::F_count>&   coeffs,
                                     const Matrix<double, config::L_max, Nr>& T,
                                     const Matrix<double, config::L_max, Nr>& T_r,
                                     const Matrix<double, config::L_max, Nr>& T_rr,
                                     const Matrix<double, config::K_max, Nr>& rhos,
                                     double                                   scale) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<config::F_count>(polys, coeffs, T, T_r, T_rr);
        update_F_from_polys(profiles, polys, rhos, scale);
    }

    template <size_t Order, size_t Nr>
        requires(c_count<Order>() > 0 && config::L_max + 1 >= c_count<Order>() && config::K_max >= 2 &&
                 config::K_max >= fourier_power<Order>())
    constexpr void update_c_profiles(Matrix<double, Nr, 3>&                   profiles,
                                     const Vector<double, c_count<Order>()>&  coeffs,
                                     const Matrix<double, config::L_max, Nr>& T,
                                     const Matrix<double, config::L_max, Nr>& T_r,
                                     const Matrix<double, config::L_max, Nr>& T_rr,
                                     const Matrix<double, config::K_max, Nr>& rhos,
                                     double                                   offset) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<c_count<Order>()>(polys, coeffs, T, T_r, T_rr);
        update_fourier_from_polys<Order>(profiles, polys, rhos, offset);
    }

    template <size_t Order, size_t Nr>
        requires(s_count<Order>() > 0 && config::L_max + 1 >= s_count<Order>() && config::K_max >= 2 &&
                 config::K_max >= fourier_power<Order>())
    constexpr void update_s_profiles(Matrix<double, Nr, 3>&                   profiles,
                                     const Vector<double, s_count<Order>()>&  coeffs,
                                     const Matrix<double, config::L_max, Nr>& T,
                                     const Matrix<double, config::L_max, Nr>& T_r,
                                     const Matrix<double, config::L_max, Nr>& T_rr,
                                     const Matrix<double, config::K_max, Nr>& rhos,
                                     double                                   offset) noexcept
    {
        Matrix<double, Nr, 3> polys{};
        update_polys<s_count<Order>()>(polys, coeffs, T, T_r, T_rr);
        update_fourier_from_polys<Order>(profiles, polys, rhos, offset);
    }
} // namespace profiles::detail

namespace profiles
{
    using detail::update_h_profiles;
    using detail::update_v_profiles;
    using detail::update_kappa_profiles;
    using detail::update_c_profiles;
    using detail::update_s_profiles;
    using detail::update_psin_profiles;
    using detail::update_F_profiles;
} // namespace profiles
