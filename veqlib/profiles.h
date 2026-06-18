#pragma once

#include "math.h"
#include "tensor.h"
#include <cstddef>

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

    template <size_t Lmax, size_t Count, size_t Nr>
        requires(Count > 0)
    constexpr void update_polys(Matrix<double, Nr, 3>&          polys,
                                const Vector<double, Count>&    coeffs,
                                const Matrix<double, Lmax, Nr>& T,
                                const Matrix<double, Lmax, Nr>& T_r,
                                const Matrix<double, Lmax, Nr>& T_rr) noexcept
    {
        static_assert(Lmax >= 1, "profile basis table must contain stored T1.. rows");
        static_assert(Count <= Lmax + 1, "profile count exceeds shared basis rows plus T0");

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

    template <size_t Kmax, size_t Nr>
    constexpr double rho_at(const Matrix<double, Kmax, Nr>& rhos, size_t i) noexcept
    {
        static_assert(Kmax >= 2, "rho table must contain rho and rho^2 rows");
        return rhos(0, i);
    }

    template <size_t Kmax, size_t Nr>
    constexpr double rho2_at(const Matrix<double, Kmax, Nr>& rhos, size_t i) noexcept
    {
        static_assert(Kmax >= 2, "rho table must contain rho and rho^2 rows");
        return rhos(1, i);
    }

    template <size_t Kmax, size_t Nr>
    constexpr double y_at(const Matrix<double, Kmax, Nr>& rhos, size_t i) noexcept
    {
        return 1.0 - rho2_at(rhos, i);
    }

    template <size_t Kmax, size_t Power, size_t Nr>
    constexpr ProfileValues rho_power_rows(const Matrix<double, Kmax, Nr>& rhos, size_t i) noexcept
    {
        static_assert(Power <= Kmax, "rho power exceeds shared rho table rows");

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

    template <size_t Kmax, size_t Nr>
    constexpr void update_enveloped_profiles(Matrix<double, Nr, 3>&          profiles,
                                             const Matrix<double, Nr, 3>&    polys,
                                             const Matrix<double, Kmax, Nr>& rhos) noexcept
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

    template <size_t Kmax, size_t Nr>
    constexpr void update_kappa_from_polys(Matrix<double, Nr, 3>&          profiles,
                                           const Matrix<double, Nr, 3>&    polys,
                                           const Matrix<double, Kmax, Nr>& rhos,
                                           double                          ka) noexcept
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

    template <size_t Kmax, size_t Nr>
    constexpr void update_psin_from_polys(Matrix<double, Nr, 3>&          profiles,
                                          const Matrix<double, Nr, 3>&    polys,
                                          const Matrix<double, Kmax, Nr>& rhos) noexcept
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

    template <size_t Kmax, size_t Nr>
    constexpr void update_F_from_polys(Matrix<double, Nr, 3>&          profiles,
                                       const Matrix<double, Nr, 3>&    polys,
                                       const Matrix<double, Kmax, Nr>& rhos,
                                       double                          scale) noexcept
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

    template <size_t Kmax, size_t Power, size_t Nr>
    constexpr void update_fourier_from_polys(Matrix<double, Nr, 3>&          profiles,
                                             const Matrix<double, Nr, 3>&    polys,
                                             const Matrix<double, Kmax, Nr>& rhos,
                                             double                          offset) noexcept
    {
        for (size_t i = 0; i < Nr; ++i)
        {
            const ProfileValues rp      = rho_power_rows<Kmax, Power>(rhos, i);
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
} // namespace profiles::detail

namespace profiles
{
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;

    template <size_t Lmax,
              size_t Kmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts>
    struct Profiles
    {
        static_assert(Lmax >= 1, "Profiles requires at least one stored Chebyshev row");
        static_assert(Kmax >= 2, "Profiles requires rho and rho^2 rows");

        static constexpr size_t basis_rows    = Lmax;
        static constexpr size_t rho_rows      = Kmax;
        static constexpr size_t c_family_size = CFamilyCounts.size();
        static constexpr size_t s_family_size = SFamilyCounts.size();

        template <size_t Order>
        static consteval size_t c_count()
        {
            static_assert(Order < c_family_size, "c profile order exceeds configured counts");
            return CFamilyCounts[Order];
        }

        template <size_t Order>
        static consteval size_t s_count()
        {
            static_assert(Order > 0, "s0 is not a physical sine profile");
            static_assert(Order <= s_family_size, "s profile order exceeds configured counts");
            return SFamilyCounts[Order - 1];
        }

        template <size_t Order>
        static consteval size_t fourier_power()
        {
            return Order < Kmax ? Order : Kmax;
        }

        template <size_t Nr>
            requires(HCount > 0 && Lmax + 1 >= HCount)
        static constexpr void update_h(Matrix<double, Nr, 3>&          profiles,
                                       const Vector<double, HCount>&   coeffs,
                                       const Matrix<double, Lmax, Nr>& T,
                                       const Matrix<double, Lmax, Nr>& T_r,
                                       const Matrix<double, Lmax, Nr>& T_rr,
                                       const Matrix<double, Kmax, Nr>& rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, HCount>(polys, coeffs, T, T_r, T_rr);
            detail::update_enveloped_profiles<Kmax>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(VCount > 0 && Lmax + 1 >= VCount)
        static constexpr void update_v(Matrix<double, Nr, 3>&          profiles,
                                       const Vector<double, VCount>&   coeffs,
                                       const Matrix<double, Lmax, Nr>& T,
                                       const Matrix<double, Lmax, Nr>& T_r,
                                       const Matrix<double, Lmax, Nr>& T_rr,
                                       const Matrix<double, Kmax, Nr>& rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, VCount>(polys, coeffs, T, T_r, T_rr);
            detail::update_enveloped_profiles<Kmax>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(KappaCount > 0 && Lmax + 1 >= KappaCount)
        static constexpr void update_kappa(Matrix<double, Nr, 3>&            profiles,
                                           const Vector<double, KappaCount>& coeffs,
                                           const Matrix<double, Lmax, Nr>&   T,
                                           const Matrix<double, Lmax, Nr>&   T_r,
                                           const Matrix<double, Lmax, Nr>&   T_rr,
                                           const Matrix<double, Kmax, Nr>&   rhos,
                                           double                            ka) noexcept
        {
            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, KappaCount>(polys, coeffs, T, T_r, T_rr);
            detail::update_kappa_from_polys<Kmax>(profiles, polys, rhos, ka);
        }

        template <size_t Nr>
            requires(PsinCount > 0 && Lmax + 1 >= PsinCount)
        static constexpr void update_psin(Matrix<double, Nr, 3>&           profiles,
                                          const Vector<double, PsinCount>& coeffs,
                                          const Matrix<double, Lmax, Nr>&  T,
                                          const Matrix<double, Lmax, Nr>&  T_r,
                                          const Matrix<double, Lmax, Nr>&  T_rr,
                                          const Matrix<double, Kmax, Nr>&  rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, PsinCount>(polys, coeffs, T, T_r, T_rr);
            detail::update_psin_from_polys<Kmax>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(FCount > 0 && Lmax + 1 >= FCount)
        static constexpr void update_F(Matrix<double, Nr, 3>&          profiles,
                                       const Vector<double, FCount>&   coeffs,
                                       const Matrix<double, Lmax, Nr>& T,
                                       const Matrix<double, Lmax, Nr>& T_r,
                                       const Matrix<double, Lmax, Nr>& T_rr,
                                       const Matrix<double, Kmax, Nr>& rhos,
                                       double                          scale) noexcept
        {
            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, FCount>(polys, coeffs, T, T_r, T_rr);
            detail::update_F_from_polys<Kmax>(profiles, polys, rhos, scale);
        }

        template <size_t Order, size_t Nr>
            requires(c_count<Order>() > 0 && Lmax + 1 >= c_count<Order>() && Kmax >= fourier_power<Order>())
        static constexpr void update_c(Matrix<double, Nr, 3>&                  profiles,
                                       const Vector<double, c_count<Order>()>& coeffs,
                                       const Matrix<double, Lmax, Nr>&         T,
                                       const Matrix<double, Lmax, Nr>&         T_r,
                                       const Matrix<double, Lmax, Nr>&         T_rr,
                                       const Matrix<double, Kmax, Nr>&         rhos,
                                       double                                  offset) noexcept
        {
            constexpr size_t Count = c_count<Order>();
            constexpr size_t Power = fourier_power<Order>();

            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, Count>(polys, coeffs, T, T_r, T_rr);
            detail::update_fourier_from_polys<Kmax, Power>(profiles, polys, rhos, offset);
        }

        template <size_t Order, size_t Nr>
            requires(s_count<Order>() > 0 && Lmax + 1 >= s_count<Order>() && Kmax >= fourier_power<Order>())
        static constexpr void update_s(Matrix<double, Nr, 3>&                  profiles,
                                       const Vector<double, s_count<Order>()>& coeffs,
                                       const Matrix<double, Lmax, Nr>&         T,
                                       const Matrix<double, Lmax, Nr>&         T_r,
                                       const Matrix<double, Lmax, Nr>&         T_rr,
                                       const Matrix<double, Kmax, Nr>&         rhos,
                                       double                                  offset) noexcept
        {
            constexpr size_t Count = s_count<Order>();
            constexpr size_t Power = fourier_power<Order>();

            Matrix<double, Nr, 3> polys{};
            detail::update_polys<Lmax, Count>(polys, coeffs, T, T_r, T_rr);
            detail::update_fourier_from_polys<Kmax, Power>(profiles, polys, rhos, offset);
        }
    };
} // namespace profiles
