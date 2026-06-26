#pragma once

#include "math.h"
#include "tensor.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <utility>

namespace profiles
{
    using std::size_t;
    using math::abs;

    inline constexpr double boundary_amplitude_prune_threshold = 1.0e-10;

    enum class ProfileMode
    {
        absent,
        fixed,
        optimized,
    };

    struct ProfileSlot
    {
        ProfileMode mode;
        size_t      coefficient_count;

        constexpr bool enabled() const noexcept { return mode != ProfileMode::absent; }
        constexpr bool optimized() const noexcept { return mode == ProfileMode::optimized; }
        constexpr bool fixed() const noexcept { return mode == ProfileMode::fixed; }

        friend constexpr bool operator==(ProfileSlot, ProfileSlot) noexcept = default;
    };

    consteval ProfileSlot absent_slot() noexcept { return {ProfileMode::absent, 0}; }

    consteval ProfileSlot fixed_slot(size_t coefficient_count = 0) noexcept
    {
        return {ProfileMode::fixed, coefficient_count};
    }

    consteval ProfileSlot optimized_slot(size_t coefficient_count) noexcept
    {
        return {ProfileMode::optimized, coefficient_count};
    }

    consteval ProfileSlot optimized_slot_from_count(size_t coefficient_count) noexcept
    {
        return coefficient_count == 0 ? absent_slot() : optimized_slot(coefficient_count);
    }

    template <auto Counts>
    consteval ProfileSlot first_optimized_slot_from_counts() noexcept
    {
        if constexpr (Counts.size() == 0)
            return absent_slot();
        else
            return optimized_slot_from_count(Counts[0]);
    }

    template <auto Counts, size_t... Indices>
    consteval auto optimized_slots_from_counts_impl(std::index_sequence<Indices...>) noexcept
    {
        return std::array<ProfileSlot, sizeof...(Indices)>{optimized_slot_from_count(Counts[Indices])...};
    }

    template <auto Counts>
    consteval auto optimized_slots_from_counts() noexcept
    {
        return optimized_slots_from_counts_impl<Counts>(std::make_index_sequence<Counts.size()>{});
    }

    template <auto Counts, size_t... Indices>
    consteval auto tail_optimized_slots_from_counts_impl(std::index_sequence<Indices...>) noexcept
    {
        return std::array<ProfileSlot, sizeof...(Indices)>{optimized_slot_from_count(Counts[Indices + 1])...};
    }

    template <auto Counts>
    consteval auto tail_optimized_slots_from_counts() noexcept
    {
        if constexpr (Counts.size() == 0)
            return std::array<ProfileSlot, 0>{};
        else
            return tail_optimized_slots_from_counts_impl<Counts>(std::make_index_sequence<Counts.size() - 1>{});
    }

    template <size_t      Lmax,
              size_t      Kmax,
              size_t      Mmax,
              ProfileSlot HSlot,
              ProfileSlot VSlot,
              ProfileSlot KappaSlot,
              ProfileSlot C0Slot,
              ProfileSlot PsinSlot,
              ProfileSlot FSlot,
              auto        CFamilySlots,
              auto        SFamilySlots,
              bool        LayoutProfileFirst = false>
    struct ProfileShape
    {
        static_assert(Lmax >= 1, "ProfileShape requires at least one stored Chebyshev row");
        static_assert(Kmax >= 2, "ProfileShape requires rho and rho^2 rows");
        static_assert(CFamilySlots.size() <= Mmax, "c-family slots exceed Mmax");
        static_assert(SFamilySlots.size() <= Mmax, "s-family slots exceed Mmax");

        static constexpr size_t L_max = Lmax;
        static constexpr size_t K_max = Kmax;
        static constexpr size_t M_max = Mmax;

        static constexpr size_t c_family_slot_count = CFamilySlots.size();
        static constexpr size_t s_family_slot_count = SFamilySlots.size();

        static constexpr size_t h_profile_id     = 0;
        static constexpr size_t v_profile_id     = 1;
        static constexpr size_t kappa_profile_id = 2;
        static constexpr size_t c0_profile_id    = 3;
        static constexpr size_t psin_profile_id  = 2 * Mmax + 4;
        static constexpr size_t F_profile_id     = 2 * Mmax + 5;
        static constexpr size_t profile_count    = 2 * Mmax + 6;

        static constexpr size_t c_profile_id(size_t order) { return c0_profile_id + order; }

        static constexpr size_t s_profile_id(size_t order) { return c0_profile_id + Mmax + order; }

        template <size_t Order>
        static consteval size_t c_profile_id()
        {
            static_assert(Order <= Mmax, "c order exceeds Mmax");
            return c_profile_id(Order);
        }

        template <size_t Order>
        static consteval size_t s_profile_id()
        {
            static_assert(Order > 0, "s0 is not a physical sine profile");
            static_assert(Order <= Mmax, "s order exceeds Mmax");
            return s_profile_id(Order);
        }

        static consteval ProfileSlot c_slot(size_t order)
        {
            if (order == 0)
                return C0Slot;
            if (order <= CFamilySlots.size())
                return CFamilySlots[order - 1];
            return absent_slot();
        }

        static consteval ProfileSlot s_slot(size_t order)
        {
            if (order == 0)
                return absent_slot();
            if (order <= SFamilySlots.size())
                return SFamilySlots[order - 1];
            return absent_slot();
        }

        static consteval ProfileSlot slot_for_profile_id(size_t profile_id)
        {
            if (profile_id == h_profile_id)
                return HSlot;
            if (profile_id == v_profile_id)
                return VSlot;
            if (profile_id == kappa_profile_id)
                return KappaSlot;
            if (profile_id >= c0_profile_id && profile_id <= c_profile_id(Mmax))
                return c_slot(profile_id - c0_profile_id);
            if (profile_id >= s_profile_id(1) && profile_id <= s_profile_id(Mmax))
                return s_slot(profile_id - c0_profile_id - Mmax);
            if (profile_id == psin_profile_id)
                return PsinSlot;
            if (profile_id == F_profile_id)
                return FSlot;
            return absent_slot();
        }

        static consteval int profile_L_for_slot(ProfileSlot slot)
        {
            return slot.optimized() ? static_cast<int>(slot.coefficient_count - 1) : -1;
        }

        static consteval bool valid_slot(ProfileSlot slot)
        {
            if (!slot.enabled())
                return slot.coefficient_count == 0;
            if (slot.optimized() && slot.coefficient_count == 0)
                return false;
            return slot.coefficient_count <= Lmax + 1;
        }

        static consteval bool valid_slots()
        {
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                if (!valid_slot(slot_for_profile_id(profile_id)))
                    return false;
            return true;
        }

        static_assert(valid_slots(), "invalid profile slot configuration");

        static consteval size_t compute_active_count()
        {
            size_t count = 0;
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                if (slot_for_profile_id(profile_id).optimized())
                    ++count;
            return count;
        }

        static consteval size_t compute_max_active_len()
        {
            size_t count = 0;
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
            {
                const ProfileSlot slot = slot_for_profile_id(profile_id);
                if (slot.optimized() && slot.coefficient_count > count)
                    count = slot.coefficient_count;
            }
            return count;
        }

        static consteval size_t compute_x_size()
        {
            size_t count = 0;
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
            {
                const ProfileSlot slot = slot_for_profile_id(profile_id);
                if (slot.optimized())
                    count += slot.coefficient_count;
            }
            return count;
        }

        static constexpr size_t active_count   = compute_active_count();
        static constexpr size_t max_active_len = compute_max_active_len();
        static constexpr size_t x_size         = compute_x_size();
        static constexpr bool   layout_profile_first = LayoutProfileFirst;

        static consteval auto make_profile_L()
        {
            std::array<int, profile_count> out{};
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                out[profile_id] = profile_L_for_slot(slot_for_profile_id(profile_id));
            return out;
        }

        static consteval auto make_active_profile_ids()
        {
            std::array<size_t, active_count> out{};
            size_t                           active_slot = 0;
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                if (slot_for_profile_id(profile_id).optimized())
                    out[active_slot++] = profile_id;
            return out;
        }

        static consteval auto make_active_lengths()
        {
            std::array<size_t, active_count> out{};
            size_t                           active_slot = 0;
            for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
            {
                const ProfileSlot slot = slot_for_profile_id(profile_id);
                if (slot.optimized())
                    out[active_slot++] = slot.coefficient_count;
            }
            return out;
        }

        static consteval size_t compute_active_c_order_count()
        {
            size_t count = 0;
            for (size_t order = 0; order <= Mmax; ++order)
                if (c_slot(order).optimized())
                    ++count;
            return count;
        }

        static consteval size_t compute_active_s_order_count()
        {
            size_t count = 0;
            for (size_t order = 1; order <= Mmax; ++order)
                if (s_slot(order).optimized())
                    ++count;
            return count;
        }

        static constexpr size_t active_c_order_count = compute_active_c_order_count();
        static constexpr size_t active_s_order_count = compute_active_s_order_count();

        static consteval auto make_active_c_orders()
        {
            std::array<size_t, active_c_order_count> out{};
            size_t                                   active_slot = 0;
            for (size_t order = 0; order <= Mmax; ++order)
                if (c_slot(order).optimized())
                    out[active_slot++] = order;
            return out;
        }

        static consteval auto make_active_s_orders()
        {
            std::array<size_t, active_s_order_count> out{};
            size_t                                   active_slot = 0;
            for (size_t order = 1; order <= Mmax; ++order)
                if (s_slot(order).optimized())
                    out[active_slot++] = order;
            return out;
        }

        static consteval auto make_coeff_index()
        {
            std::array<std::array<int, max_active_len>, profile_count> out{};
            for (auto& row : out)
                row.fill(-1);

            int x_pos = 0;
            if constexpr (!layout_profile_first)
            {
                for (size_t degree = 0; degree < max_active_len; ++degree)
                {
                    for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                    {
                        const ProfileSlot slot = slot_for_profile_id(profile_id);
                        if (slot.optimized() && degree < slot.coefficient_count)
                            out[profile_id][degree] = x_pos++;
                    }
                }
            }
            else
            {
                for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                {
                    const ProfileSlot slot = slot_for_profile_id(profile_id);
                    if (!slot.optimized())
                        continue;
                    for (size_t degree = 0; degree < slot.coefficient_count; ++degree)
                        out[profile_id][degree] = x_pos++;
                }
            }
            return out;
        }

        static consteval auto make_order_offsets()
        {
            std::array<int, max_active_len + 1> out{};
            int                                 x_pos = 0;
            if constexpr (!layout_profile_first)
            {
                for (size_t degree = 0; degree < max_active_len; ++degree)
                {
                    out[degree] = x_pos;
                    for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                    {
                        const ProfileSlot slot = slot_for_profile_id(profile_id);
                        if (slot.optimized() && degree < slot.coefficient_count)
                            ++x_pos;
                    }
                }
            }
            else
            {
                out.fill(-1);
                for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                {
                    const ProfileSlot slot = slot_for_profile_id(profile_id);
                    if (!slot.optimized())
                        continue;
                    for (size_t degree = 0; degree < slot.coefficient_count; ++degree)
                    {
                        if (out[degree] < 0)
                            out[degree] = x_pos;
                        ++x_pos;
                    }
                }
                for (size_t degree = 0; degree < max_active_len; ++degree)
                {
                    if (out[degree] < 0)
                        out[degree] = x_pos;
                }
            }
            out[max_active_len] = x_pos;
            return out;
        }

        static consteval auto make_c_family_source_profile_ids()
        {
            std::array<int, Mmax + 1> out{};
            for (size_t order = 0; order <= Mmax; ++order)
                out[order] = c_slot(order).optimized() ? static_cast<int>(c_profile_id(order)) : -1;
            return out;
        }

        static consteval auto make_s_family_source_profile_ids()
        {
            std::array<int, Mmax + 1> out{};
            out[0] = -1;
            for (size_t order = 1; order <= Mmax; ++order)
                out[order] = s_slot(order).optimized() ? static_cast<int>(s_profile_id(order)) : -1;
            return out;
        }

        static constexpr auto profile_L                   = make_profile_L();
        static constexpr auto active_profile_ids          = make_active_profile_ids();
        static constexpr auto active_lengths              = make_active_lengths();
        static constexpr auto active_c_orders             = make_active_c_orders();
        static constexpr auto active_s_orders             = make_active_s_orders();
        static constexpr auto coeff_index                 = make_coeff_index();
        static constexpr auto order_offsets               = make_order_offsets();
        static constexpr auto c_family_source_profile_ids = make_c_family_source_profile_ids();
        static constexpr auto s_family_source_profile_ids = make_s_family_source_profile_ids();
    };
} // namespace profiles

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

    template <typename T, size_t Count>
    consteval std::array<T, Count> filled_array(T value)
    {
        std::array<T, Count> out{};
        for (auto& item : out)
            item = value;
        return out;
    }

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
    using tensor::Tensor;
    using tensor::Vector;
    using tensor::uninitialized;

    template <typename Shape>
    struct ProfileEvaluator
    {
        static_assert(Shape::L_max >= 1, "ProfileEvaluator requires at least one stored Chebyshev row");
        static_assert(Shape::K_max >= 2, "ProfileEvaluator requires rho and rho^2 rows");

        using shape = Shape;

        static constexpr size_t basis_rows    = Shape::L_max;
        static constexpr size_t rho_rows      = Shape::K_max;
        static constexpr size_t c_family_size = Shape::c_family_slot_count + 1;
        static constexpr size_t s_family_size = Shape::s_family_slot_count;

        static constexpr ProfileSlot h_slot     = Shape::slot_for_profile_id(Shape::h_profile_id);
        static constexpr ProfileSlot v_slot     = Shape::slot_for_profile_id(Shape::v_profile_id);
        static constexpr ProfileSlot kappa_slot = Shape::slot_for_profile_id(Shape::kappa_profile_id);
        static constexpr ProfileSlot psin_slot  = Shape::slot_for_profile_id(Shape::psin_profile_id);
        static constexpr ProfileSlot F_slot     = Shape::slot_for_profile_id(Shape::F_profile_id);

        static constexpr size_t h_count     = h_slot.coefficient_count;
        static constexpr size_t v_count     = v_slot.coefficient_count;
        static constexpr size_t kappa_count = kappa_slot.coefficient_count;
        static constexpr size_t psin_count  = psin_slot.coefficient_count;
        static constexpr size_t F_count     = F_slot.coefficient_count;

        template <size_t Order>
        static consteval size_t c_count()
        {
            static_assert(Order < c_family_size, "c profile order exceeds configured counts");
            return Shape::c_slot(Order).coefficient_count;
        }

        template <size_t Order>
        static consteval size_t s_count()
        {
            static_assert(Order > 0, "s0 is not a physical sine profile");
            static_assert(Order <= s_family_size, "s profile order exceeds configured counts");
            return Shape::s_slot(Order).coefficient_count;
        }

        template <size_t Order>
        static consteval size_t fourier_power()
        {
            return Order < rho_rows ? Order : rho_rows;
        }

        template <size_t Nr>
            requires(h_slot.enabled() && h_count > 0 && basis_rows + 1 >= h_count)
        static constexpr void update_h(Matrix<double, Nr, 3>&                profiles,
                                       const Vector<double, h_count>&        coeffs,
                                       const Matrix<double, basis_rows, Nr>& T,
                                       const Matrix<double, basis_rows, Nr>& T_r,
                                       const Matrix<double, basis_rows, Nr>& T_rr,
                                       const Matrix<double, rho_rows, Nr>&   rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, h_count>(polys, coeffs, T, T_r, T_rr);
            detail::update_enveloped_profiles<rho_rows>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(v_slot.enabled() && v_count > 0 && basis_rows + 1 >= v_count)
        static constexpr void update_v(Matrix<double, Nr, 3>&                profiles,
                                       const Vector<double, v_count>&        coeffs,
                                       const Matrix<double, basis_rows, Nr>& T,
                                       const Matrix<double, basis_rows, Nr>& T_r,
                                       const Matrix<double, basis_rows, Nr>& T_rr,
                                       const Matrix<double, rho_rows, Nr>&   rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, v_count>(polys, coeffs, T, T_r, T_rr);
            detail::update_enveloped_profiles<rho_rows>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(kappa_slot.enabled() && kappa_count > 0 && basis_rows + 1 >= kappa_count)
        static constexpr void update_kappa(Matrix<double, Nr, 3>&                profiles,
                                           const Vector<double, kappa_count>&    coeffs,
                                           const Matrix<double, basis_rows, Nr>& T,
                                           const Matrix<double, basis_rows, Nr>& T_r,
                                           const Matrix<double, basis_rows, Nr>& T_rr,
                                           const Matrix<double, rho_rows, Nr>&   rhos,
                                           double                                ka) noexcept
        {
            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, kappa_count>(polys, coeffs, T, T_r, T_rr);
            detail::update_kappa_from_polys<rho_rows>(profiles, polys, rhos, ka);
        }

        template <size_t Nr>
            requires(psin_slot.enabled() && psin_count > 0 && basis_rows + 1 >= psin_count)
        static constexpr void update_psin(Matrix<double, Nr, 3>&                profiles,
                                          const Vector<double, psin_count>&     coeffs,
                                          const Matrix<double, basis_rows, Nr>& T,
                                          const Matrix<double, basis_rows, Nr>& T_r,
                                          const Matrix<double, basis_rows, Nr>& T_rr,
                                          const Matrix<double, rho_rows, Nr>&   rhos) noexcept
        {
            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, psin_count>(polys, coeffs, T, T_r, T_rr);
            detail::update_psin_from_polys<rho_rows>(profiles, polys, rhos);
        }

        template <size_t Nr>
            requires(F_slot.enabled() && F_count > 0 && basis_rows + 1 >= F_count)
        static constexpr void update_F(Matrix<double, Nr, 3>&                profiles,
                                       const Vector<double, F_count>&        coeffs,
                                       const Matrix<double, basis_rows, Nr>& T,
                                       const Matrix<double, basis_rows, Nr>& T_r,
                                       const Matrix<double, basis_rows, Nr>& T_rr,
                                       const Matrix<double, rho_rows, Nr>&   rhos,
                                       double                                scale) noexcept
        {
            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, F_count>(polys, coeffs, T, T_r, T_rr);
            detail::update_F_from_polys<rho_rows>(profiles, polys, rhos, scale);
        }

        template <size_t Order, size_t Nr>
            requires(c_count<Order>() > 0 && basis_rows + 1 >= c_count<Order>() && rho_rows >= fourier_power<Order>())
        static constexpr void update_c(Matrix<double, Nr, 3>&                  profiles,
                                       const Vector<double, c_count<Order>()>& coeffs,
                                       const Matrix<double, basis_rows, Nr>&   T,
                                       const Matrix<double, basis_rows, Nr>&   T_r,
                                       const Matrix<double, basis_rows, Nr>&   T_rr,
                                       const Matrix<double, rho_rows, Nr>&     rhos,
                                       double                                  offset) noexcept
        {
            constexpr size_t Count = c_count<Order>();
            constexpr size_t Power = fourier_power<Order>();

            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, Count>(polys, coeffs, T, T_r, T_rr);
            detail::update_fourier_from_polys<rho_rows, Power>(profiles, polys, rhos, offset);
        }

        template <size_t Order, size_t Nr>
            requires(s_count<Order>() > 0 && basis_rows + 1 >= s_count<Order>() && rho_rows >= fourier_power<Order>())
        static constexpr void update_s(Matrix<double, Nr, 3>&                  profiles,
                                       const Vector<double, s_count<Order>()>& coeffs,
                                       const Matrix<double, basis_rows, Nr>&   T,
                                       const Matrix<double, basis_rows, Nr>&   T_r,
                                       const Matrix<double, basis_rows, Nr>&   T_rr,
                                       const Matrix<double, rho_rows, Nr>&     rhos,
                                       double                                  offset) noexcept
        {
            constexpr size_t Count = s_count<Order>();
            constexpr size_t Power = fourier_power<Order>();

            Matrix<double, Nr, 3> polys{uninitialized};
            detail::update_polys<basis_rows, Count>(polys, coeffs, T, T_r, T_rr);
            detail::update_fourier_from_polys<rho_rows, Power>(profiles, polys, rhos, offset);
        }
    };

    template <size_t Lmax,
              size_t Kmax,
              size_t Mmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              bool   LayoutProfileFirst = false>
    struct OptimizedProfileShapeFromCountsWithMmax
    {
        static constexpr size_t Cmax        = CFamilyCounts.size() == 0 ? 0 : CFamilyCounts.size() - 1;
        static constexpr size_t Smax        = SFamilyCounts.size();
        static constexpr size_t active_Mmax = Cmax > Smax ? Cmax : Smax;
        static_assert(Mmax >= active_Mmax, "boundary Mmax must cover configured c/s profile orders");

        static constexpr auto c_family_slots = tail_optimized_slots_from_counts<CFamilyCounts>();
        static constexpr auto s_family_slots = optimized_slots_from_counts<SFamilyCounts>();

        using type = ProfileShape<Lmax,
                                  Kmax,
                                  Mmax,
                                  optimized_slot_from_count(HCount),
                                  optimized_slot_from_count(VCount),
                                  optimized_slot_from_count(KappaCount),
                                  first_optimized_slot_from_counts<CFamilyCounts>(),
                                  optimized_slot_from_count(PsinCount),
                                  optimized_slot_from_count(FCount),
                                  c_family_slots,
                                  s_family_slots,
                                  LayoutProfileFirst>;
    };

    template <size_t Lmax,
              size_t Kmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              bool   LayoutProfileFirst = false>
    struct OptimizedProfileShapeFromCounts
    {
        static constexpr size_t Cmax = CFamilyCounts.size() == 0 ? 0 : CFamilyCounts.size() - 1;
        static constexpr size_t Smax = SFamilyCounts.size();
        static constexpr size_t Mmax = Cmax > Smax ? Cmax : Smax;

        using type = typename OptimizedProfileShapeFromCountsWithMmax<Lmax,
                                                                      Kmax,
                                                                      Mmax,
                                                                      HCount,
                                                                      VCount,
                                                                      KappaCount,
                                                                      PsinCount,
                                                                      FCount,
                                                                      CFamilyCounts,
                                                                      SFamilyCounts,
                                                                      LayoutProfileFirst>::type;
    };

    template <size_t Lmax,
              size_t Kmax,
              size_t Mmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              bool   LayoutProfileFirst = false>
    using OptimizedProfileShapeFromCountsWithMmaxT =
        typename OptimizedProfileShapeFromCountsWithMmax<Lmax,
                                                         Kmax,
                                                         Mmax,
                                                         HCount,
                                                         VCount,
                                                         KappaCount,
                                                         PsinCount,
                                                         FCount,
                                                         CFamilyCounts,
                                                         SFamilyCounts,
                                                         LayoutProfileFirst>::type;

    template <size_t Lmax,
              size_t Kmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              bool   LayoutProfileFirst = false>
    using OptimizedProfileShapeFromCountsT = typename OptimizedProfileShapeFromCounts<Lmax,
                                                                                      Kmax,
                                                                                      HCount,
                                                                                      VCount,
                                                                                      KappaCount,
                                                                                      PsinCount,
                                                                                      FCount,
                                                                                      CFamilyCounts,
                                                                                      SFamilyCounts,
                                                                                      LayoutProfileFirst>::type;

    template <size_t Lmax,
              size_t Kmax,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts>
    struct Profiles : ProfileEvaluator<OptimizedProfileShapeFromCountsT<Lmax,
                                                                        Kmax,
                                                                        HCount,
                                                                        VCount,
                                                                        KappaCount,
                                                                        PsinCount,
                                                                        FCount,
                                                                        CFamilyCounts,
                                                                        SFamilyCounts>>
    {
    };

    template <typename Shape>
    struct ProfileRuntimeParams
    {
        std::array<double, Shape::profile_count> offsets{};
        std::array<double, Shape::profile_count> scales = detail::filled_array<double, Shape::profile_count>(1.0);
        std::array<size_t, Shape::profile_count> powers{};
        std::array<size_t, Shape::profile_count> envelope_powers{};
        std::array<double, Shape::profile_count> amplitude_powers =
            detail::filled_array<double, Shape::profile_count>(1.0);
    };

    template <typename Shape, typename GridType>
    struct RuntimeProfiles
    {
        static_assert(Shape::L_max == GridType::basis_rows, "RuntimeProfiles requires matching basis rows");
        static_assert(Shape::K_max == GridType::rho_power_rows, "RuntimeProfiles requires matching rho rows");
        static_assert(Shape::M_max + 1 == GridType::harmonic_rows, "RuntimeProfiles requires matching harmonics");

        using shape     = Shape;
        using grid      = GridType;
        using evaluator = ProfileEvaluator<Shape>;

        static constexpr size_t radial_nodes          = GridType::radial_nodes;
        static constexpr size_t profile_field_count   = Shape::profile_count;
        static constexpr size_t family_field_count    = Shape::M_max + 1;
        static constexpr size_t phase_component_count = 6;
        static constexpr size_t phase_tb              = 0;
        static constexpr size_t phase_tb_r            = 1;
        static constexpr size_t phase_tb_t            = 2;
        static constexpr size_t phase_tb_rr           = 3;
        static constexpr size_t phase_tb_rt           = 4;
        static constexpr size_t phase_tb_tt           = 5;

        using ProfileField  = Matrix<double, radial_nodes, 3>;
        using ProfileSlab   = Tensor<double, profile_field_count, radial_nodes, 3>;
        using FamilySlab    = Tensor<double, family_field_count, radial_nodes, 3>;
        using PhaseBaseSlab = Tensor<double, radial_nodes, GridType::theta_rows, phase_component_count>;

        ProfileSlab                            profile_fields{};
        ProfileSlab                            profile_rp_fields{};
        ProfileSlab                            profile_env_fields{};
        FamilySlab                             c_family_fields{};
        FamilySlab                             s_family_fields{};
        FamilySlab                             c_family_base_fields{};
        FamilySlab                             s_family_base_fields{};
        PhaseBaseSlab                          boundary_phase_base{};
        std::array<size_t, family_field_count> boundary_c_orders{};
        std::array<size_t, family_field_count> boundary_s_orders{};
        size_t                                 boundary_c_order_count = 0;
        size_t                                 boundary_s_order_count = 0;

        constexpr void clear() noexcept
        {
            profile_fields.clear();
            profile_rp_fields.clear();
            profile_env_fields.clear();
            c_family_fields.clear();
            s_family_fields.clear();
            c_family_base_fields.clear();
            s_family_base_fields.clear();
            boundary_phase_base.clear();
            boundary_c_order_count = 0;
            boundary_s_order_count = 0;
        }

        constexpr double& profile_field(size_t profile_id, size_t node, size_t component) noexcept
        {
            return profile_fields(profile_id, node, component);
        }

        constexpr double profile_field(size_t profile_id, size_t node, size_t component) const noexcept
        {
            return profile_fields(profile_id, node, component);
        }

        template <size_t ProfileId>
        constexpr double& profile_field(size_t node, size_t component) noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");
            return profile_fields(ProfileId, node, component);
        }

        template <size_t ProfileId>
        constexpr double profile_field(size_t node, size_t component) const noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");
            return profile_fields(ProfileId, node, component);
        }

        template <size_t ProfileId>
        constexpr ProfileField profile_matrix() const noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");

            ProfileField out{uninitialized};
            for (size_t node = 0; node < radial_nodes; ++node)
                for (size_t component = 0; component < 3; ++component)
                    out(node, component) = profile_fields(ProfileId, node, component);
            return out;
        }

        template <size_t Order>
        constexpr double& c_family_field(size_t node, size_t component) noexcept
        {
            static_assert(Order < family_field_count, "c-family order exceeds runtime family slab");
            return c_family_fields(Order, node, component);
        }

        template <size_t Order>
        constexpr double c_family_field(size_t node, size_t component) const noexcept
        {
            static_assert(Order < family_field_count, "c-family order exceeds runtime family slab");
            return c_family_fields(Order, node, component);
        }

        template <size_t Order>
        constexpr double& s_family_field(size_t node, size_t component) noexcept
        {
            static_assert(Order < family_field_count, "s-family order exceeds runtime family slab");
            return s_family_fields(Order, node, component);
        }

        template <size_t Order>
        constexpr double s_family_field(size_t node, size_t component) const noexcept
        {
            static_assert(Order < family_field_count, "s-family order exceeds runtime family slab");
            return s_family_fields(Order, node, component);
        }

        constexpr double boundary_phase_base_field(size_t node, size_t theta_node, size_t component) const noexcept
        {
            return boundary_phase_base(node, theta_node, component);
        }

        constexpr void refresh_fixed(const ProfileRuntimeParams<Shape>& params) noexcept
        {
            refresh_fixed_profile<0>(params);
            refresh_boundary_family_base(params);
            refresh_boundary_phase_base();
        }

        constexpr void refresh_active(std::span<const double, Shape::x_size> x,
                                      const ProfileRuntimeParams<Shape>&     params) noexcept
        {
            refresh_h_active(x);
            refresh_v_active(x);
            refresh_kappa_active(x, params);
            refresh_psin_active(x);
            refresh_F_active(x, params);
            refresh_c_active<0>(x, params);
            refresh_s_active<1>(x, params);
            refresh_fourier_family_fields();
        }

        constexpr void refresh_fourier_family_fields() noexcept
        {
            // Active family rows are overwritten below; boundary-only rows stay zero
            // here and enter geometry through boundary_phase_base.
            refresh_c_family_order<0>();
            refresh_s_family_order<1>();
        }

        constexpr void load_fixed_from(const RuntimeProfiles& fixed_profiles) noexcept
        {
            load_fixed_profile<0>(fixed_profiles);
            copy_boundary_base_from(fixed_profiles);
            c_family_fields.clear();
            s_family_fields.clear();
        }

    private:
        template <size_t ProfileId, size_t Count>
        static constexpr Vector<double, Count> coefficients_from_x(std::span<const double, Shape::x_size> x) noexcept
        {
            Vector<double, Count> out{uninitialized};
            for (size_t degree = 0; degree < Count; ++degree)
            {
                const int x_index = Shape::coeff_index[ProfileId][degree];
                assert(x_index >= 0);
                out[degree] = x[static_cast<size_t>(x_index)];
            }
            return out;
        }

        template <size_t ProfileId>
        constexpr void store_profile(const ProfileField& values) noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");

            for (size_t node = 0; node < radial_nodes; ++node)
                for (size_t component = 0; component < 3; ++component)
                    profile_fields(ProfileId, node, component) = values(node, component);
        }

        template <size_t ProfileId>
        constexpr void fill_profile(double value, double radial, double radial2) noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");

            for (size_t node = 0; node < radial_nodes; ++node)
            {
                profile_fields(ProfileId, node, 0) = value;
                profile_fields(ProfileId, node, 1) = radial;
                profile_fields(ProfileId, node, 2) = radial2;
            }
        }

        template <size_t ProfileId>
        constexpr void copy_profile_from(const RuntimeProfiles& fixed_profiles) noexcept
        {
            static_assert(ProfileId < profile_field_count, "profile id exceeds runtime profile slab");

            for (size_t node = 0; node < radial_nodes; ++node)
                for (size_t component = 0; component < 3; ++component)
                    profile_fields(ProfileId, node, component) =
                        fixed_profiles.profile_fields(ProfileId, node, component);
        }

        template <size_t ProfileId>
        constexpr void load_fixed_profile(const RuntimeProfiles& fixed_profiles) noexcept
        {
            if constexpr (ProfileId < profile_field_count)
            {
                constexpr ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.fixed())
                    copy_profile_from<ProfileId>(fixed_profiles);
                load_fixed_profile<ProfileId + 1>(fixed_profiles);
            }
        }

        constexpr void copy_boundary_base_from(const RuntimeProfiles& fixed_profiles) noexcept
        {
            boundary_c_order_count = fixed_profiles.boundary_c_order_count;
            boundary_s_order_count = fixed_profiles.boundary_s_order_count;
            for (size_t active = 0; active < boundary_c_order_count; ++active)
                boundary_c_orders[active] = fixed_profiles.boundary_c_orders[active];
            for (size_t active = 0; active < boundary_s_order_count; ++active)
                boundary_s_orders[active] = fixed_profiles.boundary_s_orders[active];

            for (size_t order = 0; order < family_field_count; ++order)
                for (size_t node = 0; node < radial_nodes; ++node)
                    for (size_t component = 0; component < 3; ++component)
                    {
                        c_family_base_fields(order, node, component) =
                            fixed_profiles.c_family_base_fields(order, node, component);
                        s_family_base_fields(order, node, component) =
                            fixed_profiles.s_family_base_fields(order, node, component);
                    }

            for (size_t node = 0; node < radial_nodes; ++node)
                for (size_t theta_node = 0; theta_node < GridType::theta_rows; ++theta_node)
                    for (size_t component = 0; component < phase_component_count; ++component)
                        boundary_phase_base(node, theta_node, component) =
                            fixed_profiles.boundary_phase_base(node, theta_node, component);
        }

        static constexpr size_t fourier_power_runtime(size_t order) noexcept
        {
            return order < Shape::K_max ? order : Shape::K_max;
        }

        static constexpr detail::ProfileValues rho_power_rows_runtime(size_t power, size_t node) noexcept
        {
            if (power == 0)
                return {1.0, 0.0, 0.0};
            if (power == 1)
                return {GridType::rhos(0, node), 1.0, 0.0};

            const double rho_pm2 = power == 2 ? 1.0 : GridType::rhos(power - 3, node);
            const double rho_pm1 = GridType::rhos(power - 2, node);
            const double rho_p   = GridType::rhos(power - 1, node);
            return {
                rho_p,
                static_cast<double>(power) * rho_pm1,
                static_cast<double>(power * (power - 1)) * rho_pm2,
            };
        }

        static constexpr void
        store_boundary_order(FamilySlab& family, size_t order, size_t node, double offset) noexcept
        {
            const detail::ProfileValues rp = rho_power_rows_runtime(fourier_power_runtime(order), node);
            family(order, node, 0)         = offset * rp.value;
            family(order, node, 1)         = offset * rp.diff;
            family(order, node, 2)         = offset * rp.diff2;
        }

        static constexpr void clear_boundary_order(FamilySlab& family, size_t order) noexcept
        {
            for (size_t node = 0; node < radial_nodes; ++node)
                for (size_t component = 0; component < 3; ++component)
                    family(order, node, component) = 0.0;
        }

        constexpr void remember_boundary_c_order(size_t order) noexcept
        {
            boundary_c_orders[boundary_c_order_count++] = order;
        }

        constexpr void remember_boundary_s_order(size_t order) noexcept
        {
            boundary_s_orders[boundary_s_order_count++] = order;
        }

        constexpr void refresh_boundary_family_base(const ProfileRuntimeParams<Shape>& params) noexcept
        {
            for (size_t active = 0; active < boundary_c_order_count; ++active)
                clear_boundary_order(c_family_base_fields, boundary_c_orders[active]);
            for (size_t active = 0; active < boundary_s_order_count; ++active)
                clear_boundary_order(s_family_base_fields, boundary_s_orders[active]);
            boundary_c_order_count = 0;
            boundary_s_order_count = 0;
            for (size_t order = 0; order <= Shape::M_max; ++order)
            {
                const size_t profile_id = Shape::c_profile_id(order);
                const double offset     = params.offsets[profile_id] * params.scales[profile_id];
                if (abs(offset) <= boundary_amplitude_prune_threshold)
                    continue;
                remember_boundary_c_order(order);
                for (size_t node = 0; node < radial_nodes; ++node)
                    store_boundary_order(c_family_base_fields, order, node, offset);
            }
            for (size_t order = 1; order <= Shape::M_max; ++order)
            {
                const size_t profile_id = Shape::s_profile_id(order);
                const double offset     = params.offsets[profile_id] * params.scales[profile_id];
                if (abs(offset) <= boundary_amplitude_prune_threshold)
                    continue;
                remember_boundary_s_order(order);
                for (size_t node = 0; node < radial_nodes; ++node)
                    store_boundary_order(s_family_base_fields, order, node, offset);
            }
        }

        constexpr void refresh_boundary_phase_base() noexcept
        {
            for (size_t node = 0; node < radial_nodes; ++node)
            {
                const double c0_i    = c_family_base_fields(0, node, 0);
                const double c0_r_i  = c_family_base_fields(0, node, 1);
                const double c0_rr_i = c_family_base_fields(0, node, 2);

                for (size_t theta_node = 0; theta_node < GridType::theta_rows; ++theta_node)
                {
                    double tb    = GridType::theta[theta_node] + c0_i;
                    double tb_r  = c0_r_i;
                    double tb_t  = 1.0;
                    double tb_rr = c0_rr_i;
                    double tb_rt = 0.0;
                    double tb_tt = 0.0;

                    for (size_t active = 0; active < boundary_c_order_count; ++active)
                    {
                        const size_t order = boundary_c_orders[active];
                        if (order == 0)
                            continue;
                        const double c_i       = c_family_base_fields(order, node, 0);
                        const double c_r_i     = c_family_base_fields(order, node, 1);
                        const double c_rr_i    = c_family_base_fields(order, node, 2);
                        const double cos_kt    = GridType::cos_mtheta(order, theta_node);
                        const double k_sin_kt  = GridType::m_sin_mtheta(order, theta_node);
                        const double k2_cos_kt = GridType::m2_cos_mtheta(order, theta_node);
                        tb += c_i * cos_kt;
                        tb_r += c_r_i * cos_kt;
                        tb_t -= c_i * k_sin_kt;
                        tb_rr += c_rr_i * cos_kt;
                        tb_rt -= c_r_i * k_sin_kt;
                        tb_tt -= c_i * k2_cos_kt;
                    }

                    for (size_t active = 0; active < boundary_s_order_count; ++active)
                    {
                        const size_t order     = boundary_s_orders[active];
                        const double s_i       = s_family_base_fields(order, node, 0);
                        const double s_r_i     = s_family_base_fields(order, node, 1);
                        const double s_rr_i    = s_family_base_fields(order, node, 2);
                        const double sin_kt    = GridType::sin_mtheta(order, theta_node);
                        const double k_cos_kt  = GridType::m_cos_mtheta(order, theta_node);
                        const double k2_sin_kt = GridType::m2_sin_mtheta(order, theta_node);
                        tb += s_i * sin_kt;
                        tb_r += s_r_i * sin_kt;
                        tb_t += s_i * k_cos_kt;
                        tb_rr += s_rr_i * sin_kt;
                        tb_rt += s_r_i * k_cos_kt;
                        tb_tt -= s_i * k2_sin_kt;
                    }

                    boundary_phase_base(node, theta_node, phase_tb)    = tb;
                    boundary_phase_base(node, theta_node, phase_tb_r)  = tb_r;
                    boundary_phase_base(node, theta_node, phase_tb_t)  = tb_t;
                    boundary_phase_base(node, theta_node, phase_tb_rr) = tb_rr;
                    boundary_phase_base(node, theta_node, phase_tb_rt) = tb_rt;
                    boundary_phase_base(node, theta_node, phase_tb_tt) = tb_tt;
                }
            }
        }

        template <size_t ProfileId>
        constexpr void refresh_fixed_profile(const ProfileRuntimeParams<Shape>& params) noexcept
        {
            if constexpr (ProfileId < profile_field_count)
            {
                constexpr ProfileSlot slot = Shape::slot_for_profile_id(ProfileId);
                if constexpr (slot.fixed())
                    fill_profile<ProfileId>(params.offsets[ProfileId] * params.scales[ProfileId], 0.0, 0.0);
                refresh_fixed_profile<ProfileId + 1>(params);
            }
        }

        constexpr void refresh_h_active(std::span<const double, Shape::x_size> x) noexcept
        {
            if constexpr (Shape::slot_for_profile_id(Shape::h_profile_id).optimized())
            {
                constexpr size_t profile_id = Shape::h_profile_id;
                constexpr size_t count      = evaluator::h_count;
                const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                ProfileField     out{uninitialized};
                evaluator::update_h(out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos);
                store_profile<profile_id>(out);
            }
        }

        constexpr void refresh_v_active(std::span<const double, Shape::x_size> x) noexcept
        {
            if constexpr (Shape::slot_for_profile_id(Shape::v_profile_id).optimized())
            {
                constexpr size_t profile_id = Shape::v_profile_id;
                constexpr size_t count      = evaluator::v_count;
                const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                ProfileField     out{uninitialized};
                evaluator::update_v(out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos);
                store_profile<profile_id>(out);
            }
        }

        constexpr void refresh_kappa_active(std::span<const double, Shape::x_size> x,
                                            const ProfileRuntimeParams<Shape>&     params) noexcept
        {
            if constexpr (Shape::slot_for_profile_id(Shape::kappa_profile_id).optimized())
            {
                constexpr size_t profile_id = Shape::kappa_profile_id;
                constexpr size_t count      = evaluator::kappa_count;
                const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                ProfileField     out{uninitialized};
                evaluator::update_kappa(out,
                                        coeffs,
                                        GridType::T,
                                        GridType::T_r,
                                        GridType::T_rr,
                                        GridType::rhos,
                                        params.offsets[profile_id]);
                store_profile<profile_id>(out);
            }
        }

        constexpr void refresh_psin_active(std::span<const double, Shape::x_size> x) noexcept
        {
            if constexpr (Shape::slot_for_profile_id(Shape::psin_profile_id).optimized())
            {
                constexpr size_t profile_id = Shape::psin_profile_id;
                constexpr size_t count      = evaluator::psin_count;
                const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                ProfileField     out{uninitialized};
                evaluator::update_psin(out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos);
                store_profile<profile_id>(out);
            }
        }

        constexpr void refresh_F_active(std::span<const double, Shape::x_size> x,
                                        const ProfileRuntimeParams<Shape>&     params) noexcept
        {
            if constexpr (Shape::slot_for_profile_id(Shape::F_profile_id).optimized())
            {
                constexpr size_t profile_id = Shape::F_profile_id;
                constexpr size_t count      = evaluator::F_count;
                const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                ProfileField     out{uninitialized};
                evaluator::update_F(
                    out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos, params.scales[profile_id]);
                store_profile<profile_id>(out);
            }
        }

        template <size_t Order>
        constexpr void refresh_c_active(std::span<const double, Shape::x_size> x,
                                        const ProfileRuntimeParams<Shape>&     params) noexcept
        {
            if constexpr (Order < evaluator::c_family_size)
            {
                if constexpr (Order < evaluator::c_family_size && Shape::c_slot(Order).optimized())
                {
                    constexpr size_t profile_id = Shape::template c_profile_id<Order>();
                    constexpr size_t count      = evaluator::template c_count<Order>();
                    const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                    ProfileField     out{uninitialized};
                    evaluator::template update_c<Order>(
                        out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos, 0.0);
                    store_profile<profile_id>(out);
                }
                refresh_c_active<Order + 1>(x, params);
            }
        }

        template <size_t Order>
        constexpr void refresh_s_active(std::span<const double, Shape::x_size> x,
                                        const ProfileRuntimeParams<Shape>&     params) noexcept
        {
            if constexpr (Order <= evaluator::s_family_size)
            {
                if constexpr (Order <= evaluator::s_family_size && Shape::s_slot(Order).optimized())
                {
                    constexpr size_t profile_id = Shape::template s_profile_id<Order>();
                    constexpr size_t count      = evaluator::template s_count<Order>();
                    const auto       coeffs     = coefficients_from_x<profile_id, count>(x);
                    ProfileField     out{uninitialized};
                    evaluator::template update_s<Order>(
                        out, coeffs, GridType::T, GridType::T_r, GridType::T_rr, GridType::rhos, 0.0);
                    store_profile<profile_id>(out);
                }
                refresh_s_active<Order + 1>(x, params);
            }
        }

        template <size_t ProfileId, size_t Order>
        constexpr void copy_profile_to_family(FamilySlab& family) noexcept
        {
            for (size_t node = 0; node < radial_nodes; ++node)
            {
                for (size_t component = 0; component < 3; ++component)
                {
                    const double value             = profile_fields(ProfileId, node, component);
                    family(Order, node, component) = value;
                }
            }
        }

        template <size_t Order>
        constexpr void refresh_c_family_order() noexcept
        {
            if constexpr (Order < evaluator::c_family_size)
            {
                constexpr int profile_id = Shape::c_family_source_profile_ids[Order];
                if constexpr (profile_id >= 0)
                    copy_profile_to_family<static_cast<size_t>(profile_id), Order>(c_family_fields);
                refresh_c_family_order<Order + 1>();
            }
        }

        template <size_t Order>
        constexpr void refresh_s_family_order() noexcept
        {
            if constexpr (Order <= evaluator::s_family_size)
            {
                constexpr int profile_id = Shape::s_family_source_profile_ids[Order];
                if constexpr (profile_id >= 0)
                    copy_profile_to_family<static_cast<size_t>(profile_id), Order>(s_family_fields);
                refresh_s_family_order<Order + 1>();
            }
        }
    };
} // namespace profiles
