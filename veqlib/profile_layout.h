#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace profiles
{
    using std::size_t;

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
              auto        SFamilySlots>
    struct ProfileShape
    {
        static_assert(Lmax >= 1, "ProfileShape requires at least one stored Chebyshev row");
        static_assert(Kmax >= 2, "ProfileShape requires rho and rho^2 rows");
        static_assert(CFamilySlots.size() <= Mmax, "c-family slots exceed Mmax");
        static_assert(SFamilySlots.size() <= Mmax, "s-family slots exceed Mmax");

        static constexpr size_t L_max = Lmax;
        static constexpr size_t K_max = Kmax;
        static constexpr size_t M_max = Mmax;

        static constexpr size_t h_profile_id     = 0;
        static constexpr size_t v_profile_id     = 1;
        static constexpr size_t kappa_profile_id = 2;
        static constexpr size_t c0_profile_id    = 3;
        static constexpr size_t psin_profile_id  = 2 * Mmax + 4;
        static constexpr size_t F_profile_id     = 2 * Mmax + 5;
        static constexpr size_t profile_count    = 2 * Mmax + 6;

        static consteval size_t c_profile_id(size_t order) { return c0_profile_id + order; }

        static consteval size_t s_profile_id(size_t order) { return c0_profile_id + Mmax + order; }

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

        static consteval auto make_coeff_index()
        {
            std::array<std::array<int, max_active_len>, profile_count> out{};
            for (auto& row : out)
                row.fill(-1);

            int x_pos = 0;
            for (size_t degree = 0; degree < max_active_len; ++degree)
            {
                for (size_t profile_id = 0; profile_id < profile_count; ++profile_id)
                {
                    const ProfileSlot slot = slot_for_profile_id(profile_id);
                    if (slot.optimized() && degree < slot.coefficient_count)
                        out[profile_id][degree] = x_pos++;
                }
            }
            return out;
        }

        static consteval auto make_order_offsets()
        {
            std::array<int, max_active_len + 1> out{};
            int                                 x_pos = 0;
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
            out[max_active_len] = x_pos;
            return out;
        }

        static consteval auto make_c_family_source_profile_ids()
        {
            std::array<int, Mmax + 1> out{};
            for (size_t order = 0; order <= Mmax; ++order)
                out[order] = c_slot(order).enabled() ? static_cast<int>(c_profile_id(order)) : -1;
            return out;
        }

        static consteval auto make_s_family_source_profile_ids()
        {
            std::array<int, Mmax + 1> out{};
            out[0] = -1;
            for (size_t order = 1; order <= Mmax; ++order)
                out[order] = s_slot(order).enabled() ? static_cast<int>(s_profile_id(order)) : -1;
            return out;
        }

        static constexpr auto profile_L                   = make_profile_L();
        static constexpr auto active_profile_ids          = make_active_profile_ids();
        static constexpr auto active_lengths              = make_active_lengths();
        static constexpr auto coeff_index                 = make_coeff_index();
        static constexpr auto order_offsets               = make_order_offsets();
        static constexpr auto c_family_source_profile_ids = make_c_family_source_profile_ids();
        static constexpr auto s_family_source_profile_ids = make_s_family_source_profile_ids();
    };
} // namespace profiles
