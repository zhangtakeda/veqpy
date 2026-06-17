#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <type_traits>

namespace tensor::detail
{
    inline constexpr std::size_t simd_alignment = 64;

    template <typename T>
    inline constexpr std::size_t tensor_alignment =
        std::is_floating_point_v<T> ? (alignof(T) > simd_alignment ? alignof(T) : simd_alignment) : alignof(T);

    template <std::size_t... values>
    inline constexpr std::size_t static_product = (std::size_t{1} * ... * values);

    template <std::size_t rank>
    constexpr std::array<std::size_t, rank> make_strides(const std::array<std::size_t, rank>& shape)
    {
        std::array<std::size_t, rank> strides{};
        strides[rank - 1] = 1;
        for (std::size_t i = rank - 1; i > 0; --i)
            strides[i - 1] = strides[i] * shape[i];
        return strides;
    }
}

namespace tensor
{
    inline constexpr struct Uninitialized
    {
    } uninitialized{};

    template <typename T, std::size_t... extents>
    struct Tensor;

    template <typename T, std::size_t size>
    using Vector = Tensor<T, size, 1>;

    template <typename T, std::size_t rows, std::size_t cols = rows>
    using Matrix = Tensor<T, rows, cols>;

    template <typename T, std::size_t... extents>
    struct Tensor
    {
        using value_type      = T;
        using size_type       = std::size_t;
        using pointer         = T*;
        using const_pointer   = const T*;
        using reference       = T&;
        using const_reference = const T&;
        using iterator        = T*;
        using const_iterator  = const T*;

        static constexpr size_type rank          = sizeof...(extents);
        static constexpr size_type count         = detail::static_product<extents...>;
        static constexpr size_type alignment     = detail::tensor_alignment<T>;
        static constexpr size_type storage_bytes = count * sizeof(T);

        static constexpr std::array<size_type, rank> shape   = {extents...};
        static constexpr std::array<size_type, rank> strides = detail::make_strides(shape);

        static_assert(rank > 0, "Tensor must have at least one dimension");
        static_assert(count > 0, "Tensor must contain at least one element");
        static_assert(std::is_same_v<T, std::remove_cv_t<T>>, "Tensor element type must be a non-cv-qualified type");
        static_assert(std::is_arithmetic_v<T>, "Tensor element type must be an arithmetic type");

        alignas(alignment) T values[count];

        constexpr Tensor() : values{} {}

        explicit Tensor(Uninitialized) noexcept {}

        constexpr explicit Tensor(const T& value) : values{} { fill(value); }

        constexpr Tensor(std::initializer_list<T> init) : values{}
        {
            assert(init.size() <= count);
            std::copy(init.begin(), init.end(), begin());
        }

        constexpr pointer data() noexcept { return values; }

        constexpr const_pointer data() const noexcept { return values; }

        pointer aligned_data() noexcept { return static_cast<pointer>(__builtin_assume_aligned(values, alignment)); }

        const_pointer aligned_data() const noexcept
        {
            return static_cast<const_pointer>(__builtin_assume_aligned(values, alignment));
        }

        constexpr iterator begin() noexcept { return values; }

        constexpr const_iterator begin() const noexcept { return values; }

        constexpr iterator end() noexcept { return values + count; }

        constexpr const_iterator end() const noexcept { return values + count; }

        constexpr std::span<T, count> span() noexcept { return {values, count}; }

        constexpr std::span<const T, count> span() const noexcept { return {values, count}; }

        constexpr reference operator[](size_type index) noexcept
        {
            assert(index < count);
            return values[index];
        }

        constexpr const_reference operator[](size_type index) const noexcept
        {
            assert(index < count);
            return values[index];
        }

        template <typename... Indices>
            requires(sizeof...(Indices) == rank)
        constexpr reference operator()(Indices... indices) noexcept
        {
            return values[linear_index(static_cast<size_type>(indices)...)];
        }

        template <typename... Indices>
            requires(sizeof...(Indices) == rank)
        constexpr const_reference operator()(Indices... indices) const noexcept
        {
            return values[linear_index(static_cast<size_type>(indices)...)];
        }

        constexpr void clear() noexcept { fill(T{}); }

        constexpr void fill(const T& value) noexcept { std::fill(begin(), end(), value); }

        constexpr bool is_aligned() const noexcept { return reinterpret_cast<std::uintptr_t>(values) % alignment == 0; }

    private:
        template <typename... Indices>
        static constexpr size_type linear_index(Indices... indices) noexcept
        {
            const std::array<size_type, rank> values_ = {indices...};
            size_type                         result  = 0;
            for (size_type axis = 0; axis < rank; ++axis)
            {
                assert(values_[axis] < shape[axis]);
                result += values_[axis] * strides[axis];
            }
            return result;
        }
    };
} // namespace tensor
