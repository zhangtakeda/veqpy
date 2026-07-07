#pragma once

// Fixed-size tensor containers for generated Cxx Kernel artifacts.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>
#include <type_traits>

namespace tensor::detail
{
    using std::array;
    using std::size_t;

    inline constexpr size_t simd_alignment = 64;

    inline constexpr struct Uninitialized
    {
    } uninitialized;

    template <typename T>
    inline constexpr size_t tensor_alignment =
        std::is_floating_point_v<T> ? (alignof(T) > simd_alignment ? alignof(T) : simd_alignment) : alignof(T);

    template <size_t... values>
    inline constexpr size_t static_product = (size_t{1} * ... * values);

    template <size_t rank>
    constexpr array<size_t, rank> make_strides(const array<size_t, rank>& shape)
    {
        array<size_t, rank> strides;
        strides[rank - 1] = 1;
        for (size_t i = rank - 1; i > 0; --i)
            strides[i - 1] = strides[i] * shape[i];
        return strides;
    }

    template <typename T, size_t... Extents>
    struct Tensor
    {
        using value_type      = T;
        using size_type       = size_t;
        using pointer         = T*;
        using const_pointer   = const T*;
        using reference       = T&;
        using const_reference = const T&;
        using iterator        = T*;
        using const_iterator  = const T*;

        static constexpr size_type rank          = sizeof...(Extents);
        static constexpr size_type count         = static_product<Extents...>;
        static constexpr size_type alignment     = tensor_alignment<T>;
        static constexpr size_type storage_bytes = count * sizeof(T);

        static constexpr array<size_type, rank> shape   = {Extents...};
        static constexpr array<size_type, rank> strides = make_strides(shape);

        static_assert(rank > 0, "Tensor must have at least one dimension");
        static_assert(count > 0, "Tensor must contain at least one element");
        static_assert(std::is_same_v<T, std::remove_cv_t<T>>, "Tensor element type must be a non-cv-qualified type");
        static_assert(std::is_arithmetic_v<T>, "Tensor element type must be an arithmetic type");

        alignas(alignment) T values[count];

        constexpr Tensor() : values{} {}

        explicit constexpr Tensor(Uninitialized) noexcept {}

        constexpr explicit Tensor(const T& value) { fill(value); }

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

        template <typename U>
        constexpr Tensor& operator+=(const Tensor<U, Extents...>& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] += rhs[i];
            return *this;
        }

        template <typename U>
            requires(std::is_arithmetic_v<U>)
        constexpr Tensor& operator+=(const U& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] += rhs;
            return *this;
        }

        template <typename U>
        constexpr Tensor& operator-=(const Tensor<U, Extents...>& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] -= rhs[i];
            return *this;
        }

        template <typename U>
            requires(std::is_arithmetic_v<U>)
        constexpr Tensor& operator-=(const U& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] -= rhs;
            return *this;
        }

        template <typename U>
        constexpr Tensor& operator*=(const Tensor<U, Extents...>& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] *= rhs[i];
            return *this;
        }

        template <typename U>
            requires(std::is_arithmetic_v<U>)
        constexpr Tensor& operator*=(const U& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] *= rhs;
            return *this;
        }

        template <typename U>
        constexpr Tensor& operator/=(const Tensor<U, Extents...>& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] /= rhs[i];
            return *this;
        }

        template <typename U>
            requires(std::is_arithmetic_v<U>)
        constexpr Tensor& operator/=(const U& rhs) noexcept
        {
            for (size_type i = 0; i < count; ++i)
                values[i] /= rhs;
            return *this;
        }

    private:
        template <typename... Indices>
        static constexpr size_type linear_index(Indices... indices) noexcept
        {
            const array<size_type, rank> values_ = {indices...};
            size_type                    result  = 0;
            for (size_type axis = 0; axis < rank; ++axis)
            {
                assert(values_[axis] < shape[axis]);
                result += values_[axis] * strides[axis];
            }
            return result;
        }
    };

    template <typename T, size_t size>
    using Vector = Tensor<T, size, 1>;

    template <typename T, size_t rows, size_t cols = rows>
    using Matrix = Tensor<T, rows, cols>;

    template <typename Lhs, typename Rhs, size_t... Extents, typename Operation>
    constexpr auto
    elementwise_tensor(const Tensor<Lhs, Extents...>& lhs, const Tensor<Rhs, Extents...>& rhs, Operation operation)
    {
        using Out = std::common_type_t<Lhs, Rhs>;
        Tensor<Out, Extents...> out{uninitialized};
        for (size_t i = 0; i < out.count; ++i)
            out[i] = operation(lhs[i], rhs[i]);
        return out;
    }

    template <typename Lhs, typename Rhs, size_t... Extents, typename Operation>
        requires std::is_arithmetic_v<Rhs>
    constexpr auto elementwise_scalar_right(const Tensor<Lhs, Extents...>& lhs, Rhs rhs, Operation operation)
    {
        using Out = std::common_type_t<Lhs, Rhs>;
        Tensor<Out, Extents...> out{uninitialized};
        for (size_t i = 0; i < out.count; ++i)
            out[i] = operation(lhs[i], rhs);
        return out;
    }

    template <typename Lhs, typename Rhs, size_t... Extents, typename Operation>
        requires std::is_arithmetic_v<Lhs>
    constexpr auto elementwise_scalar_left(Lhs lhs, const Tensor<Rhs, Extents...>& rhs, Operation operation)
    {
        using Out = std::common_type_t<Lhs, Rhs>;
        Tensor<Out, Extents...> out{uninitialized};
        for (size_t i = 0; i < out.count; ++i)
            out[i] = operation(lhs, rhs[i]);
        return out;
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
    constexpr auto operator+(const Tensor<Lhs, Extents...>& lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_tensor(lhs, rhs, [](auto left, auto right) constexpr { return left + right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Rhs>
    constexpr auto operator+(const Tensor<Lhs, Extents...>& lhs, Rhs rhs)
    {
        return elementwise_scalar_right(lhs, rhs, [](auto left, auto right) constexpr { return left + right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Lhs>
    constexpr auto operator+(Lhs lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_scalar_left(lhs, rhs, [](auto left, auto right) constexpr { return left + right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
    constexpr auto operator-(const Tensor<Lhs, Extents...>& lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_tensor(lhs, rhs, [](auto left, auto right) constexpr { return left - right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Rhs>
    constexpr auto operator-(const Tensor<Lhs, Extents...>& lhs, Rhs rhs)
    {
        return elementwise_scalar_right(lhs, rhs, [](auto left, auto right) constexpr { return left - right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Lhs>
    constexpr auto operator-(Lhs lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_scalar_left(lhs, rhs, [](auto left, auto right) constexpr { return left - right; });
    }

    template <typename T, size_t... Extents>
    constexpr auto operator-(const Tensor<T, Extents...>& value)
    {
        Tensor<T, Extents...> out{uninitialized};
        for (size_t i = 0; i < out.count; ++i)
            out[i] = -value[i];
        return out;
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
    constexpr auto operator*(const Tensor<Lhs, Extents...>& lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_tensor(lhs, rhs, [](auto left, auto right) constexpr { return left * right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Rhs>
    constexpr auto operator*(const Tensor<Lhs, Extents...>& lhs, Rhs rhs)
    {
        return elementwise_scalar_right(lhs, rhs, [](auto left, auto right) constexpr { return left * right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Lhs>
    constexpr auto operator*(Lhs lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_scalar_left(lhs, rhs, [](auto left, auto right) constexpr { return left * right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
    constexpr auto operator/(const Tensor<Lhs, Extents...>& lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_tensor(lhs, rhs, [](auto left, auto right) constexpr { return left / right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Rhs>
    constexpr auto operator/(const Tensor<Lhs, Extents...>& lhs, Rhs rhs)
    {
        return elementwise_scalar_right(lhs, rhs, [](auto left, auto right) constexpr { return left / right; });
    }

    template <typename Lhs, typename Rhs, size_t... Extents>
        requires std::is_arithmetic_v<Lhs>
    constexpr auto operator/(Lhs lhs, const Tensor<Rhs, Extents...>& rhs)
    {
        return elementwise_scalar_left(lhs, rhs, [](auto left, auto right) constexpr { return left / right; });
    }
} // namespace tensor::detail

namespace tensor
{
    using detail::Matrix;
    using detail::Tensor;
    using detail::Vector;

    using detail::uninitialized;
} // namespace tensor
