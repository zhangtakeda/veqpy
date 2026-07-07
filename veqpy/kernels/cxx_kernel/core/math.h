#pragma once

// Compile-time and runtime math helpers for generated Cxx Kernel artifacts.

#include "tensor.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <gcem.hpp>
#include <type_traits>

namespace math::detail
{
    using std::size_t;

    inline constexpr double pi      = 3.141592653589793238462643383279502884;
    inline constexpr double half_pi = 0.5 * pi;

    constexpr double min(double lhs, double rhs) noexcept { return rhs < lhs ? rhs : lhs; }

    constexpr double max(double lhs, double rhs) noexcept { return lhs < rhs ? rhs : lhs; }

    constexpr double clamp(double value, double lower, double upper) noexcept { return min(max(value, lower), upper); }

    constexpr double sign(double value) noexcept
    {
        if (value > 0.0)
            return 1.0;
        if (value < 0.0)
            return -1.0;
        return 0.0;
    }

    constexpr double abs(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::abs(value);
        else
            return std::abs(value);
    }

    constexpr double hypot(double lhs, double rhs)
    {
        if (std::is_constant_evaluated())
            return gcem::hypot(lhs, rhs);
        else
            return std::hypot(lhs, rhs);
    }

    inline constexpr std::uint64_t double_exponent_mask = 0x7ff0'0000'0000'0000ULL;

    constexpr bool is_finite(double value) noexcept
    {
        const auto bits = std::bit_cast<std::uint64_t>(value);
        return (bits & double_exponent_mask) != double_exponent_mask;
    }

    constexpr double sqrt(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::sqrt(value);
        else
            return std::sqrt(value);
    }

    constexpr double pow(double base, double exponent)
    {
        if (std::is_constant_evaluated())
            return gcem::pow(base, exponent);
        else
            return std::pow(base, exponent);
    }

    constexpr double exp(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::exp(value);
        else
            return std::exp(value);
    }

    constexpr double log(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::log(value);
        else
            return std::log(value);
    }

    constexpr double sin(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::sin(value);
        else
            return std::sin(value);
    }

    constexpr double cos(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::cos(value);
        else
            return std::cos(value);
    }

    template <int Order>
    struct InverseFactorial
    {
        static_assert(Order >= 0, "factorial order must be non-negative");
        static constexpr double value = InverseFactorial<Order - 1>::value / static_cast<double>(Order);
    };

    template <>
    struct InverseFactorial<0>
    {
        static constexpr double value = 1.0;
    };

    template <int Power>
    inline constexpr double taylor_coefficient_v =
        (((Power / 2) % 2 == 0) ? 1.0 : -1.0) * InverseFactorial<Power>::value;

    template <int CurrentPower, int LowestPower>
    struct TaylorHornerStep
    {
        static inline double eval(double r2, double accumulator) noexcept
        {
            if constexpr (CurrentPower < LowestPower)
            {
                return accumulator;
            }
            else
            {
                return TaylorHornerStep<CurrentPower - 2, LowestPower>::eval(
                    r2, accumulator * r2 + taylor_coefficient_v<CurrentPower>);
            }
        }
    };

    template <int HighestPower, int LowestPower>
    inline double evaluate_taylor_horner(double r2) noexcept
    {
        static_assert(HighestPower >= LowestPower, "Taylor highest power must cover the constant/linear term");
        static_assert(((HighestPower - LowestPower) % 2) == 0, "Taylor powers must have one parity");
        return TaylorHornerStep<HighestPower - 2, LowestPower>::eval(r2, taylor_coefficient_v<HighestPower>);
    }

    template <int SinOrder = 11, int CosOrder = (SinOrder == 0 ? 0 : SinOrder - 1)>
    inline void relaxed_sincos(double value, double& sin_value, double& cos_value) noexcept
    {
        static_assert(SinOrder >= 0 && CosOrder >= 0, "relaxed_sincos orders must be non-negative");

        if constexpr (SinOrder == 0 && CosOrder == 0)
        {
            sin_value = std::sin(value);
            cos_value = std::cos(value);
        }
        else
        {
            static_assert((SinOrder % 2) == 1, "relaxed_sincos sine order must be odd");
            static_assert((CosOrder % 2) == 0, "relaxed_sincos cosine order must be even");

            constexpr double inv_half_pi    = 2.0 / pi;
            const double     scaled         = value * inv_half_pi;
            const int        quadrant_index = static_cast<int>(scaled + (scaled >= 0.0 ? 0.5 : -0.5));
            const double     q              = static_cast<double>(quadrant_index);
            const double     reduced        = value - q * half_pi;
            const double     r2             = reduced * reduced;

            const double sin_reduced = reduced * evaluate_taylor_horner<SinOrder, 1>(r2);
            const double cos_reduced = evaluate_taylor_horner<CosOrder, 0>(r2);

            const unsigned quadrant = static_cast<unsigned>(quadrant_index) & 3U;
            const bool     odd      = (quadrant & 1U) != 0U;
            const double   sin_base = odd ? cos_reduced : sin_reduced;
            const double   cos_base = odd ? sin_reduced : cos_reduced;

            sin_value = quadrant < 2U ? sin_base : -sin_base;
            cos_value = quadrant == 0U || quadrant == 3U ? cos_base : -cos_base;
        }
    }

    constexpr double tan(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::tan(value);
        else
            return std::tan(value);
    }

    constexpr double arcsin(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::asin(value);
        else
            return std::asin(value);
    }

    constexpr double arccos(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::acos(value);
        else
            return std::acos(value);
    }

    constexpr double arctan(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::atan(value);
        else
            return std::atan(value);
    }

    constexpr double sinh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::sinh(value);
        else
            return std::sinh(value);
    }

    constexpr double cosh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::cosh(value);
        else
            return std::cosh(value);
    }

    constexpr double tanh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::tanh(value);
        else
            return std::tanh(value);
    }

    constexpr double arcsinh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::asinh(value);
        else
            return std::asinh(value);
    }

    constexpr double arccosh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::acosh(value);
        else
            return std::acosh(value);
    }

    constexpr double arctanh(double value)
    {
        if (std::is_constant_evaluated())
            return gcem::atanh(value);
        else
            return std::atanh(value);
    }

    template <size_t... Extents>
    constexpr void min_into(tensor::Tensor<double, Extents...>&       out,
                            const tensor::Tensor<double, Extents...>& lhs,
                            const tensor::Tensor<double, Extents...>& rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = min(lhs[i], rhs[i]);
    }

    template <size_t... Extents>
    constexpr void min_inplace(tensor::Tensor<double, Extents...>& lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        min_into(lhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> min(const tensor::Tensor<double, Extents...>& lhs,
                                                     const tensor::Tensor<double, Extents...>& rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        min_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void
    min_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = min(lhs[i], rhs);
    }

    template <size_t... Extents>
    constexpr void min_inplace(tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        min_into(lhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> min(const tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        min_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void
    min_into(tensor::Tensor<double, Extents...>& out, double lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = min(lhs, rhs[i]);
    }

    template <size_t... Extents>
    constexpr void min_inplace(double lhs, tensor::Tensor<double, Extents...>& rhs)
    {
        min_into(rhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> min(double lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        min_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void max_into(tensor::Tensor<double, Extents...>&       out,
                            const tensor::Tensor<double, Extents...>& lhs,
                            const tensor::Tensor<double, Extents...>& rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = max(lhs[i], rhs[i]);
    }

    template <size_t... Extents>
    constexpr void max_inplace(tensor::Tensor<double, Extents...>& lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        max_into(lhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> max(const tensor::Tensor<double, Extents...>& lhs,
                                                     const tensor::Tensor<double, Extents...>& rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        max_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void
    max_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = max(lhs[i], rhs);
    }

    template <size_t... Extents>
    constexpr void max_inplace(tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        max_into(lhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> max(const tensor::Tensor<double, Extents...>& lhs, double rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        max_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void
    max_into(tensor::Tensor<double, Extents...>& out, double lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = max(lhs, rhs[i]);
    }

    template <size_t... Extents>
    constexpr void max_inplace(double lhs, tensor::Tensor<double, Extents...>& rhs)
    {
        max_into(rhs, lhs, rhs);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> max(double lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        max_into(out, lhs, rhs);
        return out;
    }

    template <size_t... Extents>
    constexpr void clamp_into(tensor::Tensor<double, Extents...>&       out,
                              const tensor::Tensor<double, Extents...>& values,
                              double                                    lower,
                              double                                    upper)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = clamp(values[i], lower, upper);
    }

    template <size_t... Extents>
    constexpr void clamp_inplace(tensor::Tensor<double, Extents...>& values, double lower, double upper)
    {
        clamp_into(values, values, lower, upper);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...>
    clamp(const tensor::Tensor<double, Extents...>& values, double lower, double upper)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        clamp_into(out, values, lower, upper);
        return out;
    }

    template <size_t... Extents>
    constexpr void sign_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = sign(values[i]);
    }

    template <size_t... Extents>
    constexpr void sign_inplace(tensor::Tensor<double, Extents...>& values)
    {
        sign_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> sign(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        sign_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void abs_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = abs(values[i]);
    }

    template <size_t... Extents>
    constexpr void abs_inplace(tensor::Tensor<double, Extents...>& values)
    {
        abs_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> abs(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        abs_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr bool is_finite(const tensor::Tensor<double, Extents...>& values) noexcept
    {
        for (size_t i = 0; i < values.count; ++i)
            if (!is_finite(values[i]))
                return false;
        return true;
    }

    template <size_t... Extents>
    constexpr void sqrt_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = sqrt(values[i]);
    }

    template <size_t... Extents>
    constexpr void sqrt_inplace(tensor::Tensor<double, Extents...>& values)
    {
        sqrt_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> sqrt(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        sqrt_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void
    pow_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values, double exponent)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = pow(values[i], exponent);
    }

    template <size_t... Extents>
    constexpr void pow_inplace(tensor::Tensor<double, Extents...>& values, double exponent)
    {
        pow_into(values, values, exponent);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> pow(const tensor::Tensor<double, Extents...>& values, double exponent)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        pow_into(out, values, exponent);
        return out;
    }

    template <size_t... Extents>
    constexpr void exp_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = exp(values[i]);
    }

    template <size_t... Extents>
    constexpr void exp_inplace(tensor::Tensor<double, Extents...>& values)
    {
        exp_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> exp(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        exp_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void log_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = log(values[i]);
    }

    template <size_t... Extents>
    constexpr void log_inplace(tensor::Tensor<double, Extents...>& values)
    {
        log_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> log(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        log_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void sin_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = sin(values[i]);
    }

    template <size_t... Extents>
    constexpr void sin_inplace(tensor::Tensor<double, Extents...>& values)
    {
        sin_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> sin(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        sin_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void cos_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = cos(values[i]);
    }

    template <size_t... Extents>
    constexpr void cos_inplace(tensor::Tensor<double, Extents...>& values)
    {
        cos_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> cos(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        cos_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void tan_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = tan(values[i]);
    }

    template <size_t... Extents>
    constexpr void tan_inplace(tensor::Tensor<double, Extents...>& values)
    {
        tan_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> tan(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        tan_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arcsin_into(tensor::Tensor<double, Extents...>&       out,
                               const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arcsin(values[i]);
    }

    template <size_t... Extents>
    constexpr void arcsin_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arcsin_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arcsin(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arcsin_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arccos_into(tensor::Tensor<double, Extents...>&       out,
                               const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arccos(values[i]);
    }

    template <size_t... Extents>
    constexpr void arccos_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arccos_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arccos(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arccos_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arctan_into(tensor::Tensor<double, Extents...>&       out,
                               const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arctan(values[i]);
    }

    template <size_t... Extents>
    constexpr void arctan_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arctan_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arctan(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arctan_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void sinh_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = sinh(values[i]);
    }

    template <size_t... Extents>
    constexpr void sinh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        sinh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> sinh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        sinh_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void cosh_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = cosh(values[i]);
    }

    template <size_t... Extents>
    constexpr void cosh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        cosh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> cosh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        cosh_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void tanh_into(tensor::Tensor<double, Extents...>& out, const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = tanh(values[i]);
    }

    template <size_t... Extents>
    constexpr void tanh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        tanh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> tanh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        tanh_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arcsinh_into(tensor::Tensor<double, Extents...>&       out,
                                const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arcsinh(values[i]);
    }

    template <size_t... Extents>
    constexpr void arcsinh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arcsinh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arcsinh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arcsinh_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arccosh_into(tensor::Tensor<double, Extents...>&       out,
                                const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arccosh(values[i]);
    }

    template <size_t... Extents>
    constexpr void arccosh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arccosh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arccosh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arccosh_into(out, values);
        return out;
    }

    template <size_t... Extents>
    constexpr void arctanh_into(tensor::Tensor<double, Extents...>&       out,
                                const tensor::Tensor<double, Extents...>& values)
    {
        for (size_t i = 0; i < out.count; ++i)
            out[i] = arctanh(values[i]);
    }

    template <size_t... Extents>
    constexpr void arctanh_inplace(tensor::Tensor<double, Extents...>& values)
    {
        arctanh_into(values, values);
    }

    template <size_t... Extents>
    constexpr tensor::Tensor<double, Extents...> arctanh(const tensor::Tensor<double, Extents...>& values)
    {
        tensor::Tensor<double, Extents...> out{tensor::uninitialized};
        arctanh_into(out, values);
        return out;
    }

    // reduce

    template <size_t... Extents>
    constexpr double min(const tensor::Tensor<double, Extents...>& values) noexcept
    {
        double result = values[0];
        for (size_t i = 1; i < values.count; ++i)
            if (values[i] < result)
                result = values[i];
        return result;
    }

    template <size_t... Extents>
    constexpr double max(const tensor::Tensor<double, Extents...>& values) noexcept
    {
        double result = values[0];
        for (size_t i = 1; i < values.count; ++i)
            if (result < values[i])
                result = values[i];
        return result;
    }

    template <size_t... Extents>
    constexpr double max_abs(const tensor::Tensor<double, Extents...>& values) noexcept
    {
        double result = 0.0;
        for (size_t i = 0; i < values.count; ++i)
        {
            const double magnitude = abs(values[i]);
            if (magnitude > result)
                result = magnitude;
        }
        return result;
    }

    template <size_t... Extents>
    constexpr double sum(const tensor::Tensor<double, Extents...>& values) noexcept
    {
        double total = 0.0;
        for (size_t i = 0; i < values.count; ++i)
            total += values[i];
        return total;
    }

    template <size_t... Extents>
    constexpr double dot(const tensor::Tensor<double, Extents...>& lhs, const tensor::Tensor<double, Extents...>& rhs)
    {
        double total = 0.0;
        for (size_t i = 0; i < lhs.count; ++i)
            total += lhs[i] * rhs[i];
        return total;
    }

    template <size_t... Extents>
    constexpr double norm2(const tensor::Tensor<double, Extents...>& values)
    {
        return sqrt(dot(values, values));
    }

} // namespace math::detail

namespace math
{
    using detail::min;
    using detail::max;
    using detail::clamp;
    using detail::sign;
    using detail::abs;
    using detail::hypot;
    using detail::is_finite;
    using detail::sqrt;
    using detail::pow;
    using detail::exp;
    using detail::log;
    using detail::sin;
    using detail::cos;
    using detail::relaxed_sincos;
    using detail::tan;
    using detail::arcsin;
    using detail::arccos;
    using detail::arctan;
    using detail::sinh;
    using detail::cosh;
    using detail::tanh;
    using detail::arcsinh;
    using detail::arccosh;
    using detail::arctanh;

    using detail::min_into;
    using detail::min_inplace;
    using detail::max_into;
    using detail::max_inplace;
    using detail::clamp_into;
    using detail::clamp_inplace;
    using detail::sign_into;
    using detail::sign_inplace;
    using detail::abs_into;
    using detail::abs_inplace;
    using detail::sqrt_into;
    using detail::sqrt_inplace;
    using detail::pow_into;
    using detail::pow_inplace;
    using detail::exp_into;
    using detail::exp_inplace;
    using detail::log_into;
    using detail::log_inplace;
    using detail::sin_into;
    using detail::sin_inplace;
    using detail::cos_into;
    using detail::cos_inplace;
    using detail::tan_into;
    using detail::tan_inplace;
    using detail::arcsin_into;
    using detail::arcsin_inplace;
    using detail::arccos_into;
    using detail::arccos_inplace;
    using detail::arctan_into;
    using detail::arctan_inplace;
    using detail::sinh_into;
    using detail::sinh_inplace;
    using detail::cosh_into;
    using detail::cosh_inplace;
    using detail::tanh_into;
    using detail::tanh_inplace;
    using detail::arcsinh_into;
    using detail::arcsinh_inplace;
    using detail::arccosh_into;
    using detail::arccosh_inplace;
    using detail::arctanh_into;
    using detail::arctanh_inplace;

    using detail::max_abs;
    using detail::sum;
    using detail::dot;
    using detail::norm2;
} // namespace math
