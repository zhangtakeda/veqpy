#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>

#include <cminpack.h>
#include <nlohmann/json.hpp>

#include "grid.h"
#include "math.h"
#include "pf_psin_uniform_operator.h"
#include "profiles.h"
#include "source.h"
#include "tensor.h"

namespace
{
    using grid::Grid;
    using grid::Legendre;
    using grid::Spectral;
    using operator_pf::PfPsinUniformOperator;
    using std::size_t;
    using tensor::Vector;

    constexpr auto no_c_slots = std::array<profiles::ProfileSlot, 0>{};
    constexpr auto no_s_slots = std::array<profiles::ProfileSlot, 0>{};

    using SmokeShape = profiles::ProfileShape<
        1,
        2,
        1,
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::fixed_slot(),
        profiles::optimized_slot(1),
        profiles::absent_slot(),
        no_c_slots,
        no_s_slots>;
    using SmokeGrid     = Grid<8, 8, 1, 1, 2, Legendre, Spectral>;
    using SmokeSource   = source::UniformSourceShape<5>;
    using SmokeOperator = PfPsinUniformOperator<SmokeShape, SmokeGrid, SmokeSource>;
    using PackedVector  = SmokeOperator::PackedVector;

    static_assert(SmokeShape::x_size == 1);

    constexpr double veqpy_max_residual               = 1.0e-6;
    constexpr int    veqpy_requested_max_evaluations  = 1000;
    constexpr int    veqpy_maxfev                     = veqpy_requested_max_evaluations > 500
                                                            ? veqpy_requested_max_evaluations
                                                            : 500;
    constexpr double veqpy_hybr_eps                   = 1.0e-6;
    constexpr double veqpy_hybr_factor                = 1.0;
    constexpr int    veqpy_hybr_mode                  = 1;
    constexpr int    veqpy_hybr_nprint                = 0;
    constexpr double veqpy_accepted_residual_factor   = 10.0;
    constexpr double veqpy_accepted_residual_floor    = 1.0e-5;
    constexpr double veqpy_x_scale_floor              = 1.0e-2;
    constexpr double veqpy_core_profile_prior         = 1.5e-1;
    constexpr double veqpy_psin_profile_scale_default = 1.0;

    double norm2(std::span<const double, SmokeShape::x_size> values) noexcept
    {
        double total = 0.0;
        for (double value : values)
            total += value * value;
        return std::sqrt(total);
    }

    template <typename Values>
    nlohmann::json json_array(const Values& values)
    {
        nlohmann::json out = nlohmann::json::array();
        for (double value : values)
            out.push_back(value);
        return out;
    }

    constexpr double veqpy_acceptance_threshold() noexcept
    {
        constexpr double scaled = veqpy_max_residual * veqpy_accepted_residual_factor;
        return scaled > veqpy_accepted_residual_floor ? scaled : veqpy_accepted_residual_floor;
    }

    std::array<double, SmokeShape::x_size> decode_z_to_x(
        const std::array<double, SmokeShape::x_size>& z,
        const std::array<double, SmokeShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, SmokeShape::x_size> x{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            x[i] = z[i] * x_scale[i];
        return x;
    }

    std::array<double, SmokeShape::x_size> encode_x_to_z(
        const std::array<double, SmokeShape::x_size>& x,
        const std::array<double, SmokeShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, SmokeShape::x_size> z{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            z[i] = x[i] / x_scale[i];
        return z;
    }

    struct SolveContext
    {
        SmokeOperator op{};
        std::array<double, SmokeShape::x_size> x_scale{};
        std::array<double, SmokeShape::x_size> residual_scale{};
        int                                    evaluations = 0;

        SolveContext()
        {
            op.params.a = 0.42;
            op.params.R0 = 1.8;
            op.params.Z0 = -0.25;
            op.params.B0 = 2.1;
            op.params.fix_rho = 0.0;
            op.params.profile_params.offsets[SmokeShape::kappa_profile_id] = 1.45;
            op.params.profile_params.offsets[SmokeShape::c_profile_id<0>()] = 0.0;

            constexpr std::array<double, SmokeSource::sample_count> heat{
                2.0,
                2.75,
                3.5,
                4.25,
                5.0,
            };
            constexpr std::array<double, SmokeSource::sample_count> current{
                0.5,
                0.625,
                0.75,
                0.875,
                1.0,
            };
            op.set_uniform_sources(
                std::span<const double, SmokeSource::sample_count>{heat.data(), heat.size()},
                std::span<const double, SmokeSource::sample_count>{current.data(), current.size()}
            );
        }

        bool raw_residual(
            std::span<const double, SmokeShape::x_size> x,
            std::span<double, SmokeShape::x_size>       residual
        ) noexcept
        {
            PackedVector raw{};
            const bool   ok = op.evaluate(x, raw);
            for (size_t i = 0; i < SmokeShape::x_size; ++i)
                residual[i] = ok ? raw[i] : 1.0e20;
            return ok;
        }

        void configure_veqpy_scales(const std::array<double, SmokeShape::x_size>& x0) noexcept
        {
            const double guess_rms = norm2(std::span<const double, SmokeShape::x_size>{
                x0.data(),
                SmokeShape::x_size,
            });
            double psin_scale = veqpy_psin_profile_scale_default;
            if (std::abs(psin_scale - 1.0) <= 1.0e-12)
                psin_scale = veqpy_core_profile_prior;
            x_scale[0] = std::max(
                {psin_scale, veqpy_core_profile_prior, guess_rms, veqpy_x_scale_floor}
            );

            PackedVector initial_raw{};
            const bool   ok = op.evaluate(
                std::span<const double, SmokeShape::x_size>{x0.data(), SmokeShape::x_size},
                initial_raw
            );
            const double initial_norm = ok ? norm2(std::span<const double, SmokeShape::x_size>{
                                               initial_raw.data(),
                                               SmokeShape::x_size,
                                           })
                                           : 1.0e20;
            const double initial_rms =
                initial_norm / std::sqrt(static_cast<double>(SmokeShape::x_size));
            const double block_scale = initial_rms > 1.0 ? initial_rms : 1.0;
            for (double& scale : residual_scale)
                scale = block_scale;
        }
    };

    int pf_residual_z(void* data, int n, const double* z, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != static_cast<int>(SmokeShape::x_size))
            return 0;

        auto& context = *static_cast<SolveContext*>(data);
        ++context.evaluations;

        std::array<double, SmokeShape::x_size> z_eval{};
        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            z_eval[i] = z[i];
        const auto x = decode_z_to_x(z_eval, context.x_scale);

        PackedVector raw{};
        const bool   ok = context.raw_residual(
            std::span<const double, SmokeShape::x_size>{x.data(), SmokeShape::x_size},
            std::span<double, SmokeShape::x_size>{raw.data(), SmokeShape::x_size}
        );
        if (!ok)
        {
            for (size_t i = 0; i < SmokeShape::x_size; ++i)
                fvec[i] = 1.0e20;
            return 0;
        }

        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            fvec[i] = raw[i] / context.residual_scale[i];
        return 0;
    }
} // namespace

int main()
{
    SolveContext context;

    std::array<double, SmokeShape::x_size> x_initial{};
    context.configure_veqpy_scales(x_initial);

    PackedVector initial{};
    const bool   initial_ok = context.raw_residual(
        std::span<const double, SmokeShape::x_size>{x_initial.data(), SmokeShape::x_size},
        std::span<double, SmokeShape::x_size>{initial.data(), SmokeShape::x_size}
    );
    const double initial_norm = initial_ok ? norm2(std::span<const double, SmokeShape::x_size>{
                                             initial.data(),
                                             SmokeShape::x_size,
                                         })
                                          : 1.0e20;
    PackedVector initial_scaled{};
    for (size_t i = 0; i < SmokeShape::x_size; ++i)
        initial_scaled[i] = initial[i] / context.residual_scale[i];
    const double initial_scaled_norm = norm2(std::span<const double, SmokeShape::x_size>{
        initial_scaled.data(),
        SmokeShape::x_size,
    });

    auto z = encode_x_to_z(x_initial, context.x_scale);
    PackedVector fvec{};

    constexpr int n = static_cast<int>(SmokeShape::x_size);
    constexpr int ml = n - 1;
    constexpr int mu = n - 1;
    constexpr int lr = n * (n + 1) / 2;
    std::array<double, SmokeShape::x_size> diag{};
    diag.fill(1.0);
    std::array<double, SmokeShape::x_size * SmokeShape::x_size> fjac{};
    std::array<double, static_cast<size_t>(lr)>                 r{};
    std::array<double, SmokeShape::x_size>                      qtf{};
    std::array<double, SmokeShape::x_size>                      wa1{};
    std::array<double, SmokeShape::x_size>                      wa2{};
    std::array<double, SmokeShape::x_size>                      wa3{};
    std::array<double, SmokeShape::x_size>                      wa4{};
    int                                                         nfev = 0;
    const int info = hybrd(
        pf_residual_z,
        &context,
        n,
        z.data(),
        fvec.data(),
        veqpy_max_residual,
        veqpy_maxfev,
        ml,
        mu,
        veqpy_hybr_eps,
        diag.data(),
        veqpy_hybr_mode,
        veqpy_hybr_factor,
        veqpy_hybr_nprint,
        &nfev,
        fjac.data(),
        n,
        r.data(),
        lr,
        qtf.data(),
        wa1.data(),
        wa2.data(),
        wa3.data(),
        wa4.data()
    );

    const auto x_final = decode_z_to_x(z, context.x_scale);
    PackedVector final{};
    const bool   final_ok = context.raw_residual(
        std::span<const double, SmokeShape::x_size>{x_final.data(), SmokeShape::x_size},
        std::span<double, SmokeShape::x_size>{final.data(), SmokeShape::x_size}
    );
    const double final_norm = final_ok ? norm2(std::span<const double, SmokeShape::x_size>{
                                         final.data(),
                                         SmokeShape::x_size,
                                     })
                                      : 1.0e20;
    PackedVector final_scaled{};
    for (size_t i = 0; i < SmokeShape::x_size; ++i)
        final_scaled[i] = final[i] / context.residual_scale[i];
    const double final_scaled_norm = norm2(std::span<const double, SmokeShape::x_size>{
        final_scaled.data(),
        SmokeShape::x_size,
    });
    const bool accepted_by_veqpy = final_ok && final_norm <= veqpy_acceptance_threshold();

    nlohmann::json report = {
        {"route", "PF/psin/uniform"},
        {"x_size", SmokeShape::x_size},
        {"solver",
         {
             {"method", "hybr"},
             {"entrypoint", "cminpack::hybrd"},
             {"initial_policy", "auto/zero"},
             {"residual_normalization", "fast"},
             {"max_residual", veqpy_max_residual},
             {"acceptance_threshold", veqpy_acceptance_threshold()},
             {"requested_max_evaluations", veqpy_requested_max_evaluations},
             {"maxfev", veqpy_maxfev},
             {"eps", veqpy_hybr_eps},
             {"factor", veqpy_hybr_factor},
             {"diag_mode", veqpy_hybr_mode},
             {"ml", ml},
             {"mu", mu},
         }},
        {"normalization",
         {
             {"x_scale", json_array(context.x_scale)},
             {"residual_scale", json_array(context.residual_scale)},
             {"unknown_space", "z = x / x_scale"},
         }},
        {"initial",
         {
             {"ok", initial_ok},
             {"x", json_array(x_initial)},
             {"z", json_array(encode_x_to_z(x_initial, context.x_scale))},
             {"raw_residual", json_array(initial)},
             {"scaled_residual", json_array(initial_scaled)},
             {"raw_norm", initial_norm},
             {"scaled_norm", initial_scaled_norm},
         }},
        {"final",
         {
             {"ok", final_ok},
             {"x", json_array(x_final)},
             {"z", json_array(z)},
             {"raw_residual", json_array(final)},
             {"scaled_residual", json_array(final_scaled)},
             {"raw_norm", final_norm},
             {"scaled_norm", final_scaled_norm},
             {"accepted_by_veqpy", accepted_by_veqpy},
         }},
        {"cminpack",
         {
             {"info", info},
             {"success", info == 1},
             {"nfev", nfev},
             {"callback_evaluations", context.evaluations},
         }},
    };

    std::cout << report.dump(2) << '\n';
    return (accepted_by_veqpy && info > 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
