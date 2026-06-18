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

    double norm2(std::span<const double, SmokeShape::x_size> values)
    {
        double total = 0.0;
        for (double value : values)
            total += value * value;
        return std::sqrt(total);
    }

    struct SolveContext
    {
        SmokeOperator op{};
        int evaluations = 0;

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
    };

    int pf_residual(void* data, int n, const double* x, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != static_cast<int>(SmokeShape::x_size))
            return 0;

        auto& context = *static_cast<SolveContext*>(data);
        ++context.evaluations;

        PackedVector residual{};
        const bool ok = context.op.evaluate(
            std::span<const double, SmokeShape::x_size>{x, SmokeShape::x_size},
            residual
        );
        if (!ok)
        {
            for (size_t i = 0; i < SmokeShape::x_size; ++i)
                fvec[i] = 1.0e20;
            return 0;
        }

        for (size_t i = 0; i < SmokeShape::x_size; ++i)
            fvec[i] = residual[i];
        return 0;
    }
} // namespace

int main()
{
    SolveContext context;

    double x[SmokeShape::x_size] = {0.01};
    double fvec[SmokeShape::x_size] = {0.0};
    PackedVector initial{};
    const bool initial_ok = context.op.evaluate(
        std::span<const double, SmokeShape::x_size>{x, SmokeShape::x_size},
        initial
    );
    const double initial_norm = initial_ok ? norm2(std::span<const double, SmokeShape::x_size>{
                                             initial.data(),
                                             SmokeShape::x_size,
                                         })
                                          : 1.0e20;

    constexpr int n = static_cast<int>(SmokeShape::x_size);
    constexpr int lwa = n * (3 * n + 13) / 2;
    double work[lwa]{};
    const int info = hybrd1(pf_residual, &context, n, x, fvec, 1.0e-10, work, lwa);

    PackedVector final{};
    const bool final_ok = context.op.evaluate(
        std::span<const double, SmokeShape::x_size>{x, SmokeShape::x_size},
        final
    );
    const double final_norm = final_ok ? norm2(std::span<const double, SmokeShape::x_size>{
                                         final.data(),
                                         SmokeShape::x_size,
                                     })
                                      : 1.0e20;

    nlohmann::json report = {
        {"route", "PF/psin/uniform"},
        {"x_size", SmokeShape::x_size},
        {"initial_ok", initial_ok},
        {"final_ok", final_ok},
        {"initial_norm", initial_norm},
        {"final_norm", final_norm},
        {"cminpack_info", info},
        {"evaluations", context.evaluations},
        {"solution", {x[0]}},
        {"residual", {final[0]}},
    };

    std::cout << report.dump(2) << '\n';
    return (info > 0 && final_ok && final_norm < 1.0e-8) ? EXIT_SUCCESS : EXIT_FAILURE;
}

