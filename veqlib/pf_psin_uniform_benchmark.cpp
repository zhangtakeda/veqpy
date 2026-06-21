#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

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

    constexpr auto bench_c_slots = std::array<profiles::ProfileSlot, 1>{
        profiles::absent_slot(),
    };
    constexpr auto bench_s_slots = std::array<profiles::ProfileSlot, 1>{
        profiles::optimized_slot(3),
    };

    using BenchShape = profiles::ProfileShape<
        20,
        2,
        1,
        profiles::optimized_slot(3),
        profiles::absent_slot(),
        profiles::optimized_slot(6),
        profiles::fixed_slot(),
        profiles::optimized_slot(6),
        profiles::absent_slot(),
        bench_c_slots,
        bench_s_slots>;
    using BenchGrid     = Grid<32, 16, 20, 1, 2, Legendre, Spectral>;
    using BenchSource   = source::UniformSourceShape<51>;
    using BenchOperator = PfPsinUniformOperator<BenchShape, BenchGrid, BenchSource>;
    using PackedVector  = BenchOperator::PackedVector;

    static_assert(BenchShape::x_size == 18);

    constexpr double veqpy_max_residual              = 1.0e-6;
    constexpr int    veqpy_requested_max_evaluations = 1000;
    constexpr int    veqpy_maxfev =
        veqpy_requested_max_evaluations > 500 ? veqpy_requested_max_evaluations : 500;
    constexpr double veqpy_hybr_eps                 = 1.0e-6;
    constexpr double veqpy_hybr_factor              = 1.0;
    constexpr int    veqpy_hybr_mode                = 1;
    constexpr int    veqpy_hybr_nprint              = 0;
    constexpr double veqpy_accepted_residual_factor = 10.0;
    constexpr double veqpy_accepted_residual_floor  = 1.0e-5;

    struct CaseInput
    {
        std::string case_name;
        std::array<double, BenchSource::sample_count> heat{};
        std::array<double, BenchSource::sample_count> current{};
        std::array<double, BenchShape::x_size>        x0{};
        std::array<double, BenchShape::x_size>        x_scale{};
        std::array<double, BenchShape::x_size>        residual_scale{};
        double a         = 1.0;
        double R0        = 1.0;
        double Z0        = 0.0;
        double B0        = 1.0;
        double ka        = 1.0;
        double c0_offset = 0.0;
        double s1_offset = 0.0;
        double Ip        = source::unset_constraint();
        double beta      = source::unset_constraint();
        double fix_rho   = 0.05;
        int    repeat    = 10;
        int    warmup    = 1;
    };

    struct SolveResult
    {
        std::array<double, BenchShape::x_size> x{};
        PackedVector                          raw{};
        PackedVector                          scaled{};
        std::array<double, 2>                 alpha{};
        double                                raw_norm    = 0.0;
        double                                scaled_norm = 0.0;
        int                                   info        = 0;
        int                                   nfev        = 0;
        int                                   callbacks   = 0;
        bool                                  ok          = false;
        bool                                  accepted    = false;
    };

    double norm2(std::span<const double, BenchShape::x_size> values) noexcept
    {
        double total = 0.0;
        for (double value : values)
            total += value * value;
        return std::sqrt(total);
    }

    constexpr double veqpy_acceptance_threshold() noexcept
    {
        constexpr double scaled = veqpy_max_residual * veqpy_accepted_residual_factor;
        return scaled > veqpy_accepted_residual_floor ? scaled : veqpy_accepted_residual_floor;
    }

    template <size_t Count>
    std::array<double, Count> fixed_array(const nlohmann::json& data, const char* name)
    {
        const auto& values = data.at(name);
        if (!values.is_array() || values.size() != Count)
            throw std::runtime_error(std::string{name} + " has the wrong length");
        std::array<double, Count> out{};
        for (size_t i = 0; i < Count; ++i)
            out[i] = values.at(i).get<double>();
        return out;
    }

    double nullable_double(const nlohmann::json& data, const char* name)
    {
        const auto& value = data.at(name);
        if (value.is_null())
            return source::unset_constraint();
        return value.get<double>();
    }

    double optional_double(const nlohmann::json& data, const char* name, double fallback)
    {
        const auto iter = data.find(name);
        if (iter == data.end() || iter->is_null())
            return fallback;
        return iter->get<double>();
    }

    int optional_int(const nlohmann::json& data, const char* name, int fallback)
    {
        const auto iter = data.find(name);
        if (iter == data.end() || iter->is_null())
            return fallback;
        return iter->get<int>();
    }

    CaseInput parse_case_input(const nlohmann::json& data)
    {
        CaseInput input{};
        input.case_name      = data.value("case_name", "PF_psin_uniform");
        input.heat           = fixed_array<BenchSource::sample_count>(data, "scaled_heat");
        input.current        = fixed_array<BenchSource::sample_count>(data, "scaled_current");
        input.x0             = fixed_array<BenchShape::x_size>(data, "x0");
        input.x_scale        = fixed_array<BenchShape::x_size>(data, "x_scale");
        input.residual_scale = fixed_array<BenchShape::x_size>(data, "residual_scale");

        const auto& boundary = data.at("boundary");
        input.a              = boundary.at("a").get<double>();
        input.R0             = boundary.at("R0").get<double>();
        input.Z0             = boundary.at("Z0").get<double>();
        input.B0             = boundary.at("B0").get<double>();
        input.ka             = boundary.at("ka").get<double>();
        input.c0_offset      = optional_double(boundary, "c0_offset", 0.0);
        input.s1_offset      = optional_double(boundary, "s1_offset", 0.0);

        input.Ip      = nullable_double(data, "scaled_Ip");
        input.beta    = nullable_double(data, "beta");
        input.fix_rho = optional_double(data, "fix_rho", 0.05);
        input.repeat  = optional_int(data, "repeat", 10);
        input.warmup  = optional_int(data, "warmup", 1);
        if (input.repeat < 0 || input.warmup < 0)
            throw std::runtime_error("repeat and warmup must be non-negative");
        return input;
    }

    template <typename Values>
    nlohmann::json json_array(const Values& values)
    {
        nlohmann::json out = nlohmann::json::array();
        for (double value : values)
            out.push_back(value);
        return out;
    }

    nlohmann::json json_constraint_or_null(double value)
    {
        if (source::constraint_is_set(value))
            return value;
        return nullptr;
    }

    std::array<double, BenchShape::x_size> decode_z_to_x(
        const std::array<double, BenchShape::x_size>& z,
        const std::array<double, BenchShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, BenchShape::x_size> x{};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            x[i] = z[i] * x_scale[i];
        return x;
    }

    std::array<double, BenchShape::x_size> encode_x_to_z(
        const std::array<double, BenchShape::x_size>& x,
        const std::array<double, BenchShape::x_size>& x_scale
    ) noexcept
    {
        std::array<double, BenchShape::x_size> z{};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            z[i] = x[i] / x_scale[i];
        return z;
    }

    struct SolveContext
    {
        BenchOperator op{};
        CaseInput     input{};
        int           evaluations = 0;

        explicit SolveContext(const CaseInput& case_input) : input(case_input)
        {
            op.params.a       = input.a;
            op.params.R0      = input.R0;
            op.params.Z0      = input.Z0;
            op.params.B0      = input.B0;
            op.params.Ip      = input.Ip;
            op.params.beta    = input.beta;
            op.params.fix_rho = input.fix_rho;
            op.params.profile_params.offsets[BenchShape::kappa_profile_id] = input.ka;
            op.params.profile_params.offsets[BenchShape::c_profile_id<0>()] = input.c0_offset;
            op.params.profile_params.offsets[BenchShape::s_profile_id<1>()] = input.s1_offset;
            op.set_uniform_sources(
                std::span<const double, BenchSource::sample_count>{input.heat.data(), input.heat.size()},
                std::span<const double, BenchSource::sample_count>{input.current.data(), input.current.size()}
            );
        }

        bool raw_residual(
            std::span<const double, BenchShape::x_size> x,
            std::span<double, BenchShape::x_size>       residual
        ) noexcept
        {
            PackedVector raw{};
            const bool   ok = op.evaluate(x, raw);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                residual[i] = ok ? raw[i] : 1.0e20;
            return ok;
        }
    };

    int pf_residual_z(void* data, int n, const double* z, double* fvec, int iflag)
    {
        if (iflag <= 0 || n != static_cast<int>(BenchShape::x_size))
            return 0;

        auto& context = *static_cast<SolveContext*>(data);
        ++context.evaluations;

        std::array<double, BenchShape::x_size> z_eval{};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            z_eval[i] = z[i];
        const auto x = decode_z_to_x(z_eval, context.input.x_scale);

        PackedVector raw{};
        const bool   ok = context.raw_residual(
            std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{raw.data(), BenchShape::x_size}
        );
        if (!ok)
        {
            for (size_t i = 0; i < BenchShape::x_size; ++i)
                fvec[i] = 1.0e20;
            return 0;
        }

        for (size_t i = 0; i < BenchShape::x_size; ++i)
            fvec[i] = raw[i] / context.input.residual_scale[i];
        return 0;
    }

    SolveResult run_hybr_once(SolveContext& context)
    {
        context.evaluations = 0;
        auto z = encode_x_to_z(context.input.x0, context.input.x_scale);
        PackedVector fvec{};

        constexpr int n  = static_cast<int>(BenchShape::x_size);
        constexpr int ml = n - 1;
        constexpr int mu = n - 1;
        constexpr int lr = n * (n + 1) / 2;
        std::array<double, BenchShape::x_size> diag{};
        diag.fill(1.0);
        std::array<double, BenchShape::x_size * BenchShape::x_size> fjac{};
        std::array<double, static_cast<size_t>(lr)>                 r{};
        std::array<double, BenchShape::x_size>                      qtf{};
        std::array<double, BenchShape::x_size>                      wa1{};
        std::array<double, BenchShape::x_size>                      wa2{};
        std::array<double, BenchShape::x_size>                      wa3{};
        std::array<double, BenchShape::x_size>                      wa4{};
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

        SolveResult result{};
        result.info      = info;
        result.nfev      = nfev;
        result.callbacks = context.evaluations;
        result.x         = decode_z_to_x(z, context.input.x_scale);
        result.ok        = context.raw_residual(
            std::span<const double, BenchShape::x_size>{result.x.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{result.raw.data(), BenchShape::x_size}
        );
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            result.scaled[i] = result.raw[i] / context.input.residual_scale[i];
        result.raw_norm = result.ok
                              ? norm2(std::span<const double, BenchShape::x_size>{
                                    result.raw.data(),
                                    BenchShape::x_size,
                                })
                              : 1.0e20;
        result.scaled_norm = norm2(std::span<const double, BenchShape::x_size>{
            result.scaled.data(),
            BenchShape::x_size,
        });
        result.alpha    = {context.op.source_runtime.alpha1, context.op.source_runtime.alpha2};
        result.accepted = result.ok && result.raw_norm <= veqpy_acceptance_threshold();
        return result;
    }

    nlohmann::json solve_result_json(const SolveResult& result)
    {
        return {
            {"ok", result.ok},
            {"accepted_by_veqpy", result.accepted},
            {"x", json_array(result.x)},
            {"raw_residual", json_array(result.raw)},
            {"scaled_residual", json_array(result.scaled)},
            {"alpha", json_array(result.alpha)},
            {"raw_norm", result.raw_norm},
            {"scaled_norm", result.scaled_norm},
            {"info", result.info},
            {"nfev", result.nfev},
            {"callback_evaluations", result.callbacks},
        };
    }

    double mean(const std::vector<double>& values)
    {
        if (values.empty())
            return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    }

    double stddev(const std::vector<double>& values, double avg)
    {
        if (values.empty())
            return 0.0;
        double total = 0.0;
        for (double value : values)
        {
            const double delta = value - avg;
            total += delta * delta;
        }
        return std::sqrt(total / static_cast<double>(values.size()));
    }

    double percentile_sorted(const std::vector<double>& sorted, double percentile)
    {
        if (sorted.empty())
            return 0.0;
        const double pos   = (static_cast<double>(sorted.size()) - 1.0) * percentile;
        const auto   lower = static_cast<size_t>(std::floor(pos));
        const auto   upper = static_cast<size_t>(std::ceil(pos));
        if (lower == upper)
            return sorted[lower];
        const double fraction = pos - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    }

    nlohmann::json timing_json(const std::vector<double>& samples)
    {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double avg = mean(samples);
        return {
            {"repeat_count", samples.size()},
            {"samples_ms", json_array(samples)},
            {"avg_ms", avg},
            {"median_ms", percentile_sorted(sorted, 0.50)},
            {"p95_ms", percentile_sorted(sorted, 0.95)},
            {"min_ms", sorted.empty() ? 0.0 : sorted.front()},
            {"max_ms", sorted.empty() ? 0.0 : sorted.back()},
            {"std_ms", stddev(samples, avg)},
        };
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            std::cerr << "usage: " << argv[0] << " case.json\n";
            return EXIT_FAILURE;
        }

        std::ifstream stream(argv[1]);
        if (!stream)
            throw std::runtime_error(std::string{"failed to open "} + argv[1]);
        const CaseInput input = parse_case_input(nlohmann::json::parse(stream));
        SolveContext    context{input};

        PackedVector initial_raw{};
        const bool   initial_ok = context.raw_residual(
            std::span<const double, BenchShape::x_size>{input.x0.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{initial_raw.data(), BenchShape::x_size}
        );
        PackedVector initial_scaled{};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            initial_scaled[i] = initial_raw[i] / input.residual_scale[i];

        for (int i = 0; i < input.warmup; ++i)
            (void)run_hybr_once(context);

        std::vector<double> samples_ms;
        samples_ms.reserve(static_cast<size_t>(input.repeat));
        SolveResult final{};
        for (int i = 0; i < input.repeat; ++i)
        {
            const auto started = std::chrono::steady_clock::now();
            final              = run_hybr_once(context);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            samples_ms.push_back(
                std::chrono::duration<double, std::milli>(elapsed).count()
            );
        }
        if (input.repeat == 0)
            final = run_hybr_once(context);

        const nlohmann::json report = {
            {"case_name", input.case_name},
            {"route", "PF/psin/uniform"},
            {"x_size", BenchShape::x_size},
            {"grid",
             {
                 {"Nr", BenchGrid::radial_nodes},
                 {"Nt", BenchGrid::theta_rows},
                 {"L_max", BenchShape::L_max},
                 {"M_max", BenchShape::M_max},
                 {"K_max", BenchShape::K_max},
                 {"quadrature_scheme", "legendre"},
                 {"calculus_scheme", "spectral"},
             }},
            {"solver",
             {
                 {"method", "hybr"},
                 {"entrypoint", "cminpack::hybrd"},
                 {"max_residual", veqpy_max_residual},
                 {"acceptance_threshold", veqpy_acceptance_threshold()},
                 {"requested_max_evaluations", veqpy_requested_max_evaluations},
                 {"maxfev", veqpy_maxfev},
                 {"eps", veqpy_hybr_eps},
                 {"factor", veqpy_hybr_factor},
                 {"diag_mode", veqpy_hybr_mode},
             }},
            {"normalization",
             {
                 {"x_scale", json_array(input.x_scale)},
                 {"residual_scale", json_array(input.residual_scale)},
                 {"unknown_space", "z = x / x_scale"},
             }},
            {"constraints",
             {
                 {"scaled_Ip", json_constraint_or_null(input.Ip)},
                 {"beta", json_constraint_or_null(input.beta)},
             }},
            {"timing", timing_json(samples_ms)},
            {"initial",
             {
                 {"ok", initial_ok},
                 {"x", json_array(input.x0)},
                 {"raw_residual", json_array(initial_raw)},
                 {"scaled_residual", json_array(initial_scaled)},
                 {"raw_norm",
                  initial_ok
                      ? norm2(std::span<const double, BenchShape::x_size>{
                            initial_raw.data(),
                            BenchShape::x_size,
                        })
                      : 1.0e20},
             }},
            {"final", solve_result_json(final)},
            {"success", final.accepted && final.info > 0},
        };

        std::cout << report.dump(2) << '\n';
        return EXIT_SUCCESS;
    }
    catch (const std::exception& exc)
    {
        std::cerr << "veqlib_pf_psin_uniform_benchmark: " << exc.what() << '\n';
        return EXIT_FAILURE;
    }
}
