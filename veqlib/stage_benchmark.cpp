#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "kernel_case.h"
#include "kernel_runtime.h"
#include "kernel_topology.h"
#include "tensor.h"

namespace
{
    using Clock = std::chrono::steady_clock;
    using std::size_t;

    using veqlib_kernel_api::CaseInput;
    using veqlib_kernel_api::KernelOperator;
    using veqlib_kernel_api::KernelShape;
    using veqlib_kernel_api::PackedVector;
    using veqlib_kernel_api::SolverKind;
    using veqlib_kernel_api::build_inline_case;
    using veqlib_kernel_api::setup_for_case;
    using veqlib_kernel_api::solve_params_for_case;
    using tensor::uninitialized;

    enum class Stage
    {
        ProfilesActive,
        ProfilesAll,
        Geometry,
        SourceMaterialize,
        SourceUpdate,
        ResidualUpdate,
        ResidualPack,
        Evaluate,
        EvaluateRing,
    };

    struct Options
    {
        std::string stage     = "all";
        size_t      repeat    = 10;
        size_t      warmup    = 2;
        size_t      inner     = 1000;
        size_t      ring_size = 16;
        std::string output{};
    };

    [[noreturn]] void fail_usage(const std::string& message)
    {
        throw std::runtime_error(
            message +
            "\nusage: veqlib_stage_benchmark [--stage all|profiles_active|profiles_all|geometry|"
            "source_materialize|source_update|residual_update|residual_pack|evaluate|evaluate_ring] "
            "[--repeat N] [--warmup N] [--inner N] [--ring-size N] [--output PATH]");
    }

    size_t parse_size_arg(std::string_view name, std::string_view value)
    {
        size_t parsed = 0;
        size_t offset = 0;
        try
        {
            parsed = std::stoull(std::string{value}, &offset, 10);
        }
        catch (const std::exception&)
        {
            fail_usage(std::string{name} + " must be a positive integer");
        }
        if (offset != value.size() || parsed == 0)
            fail_usage(std::string{name} + " must be a positive integer");
        return parsed;
    }

    Options parse_options(int argc, char** argv)
    {
        Options options{};
        for (int i = 1; i < argc; ++i)
        {
            const std::string_view arg{argv[i]};
            if (arg == "--help" || arg == "-h")
            {
                std::cout
                    << "usage: veqlib_stage_benchmark [--stage all|profiles_active|profiles_all|geometry|"
                       "source_materialize|source_update|residual_update|residual_pack|evaluate|evaluate_ring] "
                       "[--repeat N] [--warmup N] [--inner N] [--ring-size N] [--output PATH]\n";
                std::exit(0);
            }
            if (i + 1 >= argc)
                fail_usage(std::string{arg} + " requires a value");
            const std::string_view value{argv[++i]};
            if (arg == "--stage")
                options.stage = std::string{value};
            else if (arg == "--repeat")
                options.repeat = parse_size_arg(arg, value);
            else if (arg == "--warmup")
                options.warmup = parse_size_arg(arg, value);
            else if (arg == "--inner")
                options.inner = parse_size_arg(arg, value);
            else if (arg == "--ring-size")
                options.ring_size = parse_size_arg(arg, value);
            else if (arg == "--output")
                options.output = std::string{value};
            else
                fail_usage("unknown argument: " + std::string{arg});
        }
        return options;
    }

    constexpr std::array<std::pair<std::string_view, Stage>, 9> stage_table{{
        {"profiles_active", Stage::ProfilesActive},
        {"profiles_all", Stage::ProfilesAll},
        {"geometry", Stage::Geometry},
        {"source_materialize", Stage::SourceMaterialize},
        {"source_update", Stage::SourceUpdate},
        {"residual_update", Stage::ResidualUpdate},
        {"residual_pack", Stage::ResidualPack},
        {"evaluate", Stage::Evaluate},
        {"evaluate_ring", Stage::EvaluateRing},
    }};

    Stage parse_stage(std::string_view name)
    {
        for (const auto& [stage_name, stage] : stage_table)
            if (name == stage_name)
                return stage;
        fail_usage("unknown stage: " + std::string{name});
    }

    std::string_view stage_name(Stage stage) noexcept
    {
        for (const auto& [name, value] : stage_table)
            if (value == stage)
                return name;
        return "unknown";
    }

    void do_not_optimize(double value) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "g"(value) : "memory");
#else
        static double sink = 0.0;
        sink += value;
#endif
    }

    std::vector<std::array<double, KernelShape::x_size>> make_state_ring(const CaseInput& input, size_t ring_size)
    {
        std::vector<std::array<double, KernelShape::x_size>> ring;
        ring.reserve(ring_size);
        for (size_t state = 0; state < ring_size; ++state)
        {
            auto x = input.x0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                const size_t pattern = ((state + 1) * (i + 3)) % 17;
                const double signed_pattern =
                    static_cast<double>(pattern) - 8.0;
                x[i] += 1.0e-8 * signed_pattern;
            }
            ring.push_back(x);
        }
        return ring;
    }

    struct BenchState
    {
        CaseInput                                      input;
        KernelOperator                                 op;
        PackedVector                                   out;
        std::vector<std::array<double, KernelShape::x_size>> ring;

        explicit BenchState(size_t ring_size)
            : input(build_inline_case(0, 0, SolverKind::Powell)),
              op(setup_for_case(input)),
              out(uninitialized),
              ring(make_state_ring(input, ring_size))
        {
            op.set_solve_params(solve_params_for_case(input));
        }

        std::span<const double, KernelShape::x_size> x_span() const noexcept
        {
            return std::span<const double, KernelShape::x_size>{input.x0.data(), KernelShape::x_size};
        }

        void prepare_profiles() noexcept
        {
            op.workspace.profiles.refresh_active(x_span(), op.plan.profile_params);
        }

        void prepare_geometry() noexcept
        {
            prepare_profiles();
            op.workspace.geometry.update(op.solve_params().a,
                                         op.solve_params().R0,
                                         op.solve_params().Z0,
                                         op.workspace.profiles);
        }

        void prepare_source_materialize() noexcept
        {
            prepare_geometry();
            op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
        }

        void prepare_source_update() noexcept
        {
            prepare_source_materialize();
            op.workspace.source_runtime.update_pf_psin_uniform_ip(op.workspace.geometry,
                                                                  op.solve_params().Ip,
                                                                  op.plan.n_axis_fix);
        }

        void prepare_residual_update() noexcept
        {
            prepare_source_update();
            op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
        }

        double profile_sink() const noexcept
        {
            return op.workspace.profiles.profile_field(KernelShape::psin_profile_id, 0, 0);
        }

        double geometry_sink() const noexcept { return op.workspace.geometry.surface_field(0, 0, 0); }

        double source_sink() const noexcept
        {
            return op.workspace.source_runtime.materialized_heat_input[0] +
                   op.workspace.source_runtime.materialized_current_input[0] +
                   op.workspace.source_runtime.alpha1 + op.workspace.source_runtime.alpha2;
        }

        double residual_sink() const noexcept { return out[0]; }
    };

    template <typename Callable>
    double time_stage_calls(size_t inner, Callable&& callable)
    {
        double sink = 0.0;
        const auto started = Clock::now();
        for (size_t i = 0; i < inner; ++i)
            sink += callable(i);
        const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - started).count();
        do_not_optimize(sink);
        return elapsed / static_cast<double>(inner);
    }

    double run_one_sample(Stage stage, size_t inner, size_t ring_size)
    {
        BenchState state{ring_size};

        switch (stage)
        {
        case Stage::ProfilesActive:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.profiles.refresh_active(state.x_span(), state.op.plan.profile_params);
                return state.profile_sink();
            });
        case Stage::ProfilesAll:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.profiles.load_fixed_from(state.op.plan.fixed_profiles);
                state.op.workspace.profiles.refresh_active(state.x_span(), state.op.plan.profile_params);
                return state.profile_sink();
            });
        case Stage::Geometry:
            state.prepare_profiles();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.geometry.update(state.op.solve_params().a,
                                                   state.op.solve_params().R0,
                                                   state.op.solve_params().Z0,
                                                   state.op.workspace.profiles);
                return state.geometry_sink();
            });
        case Stage::SourceMaterialize:
            state.prepare_geometry();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.source_runtime.materialize_profile_owned_psin(state.op.workspace.profiles,
                                                                                state.op.plan.n_axis_fix);
                return state.source_sink();
            });
        case Stage::SourceUpdate:
            state.prepare_source_materialize();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.source_runtime.update_pf_psin_uniform_ip(state.op.workspace.geometry,
                                                                           state.op.solve_params().Ip,
                                                                           state.op.plan.n_axis_fix);
                return state.source_sink();
            });
        case Stage::ResidualUpdate:
            state.prepare_source_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.residual.update_compact(state.op.workspace.source_runtime, state.op.workspace.geometry);
                return state.op.workspace.residual.surface_field(0, 0, 0);
            });
        case Stage::ResidualPack:
            state.prepare_residual_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.workspace.residual.pack_into(state.out,
                                                      state.op.solve_params().a,
                                                      state.op.solve_params().R0,
                                                      state.op.solve_params().B0);
                return state.residual_sink();
            });
        case Stage::Evaluate:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state.op.evaluate(state.x_span(), state.out);
                return state.residual_sink();
            });
        case Stage::EvaluateRing:
            return time_stage_calls(inner, [&](size_t i) noexcept {
                const auto& x = state.ring[i % state.ring.size()];
                state.op.evaluate(std::span<const double, KernelShape::x_size>{x.data(), KernelShape::x_size},
                                  state.out);
                return state.residual_sink();
            });
        }
        return 0.0;
    }

    nlohmann::json summarize_samples(const std::vector<double>& samples)
    {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());

        double median = 0.0;
        if (!sorted.empty())
        {
            const size_t mid = sorted.size() / 2;
            if ((sorted.size() % 2) == 0)
                median = 0.5 * (sorted[mid - 1] + sorted[mid]);
            else
                median = sorted[mid];
        }

        double p95 = 0.0;
        if (!sorted.empty())
        {
            const size_t index = (95 * (sorted.size() - 1) + 50) / 100;
            p95                = sorted[index];
        }

        double mean = 0.0;
        for (double sample : samples)
            mean += sample;
        if (!samples.empty())
            mean /= static_cast<double>(samples.size());

        double variance = 0.0;
        for (double sample : samples)
        {
            const double centered = sample - mean;
            variance += centered * centered;
        }
        const double stddev = samples.size() > 1 ? std::sqrt(variance / static_cast<double>(samples.size() - 1)) : 0.0;

        nlohmann::json out;
        out["samples_ns_per_call"] = samples;
        out["median_ns_per_call"]  = median;
        out["p95_ns_per_call"]     = p95;
        out["mean_ns_per_call"]    = mean;
        out["stddev_ns_per_call"]  = stddev;
        return out;
    }

    nlohmann::json run_stage(Stage stage, const Options& options)
    {
        for (size_t i = 0; i < options.warmup; ++i)
            do_not_optimize(run_one_sample(stage, options.inner, options.ring_size));

        std::vector<double> samples;
        samples.reserve(options.repeat);
        for (size_t i = 0; i < options.repeat; ++i)
            samples.push_back(run_one_sample(stage, options.inner, options.ring_size));

        nlohmann::json result = summarize_samples(samples);
        result["stage"]      = stage_name(stage);
        return result;
    }

    nlohmann::json topology_json()
    {
        nlohmann::json out;
        out["Nr"]                  = veqlib_kernel_api::KernelGrid::radial_nodes;
        out["Nt"]                  = veqlib_kernel_api::KernelGrid::theta_rows;
        out["x_size"]              = KernelShape::x_size;
        out["active_count"]        = KernelShape::active_count;
        out["L_max"]               = KernelShape::L_max;
        out["M_max"]               = KernelShape::M_max;
        out["K_max"]               = KernelShape::K_max;
        out["source_sample_count"] = veqlib_kernel_api::KernelSource::sample_count;
        return out;
    }

    nlohmann::json run_benchmark(const Options& options)
    {
        nlohmann::json root;
        root["schema"]    = "veqlib.stage_benchmark.v1";
        root["unit"]      = "ns_per_call";
        root["build"]     = "current-source";
        root["stage_arg"] = options.stage;
        root["repeat"]    = options.repeat;
        root["warmup"]    = options.warmup;
        root["inner"]     = options.inner;
        root["ring_size"] = options.ring_size;
        root["topology"]  = topology_json();
        root["results"]   = nlohmann::json::array();

        if (options.stage == "all")
        {
            for (const auto& [name, stage] : stage_table)
                root["results"].push_back(run_stage(stage, options));
        }
        else
        {
            root["results"].push_back(run_stage(parse_stage(options.stage), options));
        }
        return root;
    }

    void write_json(const nlohmann::json& data, const std::string& output)
    {
        if (output.empty())
        {
            std::cout << data.dump(2) << '\n';
            return;
        }

        const std::filesystem::path output_path{output};
        if (const auto parent = output_path.parent_path(); !parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream stream{output_path};
        if (!stream)
            throw std::runtime_error("failed to open output file: " + output);
        stream << data.dump(2) << '\n';
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options options = parse_options(argc, argv);
        write_json(run_benchmark(options), options.output);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "veqlib_stage_benchmark: " << error.what() << '\n';
        return 1;
    }
}
