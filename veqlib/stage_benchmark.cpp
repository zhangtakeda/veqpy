#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#include "geometry.h"
#include "grid.h"
#include "pf_psin_uniform_operator.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"

namespace
{
    using geometry::surface_R;
    using grid::Grid;
    using grid::Legendre;
    using grid::Spectral;
    using operator_pf::PfPsinUniformOperator;
    using profiles::OptimizedProfileShapeFromCountsT;
    using residual::surface_G;
    using source::axis_fix_count;
    using source::UniformSourceShape;
    using std::size_t;

    template <auto Counts>
    consteval size_t max_profile_count() noexcept
    {
        size_t value = 0;
        for (size_t count : Counts)
            value = value < count ? count : value;
        return value;
    }

    template <auto CFamilyCounts, auto SFamilyCounts>
    consteval size_t inferred_M_max() noexcept
    {
        constexpr size_t c_max = CFamilyCounts.size() == 0 ? 0 : CFamilyCounts.size() - 1;
        constexpr size_t s_max = SFamilyCounts.size();
        const size_t     value = c_max < s_max ? s_max : c_max;
        return value > 1 ? value : 1;
    }

    template <size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts>
    consteval size_t inferred_L_max() noexcept
    {
        size_t count             = HCount;
        count                    = count < VCount ? VCount : count;
        count                    = count < KappaCount ? KappaCount : count;
        count                    = count < PsinCount ? PsinCount : count;
        count                    = count < FCount ? FCount : count;
        constexpr size_t c_count = max_profile_count<CFamilyCounts>();
        constexpr size_t s_count = max_profile_count<SFamilyCounts>();
        count                    = count < c_count ? c_count : count;
        count                    = count < s_count ? s_count : count;
        return count > 1 ? count - 1 : 1;
    }

    template <size_t Mmax>
    consteval size_t inferred_K_max() noexcept
    {
        return Mmax > 2 ? Mmax : 2;
    }

    template <size_t Nr,
              size_t Nt,
              size_t SourceSamples,
              size_t HCount,
              size_t VCount,
              size_t KappaCount,
              size_t PsinCount,
              size_t FCount,
              auto   CFamilyCounts,
              auto   SFamilyCounts,
              typename QuadratureScheme,
              typename CalculusScheme>
    struct PfPsinUniformIpTopology
    {
        static constexpr size_t L_max =
            inferred_L_max<HCount, VCount, KappaCount, PsinCount, FCount, CFamilyCounts, SFamilyCounts>();
        static constexpr size_t M_max = inferred_M_max<CFamilyCounts, SFamilyCounts>();
        static constexpr size_t K_max = inferred_K_max<M_max>();

        using Shape    = OptimizedProfileShapeFromCountsT<L_max,
                                                          K_max,
                                                          HCount,
                                                          VCount,
                                                          KappaCount,
                                                          PsinCount,
                                                          FCount,
                                                          CFamilyCounts,
                                                          SFamilyCounts>;
        using Grid     = Grid<Nr, Nt, Shape::L_max, Shape::M_max, Shape::K_max, QuadratureScheme, CalculusScheme>;
        using Source   = UniformSourceShape<SourceSamples>;
        using Operator = PfPsinUniformOperator<Shape, Grid, Source>;
    };

    constexpr auto bench_c_counts = std::array<size_t, 1>{0};
    constexpr auto bench_s_counts = std::array<size_t, 1>{3};

    using BenchTopology =
        PfPsinUniformIpTopology<32, 16, 51, 3, 0, 6, 6, 0, bench_c_counts, bench_s_counts, Legendre, Spectral>;
    using BenchShape    = BenchTopology::Shape;
    using BenchGrid     = BenchTopology::Grid;
    using BenchSource   = BenchTopology::Source;
    using BenchOperator = BenchTopology::Operator;
    using PackedVector  = BenchOperator::PackedVector;

    static_assert(BenchShape::x_size == 18);
    static_assert(BenchShape::L_max == 5);
    static_assert(BenchShape::M_max == 1);
    static_assert(BenchShape::K_max == 2);

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_heat = {
        -0.789683058574694,    -0.7925936329632908, -0.7953979059157582, -0.7981175242684836, -0.8007734699426484,
        -0.8033829643453037,   -0.8059602413311435, -0.8085160883171674, -0.811058402798229,  -0.8135924782601793,
        -0.8161210171627857,   -0.8186443668543919, -0.8211602449193173, -0.82366353909573,   -0.8261466407410907,
        -0.8286005031116496,   -0.8310129351634512, -0.8333638631960294, -0.8356348482673257, -0.8378088362996647,
        -0.8398452317867598,   -0.8417128477427658, -0.8433930625296288, -0.8448046915616387, -0.8459047888622718,
        -0.8467054195394251,   -0.8468550092753417, -0.8466915344330476, -0.8459189587408882, -0.844353928057723,
        -0.8418811874023397,   -0.8384860115442367, -0.8339038151163928, -0.8279318154386046, -0.8204511956034657,
        -0.8111301194374045,   -0.7996783681429246, -0.7858549460849861, -0.7692183653145488, -0.7492833542853738,
        -0.7256038086449172,   -0.6975411195823384, -0.6643030775956854, -0.6249342427240242, -0.5782460674488447,
        -0.5227081311582821,   -0.4562798131359923, -0.3761603550417907, -0.2784903156513912, -0.15751845630174122,
        -0.004428769494182179,
    };

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_current = {
        -0.2884247371510828,    -0.28903009704030685, -0.28957011555052764, -0.29005075790401863, -0.2904781886031161,
        -0.2908575435658439,    -0.2911919683745232,  -0.2914838716684553,  -0.2917344967330943,  -0.2919437907627883,
        -0.29211086667346203,   -0.2922346427093772,  -0.2923145136340828,  -0.29232232828599186, -0.2923070658458569,
        -0.2922075036101631,    -0.29204221633819877, -0.2918026604476458,  -0.2914746207679046,  -0.29104564954616613,
        -0.2905098257293566,    -0.28984851066007944, -0.2890431993822433,  -0.2880834160319397,  -0.2869422749878048,
        -0.28559761849429993,   -0.28403023480001177, -0.2822015304104101,  -0.2800887239343856,  -0.27765576104313033,
        -0.274853815959196,     -0.27165492157456717, -0.26800248440388824, -0.2638385916755866,  -0.25912135836327627,
        -0.25377334278513203,   -0.2477205935589761,  -0.2409002037441097,  -0.23321180576616163, -0.22454617787318162,
        -0.2148028033443947,    -0.20384443014756062, -0.191504616045878,   -0.17758595047988296, -0.1618458187382766,
        -0.143974507066849,     -0.12356213030456985, -0.10004932271682981, -0.07266570051589127, -0.040268147498330784,
        -0.0011074929612556953,
    };

    enum class StageKind
    {
        ProfilesFixed,
        ProfilesActive,
        ProfilesAll,
        Geometry,
        SourceMaterialize,
        SourceUpdate,
        ResidualUpdate,
        ResidualPack,
        Evaluate,
    };

    struct Options
    {
        std::string stage  = "all";
        size_t      repeat = 30;
        size_t      warmup = 5;
        size_t      inner  = 1000;
    };

    struct Stats
    {
        double avg_ns    = 0.0;
        double min_ns    = 0.0;
        double max_ns    = 0.0;
        double median_ns = 0.0;
        double p95_ns    = 0.0;
        double std_ns    = 0.0;
    };

    volatile double benchmark_sink = 0.0;

    void compiler_barrier(const void* pointer) noexcept { asm volatile("" : : "r"(pointer) : "memory"); }

    std::span<const double, BenchShape::x_size> x_span(const std::array<double, BenchShape::x_size>& x) noexcept
    {
        return std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size};
    }

    void configure_operator(BenchOperator& op) noexcept
    {
        op.params.a                                                     = 1.05 / 1.85;
        op.params.R0                                                    = 1.05;
        op.params.Z0                                                    = 0.0;
        op.params.B0                                                    = 3.0;
        op.params.Ip                                                    = 3.7699111867885415;
        op.params.fix_rho                                               = 0.05;
        op.params.profile_params.offsets[BenchShape::kappa_profile_id]  = 2.2;
        op.params.profile_params.offsets[BenchShape::c_profile_id<0>()] = 0.0;
        op.params.profile_params.offsets[BenchShape::s_profile_id<1>()] = 0.52359877559829887308;
        op.set_uniform_sources(
            std::span<const double, BenchSource::sample_count>{
                benchmark_scaled_heat.data(),
                BenchSource::sample_count,
            },
            std::span<const double, BenchSource::sample_count>{
                benchmark_scaled_current.data(),
                BenchSource::sample_count,
            });
    }

    void refresh_profiles(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        op.profiles.refresh_fixed(op.params.profile_params);
        op.profiles.refresh_active(x, op.params.profile_params);
    }

    void prepare_geometry(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        op.geometry.update(op.params.a, op.params.R0, op.params.Z0, op.profiles);
    }

    void prepare_source_materialized(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        const size_t n_axis_fix = axis_fix_count<BenchGrid>(op.params.fix_rho);
        op.source_runtime.materialize_profile_owned_psin(op.profiles, n_axis_fix);
    }

    void prepare_source_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        prepare_geometry(op, x);
        const size_t n_axis_fix = axis_fix_count<BenchGrid>(op.params.fix_rho);
        op.source_runtime.materialize_profile_owned_psin(op.profiles, n_axis_fix);
        op.source_runtime.update_pf_ip_from_psin_uniform(op.geometry, op.params.Ip, n_axis_fix);
    }

    void prepare_residual_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        prepare_source_updated(op, x);
        op.residual.update_compact(op.source_runtime, op.geometry);
    }

    double consume_state(const BenchOperator& op, const PackedVector& packed) noexcept
    {
        return op.profiles.profile_field(BenchShape::psin_profile_id, 0, 0) +
               op.geometry.surface_field(surface_R, 0, 0) + op.source_runtime.alpha1 +
               op.residual.surface_fields(surface_G, 0, 0) + packed[0];
    }

    void run_stage_once(StageKind                                   stage,
                        BenchOperator&                              op,
                        std::span<const double, BenchShape::x_size> x,
                        PackedVector&                               packed)
    {
        const size_t n_axis_fix = axis_fix_count<BenchGrid>(op.params.fix_rho);
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            op.profiles.refresh_fixed(op.params.profile_params);
            compiler_barrier(op.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesActive:
            op.profiles.refresh_active(x, op.params.profile_params);
            compiler_barrier(op.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesAll:
            refresh_profiles(op, x);
            compiler_barrier(op.profiles.profile_fields.data());
            break;
        case StageKind::Geometry:
            op.geometry.update(op.params.a, op.params.R0, op.params.Z0, op.profiles);
            compiler_barrier(op.geometry.surface_fields.data());
            break;
        case StageKind::SourceMaterialize:
            op.source_runtime.materialize_profile_owned_psin(op.profiles, n_axis_fix);
            compiler_barrier(op.source_runtime.materialized_heat_input.data());
            break;
        case StageKind::SourceUpdate:
            op.source_runtime.update_pf_ip_from_psin_uniform(op.geometry, op.params.Ip, n_axis_fix);
            compiler_barrier(op.source_runtime.FFn_psin.data());
            break;
        case StageKind::ResidualUpdate:
            op.residual.update_compact(op.source_runtime, op.geometry);
            compiler_barrier(op.residual.surface_fields.data());
            break;
        case StageKind::ResidualPack:
            op.residual.pack_into(packed, op.params.a, op.params.R0, op.params.B0);
            compiler_barrier(packed.data());
            break;
        case StageKind::Evaluate:
            op.evaluate(x, packed);
            compiler_barrier(packed.data());
            break;
        }
    }

    const char* stage_name(StageKind stage) noexcept
    {
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            return "profiles_fixed";
        case StageKind::ProfilesActive:
            return "profiles_active";
        case StageKind::ProfilesAll:
            return "profiles_all";
        case StageKind::Geometry:
            return "geometry";
        case StageKind::SourceMaterialize:
            return "source_materialize";
        case StageKind::SourceUpdate:
            return "source_update";
        case StageKind::ResidualUpdate:
            return "residual_update";
        case StageKind::ResidualPack:
            return "residual_pack";
        case StageKind::Evaluate:
            return "evaluate";
        }
        return "unknown";
    }

    StageKind parse_stage_one(const std::string& value)
    {
        if (value == "profiles_fixed")
            return StageKind::ProfilesFixed;
        if (value == "profiles_active")
            return StageKind::ProfilesActive;
        if (value == "profiles_all")
            return StageKind::ProfilesAll;
        if (value == "geometry")
            return StageKind::Geometry;
        if (value == "source_materialize")
            return StageKind::SourceMaterialize;
        if (value == "source_update")
            return StageKind::SourceUpdate;
        if (value == "residual_update")
            return StageKind::ResidualUpdate;
        if (value == "residual_pack")
            return StageKind::ResidualPack;
        if (value == "evaluate")
            return StageKind::Evaluate;
        throw std::invalid_argument("unknown --stage: " + value);
    }

    std::vector<StageKind> stages_for(const std::string& value)
    {
        if (value != "all")
            return {parse_stage_one(value)};
        return {
            StageKind::ProfilesFixed,
            StageKind::ProfilesActive,
            StageKind::ProfilesAll,
            StageKind::Geometry,
            StageKind::SourceMaterialize,
            StageKind::SourceUpdate,
            StageKind::ResidualUpdate,
            StageKind::ResidualPack,
            StageKind::Evaluate,
        };
    }

    size_t parse_size_arg(const std::string& name, const std::string& value, bool allow_zero)
    {
        if (value.empty() || value.front() == '+' || value.front() == '-')
            throw std::invalid_argument(name + " must be a base-10 non-negative integer");

        size_t      parsed = 0;
        const char* first  = value.data();
        const char* last   = value.data() + value.size();
        const auto  result = std::from_chars(first, last, parsed, 10);
        if (result.ec != std::errc{} || result.ptr != last)
            throw std::invalid_argument(name + " must be a base-10 non-negative integer");
        if (!allow_zero && parsed == 0)
            throw std::invalid_argument(name + " must be positive");
        return parsed;
    }

    Options parse_args(int argc, char** argv)
    {
        Options options{};
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--help")
            {
                std::cout << "usage: veqlib_stage_benchmark [--stage all|profiles_fixed|profiles_active|"
                             "profiles_all|geometry|source_materialize|source_update|residual_update|"
                             "residual_pack|evaluate] [--repeat N] [--warmup N] [--inner N]\n";
                std::exit(0);
            }
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            const std::string value = argv[++i];
            if (arg == "--stage")
                options.stage = value;
            else if (arg == "--repeat")
                options.repeat = parse_size_arg(arg, value, false);
            else if (arg == "--warmup")
                options.warmup = parse_size_arg(arg, value, true);
            else if (arg == "--inner")
                options.inner = parse_size_arg(arg, value, false);
            else
                throw std::invalid_argument("unknown argument: " + arg);
        }
        return options;
    }

    Stats compute_stats(const std::vector<double>& samples)
    {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());

        double total = 0.0;
        for (double sample : samples)
            total += sample;
        const double avg = total / static_cast<double>(samples.size());

        double variance = 0.0;
        for (double sample : samples)
        {
            const double delta = sample - avg;
            variance += delta * delta;
        }
        variance /= static_cast<double>(samples.size());

        const size_t median_index = sorted.size() / 2;
        const double median =
            sorted.size() % 2 == 0 ? 0.5 * (sorted[median_index - 1] + sorted[median_index]) : sorted[median_index];
        const size_t p95_index = ((sorted.size() * 95 + 99) / 100) - 1;

        return {
            .avg_ns    = avg,
            .min_ns    = sorted.front(),
            .max_ns    = sorted.back(),
            .median_ns = median,
            .p95_ns    = sorted[p95_index],
            .std_ns    = std::sqrt(variance),
        };
    }

    void prepare_for_stage(StageKind stage, BenchOperator& op, std::span<const double, BenchShape::x_size> x)
    {
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            break;
        case StageKind::ProfilesActive:
            op.profiles.refresh_fixed(op.params.profile_params);
            break;
        case StageKind::ProfilesAll:
            break;
        case StageKind::Geometry:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceMaterialize:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceUpdate:
            prepare_source_materialized(op, x);
            op.geometry.update(op.params.a, op.params.R0, op.params.Z0, op.profiles);
            break;
        case StageKind::ResidualUpdate:
            prepare_source_updated(op, x);
            break;
        case StageKind::ResidualPack:
            prepare_residual_updated(op, x);
            break;
        case StageKind::Evaluate:
            break;
        }
    }

    nlohmann::json run_benchmark(StageKind stage, const Options& options)
    {
        auto op = std::make_unique<BenchOperator>();
        configure_operator(*op);
        std::array<double, BenchShape::x_size> x{};
        PackedVector                           packed{};
        const auto                             x_values = x_span(x);

        prepare_for_stage(stage, *op, x_values);

        for (size_t sample = 0; sample < options.warmup; ++sample)
            for (size_t i = 0; i < options.inner; ++i)
                run_stage_once(stage, *op, x_values, packed);

        std::vector<double> samples;
        samples.reserve(options.repeat);
        using clock = std::chrono::steady_clock;
        for (size_t sample = 0; sample < options.repeat; ++sample)
        {
            const auto start = clock::now();
            for (size_t i = 0; i < options.inner; ++i)
                run_stage_once(stage, *op, x_values, packed);
            const auto                                     stop    = clock::now();
            const std::chrono::duration<double, std::nano> elapsed = stop - start;
            samples.push_back(elapsed.count() / static_cast<double>(options.inner));
        }

        benchmark_sink += consume_state(*op, packed);
        const Stats stats = compute_stats(samples);

        return {
            {"stage", stage_name(stage)},
            {"repeat", options.repeat},
            {"warmup", options.warmup},
            {"inner", options.inner},
            {"calls", options.repeat * options.inner},
            {"avg_ns", stats.avg_ns},
            {"min_ns", stats.min_ns},
            {"max_ns", stats.max_ns},
            {"median_ns", stats.median_ns},
            {"p95_ns", stats.p95_ns},
            {"std_ns", stats.std_ns},
            {"samples_ns", samples},
        };
    }
} // namespace

int main(int argc, char** argv)
{
    try
    {
        const Options  options = parse_args(argc, argv);
        nlohmann::json results = nlohmann::json::array();
        for (StageKind stage : stages_for(options.stage))
            results.push_back(run_benchmark(stage, options));

        const double         sink   = benchmark_sink;
        const nlohmann::json report = {
            {"schema_version", 1},
            {"case_name", "PF_psin_uniform_Ip"},
            {"topology",
             {
                 {"Nr", BenchGrid::radial_nodes},
                 {"Nt", BenchGrid::theta_rows},
                 {"L_max", BenchShape::L_max},
                 {"M_max", BenchShape::M_max},
                 {"K_max", BenchShape::K_max},
                 {"x_size", BenchShape::x_size},
             }},
            {"results", results},
            {"sink", sink},
        };
        std::cout << report.dump(2) << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "veqlib_stage_benchmark: " << error.what() << '\n';
        return 2;
    }
}
