#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
    constexpr double veqpy_x_scale_floor            = 1.0e-2;
    constexpr double veqpy_core_profile_prior       = 1.5e-1;
    constexpr double veqpy_fourier_profile_prior    = 5.0e-2;
    constexpr double veqpy_F_profile_prior          = 2.5e-1;
    constexpr double veqpy_kappa_profile_prior      = 1.0;

    struct CaseInput
    {
        std::string case_name = "PF_psin_uniform_Ip";
        std::array<double, BenchSource::sample_count> heat{};
        std::array<double, BenchSource::sample_count> current{};
        std::array<double, BenchShape::x_size>        x0{};
        std::array<double, BenchShape::x_size>        x_scale{};
        std::array<double, BenchShape::x_size>        residual_scale{};
        double a         = 1.05 / 1.85;
        double R0        = 1.05;
        double Z0        = 0.0;
        double B0        = 3.0;
        double ka        = 2.2;
        double c0_offset = 0.0;
        double s1_offset = 0.52359877559829887308;
        double Ip        = 3.7699111867885415;
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

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_heat = {
        -0.789683058574694,   -0.7925936329632908,  -0.7953979059157582,
        -0.7981175242684836,  -0.8007734699426484,  -0.8033829643453037,
        -0.8059602413311435,  -0.8085160883171674,  -0.811058402798229,
        -0.8135924782601793,  -0.8161210171627857,  -0.8186443668543919,
        -0.8211602449193173,  -0.82366353909573,    -0.8261466407410907,
        -0.8286005031116496,  -0.8310129351634512,  -0.8333638631960294,
        -0.8356348482673257,  -0.8378088362996647,  -0.8398452317867598,
        -0.8417128477427658,  -0.8433930625296288,  -0.8448046915616387,
        -0.8459047888622718,  -0.8467054195394251,  -0.8468550092753417,
        -0.8466915344330476,  -0.8459189587408882,  -0.844353928057723,
        -0.8418811874023397,  -0.8384860115442367,  -0.8339038151163928,
        -0.8279318154386046,  -0.8204511956034657,  -0.8111301194374045,
        -0.7996783681429246,  -0.7858549460849861,  -0.7692183653145488,
        -0.7492833542853738,  -0.7256038086449172,  -0.6975411195823384,
        -0.6643030775956854,  -0.6249342427240242,  -0.5782460674488447,
        -0.5227081311582821,  -0.4562798131359923,  -0.3761603550417907,
        -0.2784903156513912,  -0.15751845630174122, -0.004428769494182179,
    };

    constexpr std::array<double, BenchSource::sample_count> benchmark_scaled_current = {
        -0.2884247371510828,   -0.28903009704030685,  -0.28957011555052764,
        -0.29005075790401863,  -0.2904781886031161,   -0.2908575435658439,
        -0.2911919683745232,   -0.2914838716684553,   -0.2917344967330943,
        -0.2919437907627883,   -0.29211086667346203,  -0.2922346427093772,
        -0.2923145136340828,   -0.29232232828599186,  -0.2923070658458569,
        -0.2922075036101631,   -0.29204221633819877,  -0.2918026604476458,
        -0.2914746207679046,   -0.29104564954616613,  -0.2905098257293566,
        -0.28984851066007944,  -0.2890431993822433,   -0.2880834160319397,
        -0.2869422749878048,   -0.28559761849429993,  -0.28403023480001177,
        -0.2822015304104101,   -0.2800887239343856,   -0.27765576104313033,
        -0.274853815959196,    -0.27165492157456717,  -0.26800248440388824,
        -0.2638385916755866,   -0.25912135836327627,  -0.25377334278513203,
        -0.2477205935589761,   -0.2409002037441097,   -0.23321180576616163,
        -0.22454617787318162,  -0.2148028033443947,   -0.20384443014756062,
        -0.191504616045878,    -0.17758595047988296,  -0.1618458187382766,
        -0.143974507066849,    -0.12356213030456985,  -0.10004932271682981,
        -0.07266570051589127,  -0.040268147498330784, -0.0011074929612556953,
    };

    template <typename Shape>
    constexpr bool is_c_profile_id(size_t profile_id) noexcept
    {
        return profile_id >= Shape::c0_profile_id && profile_id <= Shape::c0_profile_id + Shape::M_max;
    }

    template <typename Shape>
    constexpr bool is_s_profile_id(size_t profile_id) noexcept
    {
        return profile_id >= Shape::c0_profile_id + Shape::M_max + 1 &&
               profile_id <= Shape::c0_profile_id + 2 * Shape::M_max;
    }

    template <typename Shape>
    constexpr double x_scale_profile_prior(size_t profile_id) noexcept
    {
        if (profile_id == Shape::h_profile_id || profile_id == Shape::v_profile_id ||
            profile_id == Shape::psin_profile_id)
            return veqpy_core_profile_prior;
        if (profile_id == Shape::kappa_profile_id)
            return veqpy_kappa_profile_prior;
        if (is_c_profile_id<Shape>(profile_id) || is_s_profile_id<Shape>(profile_id))
            return veqpy_fourier_profile_prior;
        if (profile_id == Shape::F_profile_id)
            return veqpy_F_profile_prior;
        return veqpy_F_profile_prior;
    }

    template <typename Shape>
    constexpr bool x_scale_offsetless(size_t profile_id) noexcept
    {
        return profile_id == Shape::h_profile_id || profile_id == Shape::v_profile_id ||
               profile_id == Shape::psin_profile_id;
    }

    profiles::ProfileRuntimeParams<BenchShape> profile_params_for_case(const CaseInput& input) noexcept
    {
        profiles::ProfileRuntimeParams<BenchShape> params{};
        params.offsets[BenchShape::kappa_profile_id] = input.ka;
        params.offsets[BenchShape::c_profile_id<0>()] = input.c0_offset;
        params.offsets[BenchShape::s_profile_id<1>()] = input.s1_offset;
        return params;
    }

    template <typename Shape>
    std::array<double, Shape::x_size> build_x_block_scale_vector(
        const std::array<double, Shape::x_size>&             x_guess,
        const profiles::ProfileRuntimeParams<Shape>& profile_params
    ) noexcept
    {
        std::array<double, Shape::x_size> scale{};
        scale.fill(1.0);
        for (size_t active_slot = 0; active_slot < Shape::active_count; ++active_slot)
        {
            const size_t profile_id = Shape::active_profile_ids[active_slot];
            const size_t length     = Shape::active_lengths[active_slot];
            if (length == 0)
                continue;

            double guess_norm2 = 0.0;
            for (size_t degree = 0; degree < length; ++degree)
            {
                const auto x_index = static_cast<size_t>(Shape::coeff_index[profile_id][degree]);
                guess_norm2 += x_guess[x_index] * x_guess[x_index];
            }

            const double guess_rms = std::sqrt(guess_norm2 / static_cast<double>(length));
            const double prior     = x_scale_profile_prior<Shape>(profile_id);
            const double offset_scale =
                x_scale_offsetless<Shape>(profile_id) ? 0.0 : std::abs(profile_params.offsets[profile_id]);
            double profile_scale = std::abs(profile_params.scales[profile_id]);
            if (std::abs(profile_scale - 1.0) <= 1.0e-12)
                profile_scale = prior;
            const double block_scale =
                std::max({offset_scale, profile_scale, prior, guess_rms, veqpy_x_scale_floor});

            for (size_t degree = 0; degree < length; ++degree)
            {
                const auto x_index = static_cast<size_t>(Shape::coeff_index[profile_id][degree]);
                scale[x_index]     = block_scale;
            }
        }
        return scale;
    }

    std::array<double, BenchShape::x_size> build_block_rms_residual_scale(const PackedVector& residual) noexcept
    {
        std::array<double, BenchShape::x_size> scale{};
        size_t                                 offset = 0;
        for (size_t block = 0; block < BenchShape::active_count; ++block)
        {
            const size_t length = BenchShape::active_lengths[block];
            double       total  = 0.0;
            for (size_t i = 0; i < length; ++i)
                total += residual[offset + i] * residual[offset + i];
            const double rms = std::sqrt(total / static_cast<double>(length));
            const double block_scale = rms > 1.0 ? rms : 1.0;
            for (size_t i = 0; i < length; ++i)
                scale[offset + i] = block_scale;
            offset += length;
        }
        return scale;
    }

    int parse_nonnegative_int(const char* flag, const char* value)
    {
        char*      end    = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < 0 || parsed > 1000000000L)
            throw std::runtime_error(std::string{"invalid "} + flag + " value: " + value);
        return static_cast<int>(parsed);
    }

    CaseInput build_inline_case(int repeat, int warmup)
    {
        CaseInput input{};
        input.heat           = benchmark_scaled_heat;
        input.current        = benchmark_scaled_current;
        input.repeat         = repeat;
        input.warmup         = warmup;
        input.x_scale        = build_x_block_scale_vector<BenchShape>(input.x0, profile_params_for_case(input));
        input.residual_scale.fill(1.0);
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
            op.params.a              = input.a;
            op.params.R0             = input.R0;
            op.params.Z0             = input.Z0;
            op.params.B0             = input.B0;
            op.params.Ip             = input.Ip;
            op.params.beta           = input.beta;
            op.params.fix_rho        = input.fix_rho;
            op.params.profile_params = profile_params_for_case(input);
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
        int repeat = 10;
        int warmup = 1;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--repeat")
            {
                if (++i >= argc)
                    throw std::runtime_error("--repeat requires a value");
                repeat = parse_nonnegative_int("--repeat", argv[i]);
            }
            else if (arg == "--warmup")
            {
                if (++i >= argc)
                    throw std::runtime_error("--warmup requires a value");
                warmup = parse_nonnegative_int("--warmup", argv[i]);
            }
            else if (arg == "--help" || arg == "-h")
            {
                std::cout << "usage: " << argv[0] << " [--repeat N] [--warmup N]\n";
                return EXIT_SUCCESS;
            }
            else
            {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }

        CaseInput    input = build_inline_case(repeat, warmup);
        SolveContext context{input};

        PackedVector initial_raw{};
        const bool   initial_ok = context.raw_residual(
            std::span<const double, BenchShape::x_size>{input.x0.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{initial_raw.data(), BenchShape::x_size}
        );
        context.input.residual_scale = build_block_rms_residual_scale(initial_raw);
        input.residual_scale         = context.input.residual_scale;
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
            {"route", "PF/psin/uniform/Ip"},
            {"source_topology",
             {
                 {"route", "PF"},
                 {"coordinate", "psin"},
                 {"nodes", "uniform"},
                 {"constraint", "Ip"},
             }},
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
            {"source",
             {
                 {"scaled_heat", json_array(input.heat)},
                 {"scaled_current", json_array(input.current)},
             }},
            {"normalization",
             {
                 {"x_scale", json_array(input.x_scale)},
                 {"residual_scale", json_array(input.residual_scale)},
                 {"x_scale_builder", "VEQPy _build_x_block_scale_vector equivalent"},
                 {"residual_scale_builder", "fast/block_rms initial residual block RMS"},
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
                 {"policy", "benchmark.py robust zero profile coefficients"},
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
