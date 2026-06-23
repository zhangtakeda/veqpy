#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nlohmann/json.hpp>

#include "test_cli.h"
#include "tensor.h"

namespace nb = nanobind;

namespace veqlib_python
{
    using std::size_t;

    using veqlib_pf_psin_uniform_benchmark_cli::BenchGrid;
    using veqlib_pf_psin_uniform_benchmark_cli::BenchShape;
    using veqlib_pf_psin_uniform_benchmark_cli::BenchSource;
    using veqlib_pf_psin_uniform_benchmark_cli::CaseInput;
    using veqlib_pf_psin_uniform_benchmark_cli::PackedVector;
    using veqlib_pf_psin_uniform_benchmark_cli::ScanConfig;
    using veqlib_pf_psin_uniform_benchmark_cli::SolveContext;
    using veqlib_pf_psin_uniform_benchmark_cli::SolveResult;
    using veqlib_pf_psin_uniform_benchmark_cli::SolverKind;
    using veqlib_pf_psin_uniform_benchmark_cli::build_block_rms_residual_scale;
    using veqlib_pf_psin_uniform_benchmark_cli::build_inline_case;
    using veqlib_pf_psin_uniform_benchmark_cli::json_array;
    using veqlib_pf_psin_uniform_benchmark_cli::norm2;
    using veqlib_pf_psin_uniform_benchmark_cli::parse_scan_policy;
    using veqlib_pf_psin_uniform_benchmark_cli::parse_solver_kind;
    using veqlib_pf_psin_uniform_benchmark_cli::run_parameter_scan_report;
    using veqlib_pf_psin_uniform_benchmark_cli::run_solver_once;
    using veqlib_pf_psin_uniform_benchmark_cli::solve_result_json;
    using veqlib_pf_psin_uniform_benchmark_cli::solver_entrypoint;
    using veqlib_pf_psin_uniform_benchmark_cli::solver_info_succeeded;
    using veqlib_pf_psin_uniform_benchmark_cli::solver_jacobian;
    using veqlib_pf_psin_uniform_benchmark_cli::solver_method;
    using veqlib_pf_psin_uniform_benchmark_cli::supported_enzyme_width;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_acceptance_threshold;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_hybr_eps;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_hybr_factor;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_hybr_mode;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_max_residual;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_maxfev;
    using veqlib_pf_psin_uniform_benchmark_cli::veqpy_requested_max_evaluations;

    using tensor::uninitialized;

    using CliEntrypoint = int (*)(int, char**);
    using PackedArrayView =
        nb::ndarray<nb::numpy, const double, nb::shape<BenchShape::x_size>, nb::c_contig>;
    using MutablePackedArrayView =
        nb::ndarray<nb::numpy, double, nb::shape<BenchShape::x_size>, nb::c_contig>;
    using AlphaArrayView = nb::ndarray<nb::numpy, const double, nb::shape<2>, nb::c_contig>;

    struct StreamRedirect
    {
        std::ostream& stream;
        std::streambuf* old_buffer;

        StreamRedirect(std::ostream& target, std::ostringstream& replacement)
            : stream(target),
              old_buffer(target.rdbuf(replacement.rdbuf()))
        {
        }

        ~StreamRedirect() { stream.rdbuf(old_buffer); }
    };

    std::string run_cli_json(CliEntrypoint entrypoint, const std::vector<std::string>& args)
    {
        if (args.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("too many CLI arguments");

        std::vector<char*> argv;
        argv.reserve(args.size());
        for (const std::string& arg : args)
            argv.push_back(const_cast<char*>(arg.c_str()));

        std::ostringstream out;
        std::ostringstream err;
        const StreamRedirect redirect_out{std::cout, out};
        const StreamRedirect redirect_err{std::cerr, err};
        const int code = entrypoint(static_cast<int>(argv.size()), argv.data());

        if (code != EXIT_SUCCESS)
        {
            const std::string message = err.str().empty() ? out.str() : err.str();
            throw std::runtime_error(message.empty() ? "VEQlib CLI entrypoint failed" : message);
        }
        return out.str();
    }

    std::string solve_pf_psin_uniform_ip_json(
        int                repeat,
        int                warmup,
        const std::string& solver,
        int                enzyme_width
    )
    {
        std::vector<std::string> args{
            "veqlib_ext",
            "--repeat",
            std::to_string(repeat),
            "--warmup",
            std::to_string(warmup),
            "--solver",
            solver,
            "--enzyme-width",
            std::to_string(enzyme_width),
        };
        return run_cli_json(veqlib_pf_psin_uniform_benchmark_cli::run, args);
    }

    std::string scan_pf_psin_uniform_ip_json(
        int                points,
        const std::string& policy,
        double             relative_step,
        const std::string& solver,
        int                enzyme_width
    )
    {
        if (points <= 0)
            throw std::runtime_error("points must be positive for PF/psin/uniform/Ip scan");
        if (!supported_enzyme_width(enzyme_width))
            throw std::runtime_error("enzyme_width must be one of 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 18");
        if (relative_step < -1.0 || relative_step > 1.0)
            throw std::runtime_error("relative_step must be in [-1, 1]");

        ScanConfig scan{};
        scan.points        = points;
        scan.policy        = parse_scan_policy(policy.c_str());
        scan.relative_step = relative_step;

        const SolverKind solver_kind = parse_solver_kind(solver.c_str());
        CaseInput        input       = build_inline_case(0, 0, solver_kind, enzyme_width);
        return run_parameter_scan_report(input, scan).dump(2);
    }

    std::string validate_pf_psin_uniform_ip_json()
    {
        return run_cli_json(veqlib_pf_psin_uniform_validation_cli::run, {"veqlib_ext"});
    }

    std::string stage_pf_psin_uniform_ip_json(
        const std::string& stage,
        size_t             repeat,
        size_t             warmup,
        size_t             inner,
        size_t             ring_size
    )
    {
        std::vector<std::string> args{
            "veqlib_ext",
            "--stage",
            stage,
            "--repeat",
            std::to_string(repeat),
            "--warmup",
            std::to_string(warmup),
            "--inner",
            std::to_string(inner),
            "--ring-size",
            std::to_string(ring_size),
        };
        return run_cli_json(veqlib_stage_benchmark_cli::run, args);
    }

    std::unique_ptr<SolveContext> make_context(SolverKind solver, int enzyme_width)
    {
        if (!supported_enzyme_width(enzyme_width))
            throw std::runtime_error("enzyme_width must be one of 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 18");

        CaseInput input = build_inline_case(0, 0, solver, enzyme_width);
        auto      context = std::make_unique<SolveContext>(input);

        PackedVector initial_raw{uninitialized};
        context->raw_residual(
            std::span<const double, BenchShape::x_size>{input.x0.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{initial_raw.data(), BenchShape::x_size}
        );
        context->input.residual_scale = build_block_rms_residual_scale(initial_raw);
        return context;
    }

    nlohmann::json solver_json(const CaseInput& input)
    {
        return {
            {"method", solver_method(input.solver)},
            {"entrypoint", solver_entrypoint(input.solver)},
            {"jacobian", solver_jacobian(input)},
            {"enzyme_width",
             input.solver == SolverKind::EnzymeJacobian ? nlohmann::json(input.enzyme_width)
                                                         : nlohmann::json(nullptr)},
            {"max_residual", veqpy_max_residual},
            {"acceptance_threshold", veqpy_acceptance_threshold()},
            {"requested_max_evaluations", veqpy_requested_max_evaluations},
            {"maxfev", veqpy_maxfev},
            {"eps", veqpy_hybr_eps},
            {"factor", veqpy_hybr_factor},
            {"diag_mode", veqpy_hybr_mode},
        };
    }

    nlohmann::json case_prefix_json(SolveContext& context)
    {
        const CaseInput& input = context.input;

        PackedVector initial_raw{uninitialized};
        context.raw_residual(
            std::span<const double, BenchShape::x_size>{input.x0.data(), BenchShape::x_size},
            std::span<double, BenchShape::x_size>{initial_raw.data(), BenchShape::x_size}
        );
        PackedVector initial_scaled{uninitialized};
        for (size_t i = 0; i < BenchShape::x_size; ++i)
            initial_scaled[i] = initial_raw[i] / input.residual_scale[i];

        return {
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
            {"solver", solver_json(input)},
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
            {"constraints", {{"scaled_Ip", input.Ip}}},
            {"initial",
             {
                 {"x", json_array(input.x0)},
                 {"policy", "benchmark.py robust zero profile coefficients"},
                 {"raw_residual", json_array(initial_raw)},
                 {"scaled_residual", json_array(initial_scaled)},
                 {"raw_norm",
                  norm2(std::span<const double, BenchShape::x_size>{
                      initial_raw.data(),
                      BenchShape::x_size,
                  })},
             }},
        };
    }

    PackedArrayView packed_view(const double* data, nb::handle owner)
    {
        return PackedArrayView(data, {BenchShape::x_size}, owner);
    }

    AlphaArrayView alpha_view(const double* data, nb::handle owner)
    {
        return AlphaArrayView(data, {2}, owner);
    }


    nb::dict topology_metadata_dict(const CaseInput& input)
    {
        nb::dict source;
        source["route"]       = "PF";
        source["coordinate"]  = "psin";
        source["constraint"]  = "Ip";
        source["nodes"]       = "uniform";
        source["sample_count"] = BenchSource::sample_count;

        nb::dict grid;
        grid["Nr"]         = BenchGrid::radial_nodes;
        grid["Nt"]         = BenchGrid::theta_rows;
        grid["L_max"]      = BenchShape::L_max;
        grid["M_max"]      = BenchShape::M_max;
        grid["K_max"]      = BenchShape::K_max;
        grid["quadrature"] = "legendre";
        grid["calculus"]   = "spectral";

        nb::dict solver;
        solver["method"]       = solver_method(input.solver);
        solver["entrypoint"]   = solver_entrypoint(input.solver);
        solver["jacobian"]     = solver_jacobian(input);
        solver["enzyme_width"] = input.enzyme_width;

        nb::dict out;
        out["schema"]         = "veqlib.kernel.metadata.v1";
        out["backend"]        = "veqlib.nanobind";
        out["route"]          = "PF/psin/uniform/Ip";
        out["x_size"]         = BenchShape::x_size;
        out["active_count"]   = BenchShape::active_count;
        out["source"]         = source;
        out["grid"]           = grid;
        out["solver"]         = solver;
        out["case_mutation"]  = "not_implemented_mvp";
        return out;
    }

    class KernelSolver
    {
    public:
        explicit KernelSolver(const std::string& solver = "residual", int enzyme_width = 1)
            : solver_(parse_solver_kind(solver.c_str())),
              enzyme_width_(enzyme_width),
              context_(make_context(solver_, enzyme_width_))
        {
        }

        nb::dict metadata() const { return topology_metadata_dict(context_->input); }

        std::string metadata_json() const { return case_prefix_json(*context_).dump(2); }

        void set_case_json(const std::string& payload)
        {
            if (payload.empty())
            {
                last_case_json_ = "{}";
                return;
            }
            const nlohmann::json data = nlohmann::json::parse(payload);
            if (!data.is_object())
                throw std::runtime_error("KernelSolver.set_case_json expects a JSON object");
            if (!data.empty())
            {
                throw std::runtime_error(
                    "KernelSolver MVP does not yet support runtime case mutation; "
                    "construct a new benchmark case through the legacy PF facade"
                );
            }
            last_case_json_ = data.dump();
        }

        void warmup(size_t count)
        {
            for (size_t i = 0; i < count; ++i)
                (void)run_solver_once(*context_);
        }

        std::string solve_json()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_ = run_solver_once(*context_);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();

            const nlohmann::json report = {
                {"schema", "veqlib.kernel.solve_result.v1"},
                {"route", "PF/psin/uniform/Ip"},
                {"x_size", BenchShape::x_size},
                {"solver", solver_json(context_->input)},
                {"elapsed_ms", last_elapsed_ms_},
                {"final", solve_result_json(last_result_)},
                {"success",
                 last_result_.accepted && solver_info_succeeded(context_->input.solver, last_result_.info)},
            };
            return report.dump(2);
        }

        nb::tuple solve_direct()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_ = run_solver_once(*context_);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();

            nb::object owner = nb::cast(this, nb::rv_policy::reference);
            return nb::make_tuple(
                last_elapsed_ms_,
                last_result_.accepted && solver_info_succeeded(context_->input.solver, last_result_.info),
                last_result_.info,
                last_result_.nfev,
                last_result_.njev,
                last_result_.callbacks,
                last_result_.jacobian_component_evaluations,
                last_result_.jvp_evaluations,
                last_result_.linear_iterations,
                last_result_.raw_norm,
                last_result_.scaled_norm,
                packed_view(last_result_.x.data(), owner),
                packed_view(last_result_.raw.data(), owner),
                packed_view(last_result_.scaled.data(), owner),
                alpha_view(last_result_.alpha.data(), owner)
            );
        }

        void residual_var_into(PackedArrayView x, MutablePackedArrayView out)
        {
            context_->raw_residual(
                std::span<const double, BenchShape::x_size>{x.data(), BenchShape::x_size},
                std::span<double, BenchShape::x_size>{out.data(), BenchShape::x_size}
            );
        }

        std::string stage_benchmark_json(
            const std::string& stage,
            size_t             repeat,
            size_t             warmup,
            size_t             inner,
            size_t             ring_size
        ) const
        {
            return stage_pf_psin_uniform_ip_json(stage, repeat, warmup, inner, ring_size);
        }

        double last_elapsed_ms() const noexcept { return last_elapsed_ms_; }

    private:
        SolverKind                    solver_;
        int                           enzyme_width_;
        std::unique_ptr<SolveContext> context_;
        SolveResult                   last_result_{};
        std::string                   last_case_json_ = "{}";
        double                        last_elapsed_ms_ = 0.0;
    };

    class PfPsinUniformIpSolver
    {
    public:
        explicit PfPsinUniformIpSolver(const std::string& solver = "residual", int enzyme_width = 1)
            : solver_(parse_solver_kind(solver.c_str())),
              enzyme_width_(enzyme_width),
              context_(make_context(solver_, enzyme_width_))
        {
        }

        void warmup(size_t count)
        {
            for (size_t i = 0; i < count; ++i)
                (void)run_solver_once(*context_);
        }

        std::string initial_json() { return case_prefix_json(*context_).dump(2); }

        std::string solve_json()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_ = run_solver_once(*context_);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();

            const nlohmann::json report = {
                {"route", "PF/psin/uniform/Ip"},
                {"x_size", BenchShape::x_size},
                {"solver", solver_json(context_->input)},
                {"elapsed_ms", last_elapsed_ms_},
                {"final", solve_result_json(last_result_)},
                {"success",
                 last_result_.accepted && solver_info_succeeded(context_->input.solver, last_result_.info)},
            };
            return report.dump(2);
        }

        nb::tuple solve_direct()
        {
            const auto started = std::chrono::steady_clock::now();
            last_result_ = run_solver_once(*context_);
            const auto elapsed = std::chrono::steady_clock::now() - started;
            last_elapsed_ms_ = std::chrono::duration<double, std::milli>(elapsed).count();

            nb::object owner = nb::cast(this, nb::rv_policy::reference);
            return nb::make_tuple(
                last_elapsed_ms_,
                last_result_.accepted && solver_info_succeeded(context_->input.solver, last_result_.info),
                last_result_.info,
                last_result_.nfev,
                last_result_.njev,
                last_result_.callbacks,
                last_result_.jacobian_component_evaluations,
                last_result_.jvp_evaluations,
                last_result_.linear_iterations,
                last_result_.raw_norm,
                last_result_.scaled_norm,
                packed_view(last_result_.x.data(), owner),
                packed_view(last_result_.raw.data(), owner),
                packed_view(last_result_.scaled.data(), owner),
                alpha_view(last_result_.alpha.data(), owner)
            );
        }

        double last_elapsed_ms() const noexcept { return last_elapsed_ms_; }

    private:
        SolverKind                    solver_;
        int                           enzyme_width_;
        std::unique_ptr<SolveContext> context_;
        SolveResult                   last_result_{};
        double                        last_elapsed_ms_ = 0.0;
    };
} // namespace veqlib_python

NB_MODULE(veqlib_ext, module)
{
    module.doc() =
        "Single-thread nanobind bridge for VEQlib kernel validation and timing.";

    module.def(
        "solve_pf_psin_uniform_ip_json",
        &veqlib_python::solve_pf_psin_uniform_ip_json,
        nb::arg("repeat") = 10,
        nb::arg("warmup") = 1,
        nb::arg("solver") = "residual",
        nb::arg("enzyme_width") = 1,
        "Run the CLI-equivalent PF/psin/uniform/Ip solve benchmark and return JSON."
    );
    module.def(
        "validate_pf_psin_uniform_ip_json",
        &veqlib_python::validate_pf_psin_uniform_ip_json,
        "Run the PF/psin/uniform/Ip validation payload and return JSON."
    );
    module.def(
        "scan_pf_psin_uniform_ip_json",
        &veqlib_python::scan_pf_psin_uniform_ip_json,
        nb::arg("points"),
        nb::arg("policy") = "warm",
        nb::arg("relative_step") = 5.0e-3,
        nb::arg("solver") = "residual",
        nb::arg("enzyme_width") = 1,
        "Run the PF/psin/uniform/Ip Ip-scan path and return JSON."
    );
    module.def(
        "stage_pf_psin_uniform_ip_json",
        &veqlib_python::stage_pf_psin_uniform_ip_json,
        nb::arg("stage") = "all",
        nb::arg("repeat") = 30,
        nb::arg("warmup") = 5,
        nb::arg("inner") = 1000,
        nb::arg("ring_size") = 16,
        "Run the PF/psin/uniform/Ip stage benchmark and return JSON."
    );


    nb::class_<veqlib_python::KernelSolver>(module, "KernelSolver")
        .def(
            nb::init<const std::string&, int>(),
            nb::arg("solver") = "residual",
            nb::arg("enzyme_width") = 1
        )
        .def("metadata", &veqlib_python::KernelSolver::metadata)
        .def("metadata_json", &veqlib_python::KernelSolver::metadata_json)
        .def("set_case_json", &veqlib_python::KernelSolver::set_case_json, nb::arg("payload"))
        .def("warmup", &veqlib_python::KernelSolver::warmup, nb::arg("count"))
        .def("solve_json", &veqlib_python::KernelSolver::solve_json)
        .def(
            "solve_direct",
            &veqlib_python::KernelSolver::solve_direct,
            "Run one solve and return scalars plus read-only NumPy views without JSON serialization."
        )
        .def(
            "residual_var_into",
            &veqlib_python::KernelSolver::residual_var_into,
            nb::arg("x"),
            nb::arg("out"),
            "Evaluate the raw variational residual into a caller-owned packed output array."
        )
        .def(
            "stage_benchmark_json",
            &veqlib_python::KernelSolver::stage_benchmark_json,
            nb::arg("stage") = "all",
            nb::arg("repeat") = 30,
            nb::arg("warmup") = 5,
            nb::arg("inner") = 1000,
            nb::arg("ring_size") = 16
        )
        .def_prop_ro("last_elapsed_ms", &veqlib_python::KernelSolver::last_elapsed_ms);

    nb::class_<veqlib_python::PfPsinUniformIpSolver>(module, "PfPsinUniformIpSolver")
        .def(
            nb::init<const std::string&, int>(),
            nb::arg("solver") = "residual",
            nb::arg("enzyme_width") = 1
        )
        .def("warmup", &veqlib_python::PfPsinUniformIpSolver::warmup, nb::arg("count"))
        .def("initial_json", &veqlib_python::PfPsinUniformIpSolver::initial_json)
        .def("solve_json", &veqlib_python::PfPsinUniformIpSolver::solve_json)
        .def(
            "solve_direct",
            &veqlib_python::PfPsinUniformIpSolver::solve_direct,
            "Run one solve and return scalars plus read-only NumPy views without JSON serialization."
        )
        .def_prop_ro("last_elapsed_ms", &veqlib_python::PfPsinUniformIpSolver::last_elapsed_ms);
}
