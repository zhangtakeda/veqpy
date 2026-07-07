// Native stage microbenchmark driver for generated Cxx Kernel artifacts.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
#include "math.h"
#include "tensor.h"

#ifndef VEQPY_CXX_STAGE_GIT_SHA
#define VEQPY_CXX_STAGE_GIT_SHA "unknown"
#endif
#ifndef VEQPY_CXX_STAGE_GIT_DIRTY
#define VEQPY_CXX_STAGE_GIT_DIRTY 0
#endif
#ifndef VEQPY_CXX_STAGE_CXX_COMPILER_ID
#define VEQPY_CXX_STAGE_CXX_COMPILER_ID "unknown"
#endif
#ifndef VEQPY_CXX_STAGE_CXX_COMPILER_VERSION
#define VEQPY_CXX_STAGE_CXX_COMPILER_VERSION "unknown"
#endif
#ifndef VEQPY_CXX_STAGE_BUILD_TYPE
#define VEQPY_CXX_STAGE_BUILD_TYPE "unknown"
#endif
#ifndef VEQPY_CXX_STAGE_FP_MODE
#define VEQPY_CXX_STAGE_FP_MODE "unknown"
#endif
#ifndef VEQPY_CXX_STAGE_NATIVE_OPTIMIZATIONS
#define VEQPY_CXX_STAGE_NATIVE_OPTIMIZATIONS 0
#endif
#ifndef VEQPY_CXX_STAGE_THIN_LTO
#define VEQPY_CXX_STAGE_THIN_LTO 0
#endif
#ifndef VEQPY_CXX_STAGE_ANALYSIS_BUILD
#define VEQPY_CXX_STAGE_ANALYSIS_BUILD 0
#endif
#ifndef VEQPY_CXX_STAGE_ENZYME
#define VEQPY_CXX_STAGE_ENZYME 0
#endif

namespace
{
    using Clock = std::chrono::steady_clock;
    using std::size_t;

    using cxx_kernel_api::RuntimeCase;
    using cxx_kernel_api::CompiledOperator;
    using cxx_kernel_api::CompiledShape;
    using cxx_kernel_api::PackedVector;
    using cxx_kernel_api::SolverKind;
    using cxx_kernel_api::build_inline_case;
    using cxx_kernel_api::setup_for_case;
    using cxx_kernel_api::runtime_scalars_for_case;
    using tensor::uninitialized;

    enum class Stage
    {
        ProfilesActive,
        ProfilesAll,
        GeometryPhase,
        GeometryPhaseSincos,
        GeometryMetricCompute,
        Geometry,
        SourceMaterialize,
        SourceMatCopyPsinR,
        SourceMatRegularizePsinR,
        SourceMatDerivAccum,
        SourceMatStoreCoordinate,
        SourceMatCopyProfileRoot,
        SourceMatPrepareQueries,
        SourceMatInterpolate,
        SourceUpdate,
        SourceUpdateFillIntegrand,
        SourceUpdateAccumIntegrand,
        SourceUpdateNormalizePsinR,
        SourceUpdateDerivAccum,
        SourceUpdateCopySources,
        SourceUpdateRegularizeFfn,
        SourceUpdateAlpha,
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
            "\nusage: cxx_stage_benchmark [--stage all|<stage-name>] "
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
                    << "usage: cxx_stage_benchmark [--stage all|<stage-name>] "
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

    constexpr std::array<std::pair<std::string_view, Stage>, 26> stage_table{{
        {"profiles_active", Stage::ProfilesActive},
        {"profiles_all", Stage::ProfilesAll},
        {"geometry_phase", Stage::GeometryPhase},
        {"geometry_phase_sincos", Stage::GeometryPhaseSincos},
        {"geometry_metric_compute", Stage::GeometryMetricCompute},
        {"geometry", Stage::Geometry},
        {"source_materialize", Stage::SourceMaterialize},
        {"source_mat_copy_psin_r", Stage::SourceMatCopyPsinR},
        {"source_mat_regularize_psin_r", Stage::SourceMatRegularizePsinR},
        {"source_mat_deriv_accum", Stage::SourceMatDerivAccum},
        {"source_mat_store_coordinate", Stage::SourceMatStoreCoordinate},
        {"source_mat_copy_profile_root", Stage::SourceMatCopyProfileRoot},
        {"source_mat_prepare_queries", Stage::SourceMatPrepareQueries},
        {"source_mat_interpolate", Stage::SourceMatInterpolate},
        {"source_update", Stage::SourceUpdate},
        {"source_update_fill_integrand", Stage::SourceUpdateFillIntegrand},
        {"source_update_accum_integrand", Stage::SourceUpdateAccumIntegrand},
        {"source_update_normalize_psin_r", Stage::SourceUpdateNormalizePsinR},
        {"source_update_deriv_accum", Stage::SourceUpdateDerivAccum},
        {"source_update_copy_sources", Stage::SourceUpdateCopySources},
        {"source_update_regularize_ffn", Stage::SourceUpdateRegularizeFfn},
        {"source_update_alpha", Stage::SourceUpdateAlpha},
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

    template <typename T>
    void do_not_optimize_address(const T* ptr) noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        asm volatile("" : : "g"(ptr) : "memory");
#else
        (void)ptr;
#endif
    }

    template <typename ProfilesRuntime>
    void fill_geometry_phase_row(size_t i,
                                 const ProfilesRuntime& runtime_profiles,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_values,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_r_values,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_t_values,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_rr_values,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_rt_values,
                                 std::array<double, ProfilesRuntime::grid::theta_rows>& tb_tt_values) noexcept
    {
        using Shape       = typename ProfilesRuntime::shape;
        using ProfileGrid = typename ProfilesRuntime::grid;

        for (size_t j = 0; j < ProfileGrid::theta_rows; ++j)
        {
            double tb_ij    = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb);
            double tb_r_ij  = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_r);
            double tb_t_ij  = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_t);
            double tb_rr_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_rr);
            double tb_rt_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_rt);
            double tb_tt_ij = runtime_profiles.boundary_phase_base_field(i, j, ProfilesRuntime::phase_tb_tt);

            for (size_t active_index = 0; active_index < Shape::active_c_order_count; ++active_index)
            {
                const size_t order  = Shape::active_c_orders[active_index];
                const double c_i    = runtime_profiles.c_family_fields(order, i, geometry::detail::profile_value);
                const double c_r_i  = runtime_profiles.c_family_fields(order, i, geometry::detail::profile_radial);
                const double c_rr_i = runtime_profiles.c_family_fields(order, i, geometry::detail::profile_radial2);

                if (order == 0)
                {
                    tb_ij += c_i;
                    tb_r_ij += c_r_i;
                    tb_rr_ij += c_rr_i;
                    continue;
                }

                const double cos_kt    = ProfileGrid::cos_mtheta(order, j);
                const double k_sin_kt  = ProfileGrid::m_sin_mtheta(order, j);
                const double k2_cos_kt = ProfileGrid::m2_cos_mtheta(order, j);

                tb_ij += c_i * cos_kt;
                tb_r_ij += c_r_i * cos_kt;
                tb_t_ij -= c_i * k_sin_kt;
                tb_rr_ij += c_rr_i * cos_kt;
                tb_rt_ij -= c_r_i * k_sin_kt;
                tb_tt_ij -= c_i * k2_cos_kt;
            }

            for (size_t active_index = 0; active_index < Shape::active_s_order_count; ++active_index)
            {
                const size_t order     = Shape::active_s_orders[active_index];
                const double s_i       = runtime_profiles.s_family_fields(order, i, geometry::detail::profile_value);
                const double s_r_i     = runtime_profiles.s_family_fields(order, i, geometry::detail::profile_radial);
                const double s_rr_i    = runtime_profiles.s_family_fields(order, i, geometry::detail::profile_radial2);
                const double sin_kt    = ProfileGrid::sin_mtheta(order, j);
                const double k_cos_kt  = ProfileGrid::m_cos_mtheta(order, j);
                const double k2_sin_kt = ProfileGrid::m2_sin_mtheta(order, j);

                tb_ij += s_i * sin_kt;
                tb_r_ij += s_r_i * sin_kt;
                tb_t_ij += s_i * k_cos_kt;
                tb_rr_ij += s_rr_i * sin_kt;
                tb_rt_ij += s_r_i * k_cos_kt;
                tb_tt_ij -= s_i * k2_sin_kt;
            }

            tb_values[j]    = tb_ij;
            tb_r_values[j]  = tb_r_ij;
            tb_t_values[j]  = tb_t_ij;
            tb_rr_values[j] = tb_rr_ij;
            tb_rt_values[j] = tb_rt_ij;
            tb_tt_values[j] = tb_tt_ij;
        }
    }

    template <typename ProfilesRuntime>
    double benchmark_geometry_phase(const ProfilesRuntime& runtime_profiles) noexcept
    {
        using ProfileGrid = typename ProfilesRuntime::grid;

        double sink = 0.0;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_r_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_t_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rr_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rt_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_tt_values;

        for (size_t i = 0; i < ProfileGrid::radial_nodes; ++i)
        {
            fill_geometry_phase_row(
                i, runtime_profiles, tb_values, tb_r_values, tb_t_values, tb_rr_values, tb_rt_values, tb_tt_values);
            for (size_t j = 0; j < ProfileGrid::theta_rows; ++j)
                sink += tb_values[j] + tb_r_values[j] + tb_t_values[j] + tb_rr_values[j] + tb_rt_values[j] +
                        tb_tt_values[j];
        }
        return sink;
    }

    template <typename ProfilesRuntime>
    double benchmark_geometry_phase_sincos(const ProfilesRuntime& runtime_profiles) noexcept
    {
        using ProfileGrid = typename ProfilesRuntime::grid;

        double sink = 0.0;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_r_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_t_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rr_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rt_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_tt_values;

        for (size_t i = 0; i < ProfileGrid::radial_nodes; ++i)
        {
            fill_geometry_phase_row(
                i, runtime_profiles, tb_values, tb_r_values, tb_t_values, tb_rr_values, tb_rt_values, tb_tt_values);
            for (size_t j = 0; j < ProfileGrid::theta_rows; ++j)
            {
                double sin_tb = 0.0;
                double cos_tb = 0.0;
                math::relaxed_sincos(tb_values[j], sin_tb, cos_tb);
                sink += sin_tb + cos_tb + tb_r_values[j] + tb_t_values[j] + tb_rr_values[j] + tb_rt_values[j] +
                        tb_tt_values[j];
            }
        }
        return sink;
    }

    template <typename ProfilesRuntime>
    double benchmark_geometry_metric_compute(double a, double R0, const ProfilesRuntime& runtime_profiles) noexcept
    {
        using Shape       = typename ProfilesRuntime::shape;
        using ProfileGrid = typename ProfilesRuntime::grid;

        double sink = 0.0;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_r_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_t_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rr_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_rt_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> tb_tt_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> sin_tb_values;
        alignas(tensor::detail::simd_alignment) std::array<double, ProfileGrid::theta_rows> cos_tb_values;

        for (size_t i = 0; i < ProfileGrid::radial_nodes; ++i)
        {
            const double rho_i  = ProfileGrid::nodes[i];
            const double h_i    = runtime_profiles.profile_field(Shape::h_profile_id, i, geometry::detail::profile_value);
            const double h_r_i  = runtime_profiles.profile_field(Shape::h_profile_id, i, geometry::detail::profile_radial);
            const double h_rr_i = runtime_profiles.profile_field(Shape::h_profile_id, i, geometry::detail::profile_radial2);
            const double v_r_i  = runtime_profiles.profile_field(Shape::v_profile_id, i, geometry::detail::profile_radial);
            const double v_rr_i = runtime_profiles.profile_field(Shape::v_profile_id, i, geometry::detail::profile_radial2);
            const double k_i =
                runtime_profiles.profile_field(Shape::kappa_profile_id, i, geometry::detail::profile_value);
            const double k_r_i =
                runtime_profiles.profile_field(Shape::kappa_profile_id, i, geometry::detail::profile_radial);
            const double k_rr_i =
                runtime_profiles.profile_field(Shape::kappa_profile_id, i, geometry::detail::profile_radial2);

            double sum_J          = 0.0;
            double sum_JR         = 0.0;
            double sum_gttdivJR   = 0.0;
            double sum_gttdivJR_r = 0.0;
            double sum_JdivR      = 0.0;

            fill_geometry_phase_row(
                i, runtime_profiles, tb_values, tb_r_values, tb_t_values, tb_rr_values, tb_rt_values, tb_tt_values);
            for (size_t j = 0; j < ProfileGrid::theta_rows; ++j)
                math::relaxed_sincos(tb_values[j], sin_tb_values[j], cos_tb_values[j]);

            for (size_t j = 0; j < ProfileGrid::theta_rows; ++j)
            {
                const double sin_t = ProfileGrid::sin_mtheta(1, j);
                const double cos_t = ProfileGrid::cos_mtheta(1, j);

                const double tb_r_ij   = tb_r_values[j];
                const double tb_t_ij   = tb_t_values[j];
                const double tb_rr_ij  = tb_rr_values[j];
                const double tb_rt_ij  = tb_rt_values[j];
                const double tb_tt_ij  = tb_tt_values[j];
                const double cos_tb_ij = cos_tb_values[j];
                const double sin_tb_ij = sin_tb_values[j];

                double R_ij = R0 + a * (h_i + rho_i * cos_tb_ij);
                if (R_ij < 1.0e-6)
                    R_ij = 1.0e-6;

                const double R_r_ij  = a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij);
                const double R_t_ij  = -a * rho_i * sin_tb_ij * tb_t_ij;
                const double R_rr_ij = a * (h_rr_i - 2.0 * sin_tb_ij * tb_r_ij -
                                            rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij));
                const double R_rt_ij =
                    -a * (sin_tb_ij * tb_t_ij + rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij));
                const double R_tt_ij = -a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij);

                const double Z_r_ij  = a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t);
                const double Z_t_ij  = -a * rho_i * k_i * cos_t;
                const double Z_rr_ij = a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t);
                const double Z_rt_ij = -a * (k_i + rho_i * k_r_i) * cos_t;
                const double Z_tt_ij = a * rho_i * k_i * sin_t;

                double J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij;
                if (J_ij < 1.0e-6)
                    J_ij = 1.0e-6;

                const double J_r_ij = -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij -
                                        R_t_ij * Z_rr_ij);
                const double J_t_ij = -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij -
                                        R_t_ij * Z_rt_ij);
                const double JR_ij   = J_ij * R_ij;
                const double JR_r_ij = J_r_ij * R_ij + J_ij * R_r_ij;
                const double JR_t_ij = J_t_ij * R_ij + J_ij * R_t_ij;
                const double JdivR_ij = J_ij / R_ij;
                const double grt_ij   = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij;
                const double grt_t_ij =
                    R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij;
                const double gtt_ij   = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij;
                const double gtt_r_ij = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij);
                const double inv_JR   = 1.0 / JR_ij;
                const double grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR;
                const double gttdivJR_ij   = gtt_ij * inv_JR;
                const double gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR;

                sink += sin_tb_ij + R_ij + R_t_ij + Z_t_ij + J_ij + JdivR_ij + grtdivJR_t_ij + gttdivJR_ij +
                        gttdivJR_r_ij;
                sum_J += J_ij;
                sum_JR += JR_ij;
                sum_gttdivJR += gttdivJR_ij;
                sum_gttdivJR_r += gttdivJR_r_ij;
                sum_JdivR += JdivR_ij;
            }

            constexpr double theta_scale = 2.0 * geometry::detail::pi / static_cast<double>(ProfileGrid::theta_rows);
            constexpr double mean_scale  = 1.0 / static_cast<double>(ProfileGrid::theta_rows);
            sink += sum_J * theta_scale + sum_JR * theta_scale * 2.0 * geometry::detail::pi +
                    sum_gttdivJR * mean_scale + sum_gttdivJR_r * mean_scale + sum_JdivR * mean_scale;
        }
        return sink;
    }

    std::vector<std::array<double, CompiledShape::x_size>> make_state_ring(const RuntimeCase& input, size_t ring_size)
    {
        std::vector<std::array<double, CompiledShape::x_size>> ring;
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
        using SourceRuntime     = CompiledOperator::Source;
        using SourceRadialVector = SourceRuntime::RadialVector;

        RuntimeCase                                      input;
        CompiledOperator                                 op;
        PackedVector                                   out;
        std::vector<std::array<double, CompiledShape::x_size>> ring;
        SourceRadialVector                             source_scratch0{uninitialized};
        SourceRadialVector                             source_scratch1{uninitialized};

        explicit BenchState(size_t ring_size)
            : input(build_inline_case(0, 0, SolverKind::Powell)),
              op(setup_for_case(input)),
              out(uninitialized),
              ring(make_state_ring(input, ring_size))
        {
            op.set_runtime_scalars(runtime_scalars_for_case(input));
        }

        std::span<const double, CompiledShape::x_size> x_span() const noexcept
        {
            return std::span<const double, CompiledShape::x_size>{input.x0.data(), CompiledShape::x_size};
        }

        void prepare_profiles() noexcept
        {
            op.workspace.profiles.refresh_active(x_span(), op.plan.profile_params);
        }

        void prepare_geometry() noexcept
        {
            prepare_profiles();
            op.workspace.geometry.update(op.runtime_scalars().a,
                                         op.runtime_scalars().R0,
                                         op.runtime_scalars().Z0,
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
                                                                  op.runtime_scalars().Ip,
                                                                  op.plan.n_axis_fix);
        }

        void prepare_residual_update() noexcept
        {
            prepare_source_update();
            op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
        }

        double profile_sink() const noexcept
        {
            return op.workspace.profiles.profile_field(CompiledShape::psin_profile_id, 0, 0);
        }

        double geometry_sink() const noexcept { return op.workspace.geometry.surface_field(0, 0, 0); }

        double source_sink() const noexcept
        {
            return op.workspace.source_runtime.materialized_heat_input[0] +
                   op.workspace.source_runtime.materialized_current_input[0] +
                   op.workspace.source_runtime.alpha1 + op.workspace.source_runtime.alpha2;
        }

        double source_root_sink() const noexcept
        {
            return op.workspace.source_runtime.benchmark_root_field(source::root_psin, 0) +
                   op.workspace.source_runtime.benchmark_root_field(source::root_psin_r, 0) +
                   op.workspace.source_runtime.benchmark_root_field(source::root_psin_rr, 0);
        }

        double source_query_sink() const noexcept
        {
            return op.workspace.source_runtime.benchmark_query_sink(0);
        }

        double source_scratch_sink() const noexcept
        {
            return source_scratch0[0] + source_scratch0[SourceRuntime::radial_nodes - 1] + source_scratch1[0] +
                   source_scratch1[SourceRuntime::radial_nodes - 1];
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
        auto state = std::make_unique<BenchState>(ring_size);

        switch (stage)
        {
        case Stage::ProfilesActive:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.profiles.refresh_active(state->x_span(), state->op.plan.profile_params);
                return state->profile_sink();
            });
        case Stage::ProfilesAll:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.profiles.load_fixed_from(state->op.plan.fixed_profiles);
                state->op.workspace.profiles.refresh_active(state->x_span(), state->op.plan.profile_params);
                return state->profile_sink();
            });
        case Stage::GeometryPhase:
            state->prepare_profiles();
            return time_stage_calls(inner, [&](size_t) noexcept {
                return benchmark_geometry_phase(state->op.workspace.profiles);
            });
        case Stage::GeometryPhaseSincos:
            state->prepare_profiles();
            return time_stage_calls(inner, [&](size_t) noexcept {
                return benchmark_geometry_phase_sincos(state->op.workspace.profiles);
            });
        case Stage::GeometryMetricCompute:
            state->prepare_profiles();
            return time_stage_calls(inner, [&](size_t) noexcept {
                return benchmark_geometry_metric_compute(
                    state->op.runtime_scalars().a, state->op.runtime_scalars().R0, state->op.workspace.profiles);
            });
        case Stage::Geometry:
            state->prepare_profiles();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.geometry.update(state->op.runtime_scalars().a,
                                                    state->op.runtime_scalars().R0,
                                                    state->op.runtime_scalars().Z0,
                                                    state->op.workspace.profiles);
                return state->geometry_sink();
            });
        case Stage::SourceMaterialize:
            state->prepare_geometry();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.materialize_profile_owned_psin(state->op.workspace.profiles,
                                                                                 state->op.plan.n_axis_fix);
                return state->source_sink();
            });
        case Stage::SourceMatCopyPsinR:
            state->prepare_geometry();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_copy_profile_psin_r(state->op.workspace.profiles);
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_root_data());
                return state->source_root_sink();
            });
        case Stage::SourceMatRegularizePsinR:
            state->prepare_geometry();
            state->op.workspace.source_runtime.benchmark_copy_profile_psin_r(state->op.workspace.profiles);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_regularize_psin_r(state->op.plan.n_axis_fix);
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_root_data());
                return state->source_root_sink();
            });
        case Stage::SourceMatDerivAccum:
            state->prepare_geometry();
            state->op.workspace.source_runtime.benchmark_copy_profile_psin_r(state->op.workspace.profiles);
            state->op.workspace.source_runtime.benchmark_regularize_psin_r(state->op.plan.n_axis_fix);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_DA_psin_packed_into(state->source_scratch0,
                                                                                 state->source_scratch1);
                do_not_optimize_address(state->source_scratch0.data());
                do_not_optimize_address(state->source_scratch1.data());
                return state->source_scratch_sink();
            });
        case Stage::SourceMatStoreCoordinate:
            state->prepare_geometry();
            state->op.workspace.source_runtime.benchmark_copy_profile_psin_r(state->op.workspace.profiles);
            state->op.workspace.source_runtime.benchmark_regularize_psin_r(state->op.plan.n_axis_fix);
            state->op.workspace.source_runtime.benchmark_DA_psin_packed_into(state->source_scratch0,
                                                                             state->source_scratch1);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_store_psin_coordinate(state->source_scratch1);
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_root_data());
                return state->source_root_sink();
            });
        case Stage::SourceMatCopyProfileRoot:
            state->prepare_geometry();
            state->op.workspace.source_runtime.benchmark_copy_profile_psin_r(state->op.workspace.profiles);
            state->op.workspace.source_runtime.benchmark_regularize_psin_r(state->op.plan.n_axis_fix);
            state->op.workspace.source_runtime.benchmark_DA_psin_packed_into(state->source_scratch0,
                                                                             state->source_scratch1);
            state->op.workspace.source_runtime.benchmark_store_psin_coordinate(state->source_scratch1);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_copy_source_target_to_profile_root();
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_profile_root_data());
                return state->source_root_sink();
            });
        case Stage::SourceMatPrepareQueries:
            state->prepare_source_materialize();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_prepare_psin_queries();
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_query_data());
                return state->source_query_sink();
            });
        case Stage::SourceMatInterpolate:
            state->prepare_source_materialize();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_interpolate_pair();
                do_not_optimize_address(state->op.workspace.source_runtime.materialized_heat_input.data());
                do_not_optimize_address(state->op.workspace.source_runtime.materialized_current_input.data());
                return state->source_sink();
            });
        case Stage::SourceUpdate:
            state->prepare_source_materialize();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.update_pf_psin_uniform_ip(state->op.workspace.geometry,
                                                                            state->op.runtime_scalars().Ip,
                                                                            state->op.plan.n_axis_fix);
                return state->source_sink();
            });
        case Stage::SourceUpdateFillIntegrand:
            state->prepare_source_materialize();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(state->source_scratch0,
                                                                                   state->op.workspace.geometry);
                do_not_optimize_address(state->source_scratch0.data());
                return state->source_scratch_sink();
            });
        case Stage::SourceUpdateAccumIntegrand:
            state->prepare_source_materialize();
            state->op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(state->source_scratch0,
                                                                               state->op.workspace.geometry);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_A_integrand_into(state->source_scratch1,
                                                                              state->source_scratch0);
                do_not_optimize_address(state->source_scratch1.data());
                return state->source_scratch_sink();
            });
        case Stage::SourceUpdateNormalizePsinR:
            state->prepare_source_materialize();
            state->op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(state->source_scratch0,
                                                                               state->op.workspace.geometry);
            state->op.workspace.source_runtime.benchmark_A_integrand_into(state->source_scratch1,
                                                                          state->source_scratch0);
            return time_stage_calls(inner, [&](size_t) noexcept {
                const double integral_prof =
                    state->op.workspace.source_runtime.benchmark_normalize_psin_r_into(state->source_scratch0,
                                                                                       state->source_scratch1,
                                                                                       state->op.workspace.geometry,
                                                                                       state->op.plan.n_axis_fix);
                do_not_optimize_address(state->source_scratch0.data());
                do_not_optimize_address(state->op.workspace.source_runtime.benchmark_root_data());
                return integral_prof + state->source_scratch_sink();
            });
        case Stage::SourceUpdateDerivAccum:
            state->prepare_source_materialize();
            state->op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(state->source_scratch0,
                                                                               state->op.workspace.geometry);
            state->op.workspace.source_runtime.benchmark_A_integrand_into(state->source_scratch1,
                                                                          state->source_scratch0);
            state->op.workspace.source_runtime.benchmark_normalize_psin_r_into(state->source_scratch0,
                                                                               state->source_scratch1,
                                                                               state->op.workspace.geometry,
                                                                               state->op.plan.n_axis_fix);
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_DA_psin_packed_into(state->source_scratch0,
                                                                                 state->source_scratch1);
                do_not_optimize_address(state->source_scratch0.data());
                do_not_optimize_address(state->source_scratch1.data());
                return state->source_scratch_sink();
            });
        case Stage::SourceUpdateCopySources:
            state->prepare_source_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_copy_materialized_sources();
                do_not_optimize_address(state->op.workspace.source_runtime.Pn_psin.data());
                do_not_optimize_address(state->op.workspace.source_runtime.FFn_psin.data());
                return state->source_sink();
            });
        case Stage::SourceUpdateRegularizeFfn:
            state->prepare_source_update();
            state->op.workspace.source_runtime.benchmark_copy_materialized_sources();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_regularize_ffn_psin(state->op.plan.n_axis_fix);
                do_not_optimize_address(state->op.workspace.source_runtime.FFn_psin.data());
                return state->source_sink();
            });
        case Stage::SourceUpdateAlpha:
            state->prepare_source_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.source_runtime.benchmark_update_alpha_from_integral(state->op.workspace.geometry,
                                                                                       state->op.runtime_scalars().Ip,
                                                                                       1.0);
                return state->source_sink();
            });
        case Stage::ResidualUpdate:
            state->prepare_source_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.residual.update_compact(state->op.workspace.source_runtime,
                                                            state->op.workspace.geometry);
                return state->op.workspace.residual.surface_field(0, 0, 0);
            });
        case Stage::ResidualPack:
            state->prepare_residual_update();
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.workspace.residual.pack_into(state->out,
                                                       state->op.runtime_scalars().a,
                                                       state->op.runtime_scalars().R0,
                                                       state->op.runtime_scalars().B0);
                return state->residual_sink();
            });
        case Stage::Evaluate:
            return time_stage_calls(inner, [&](size_t) noexcept {
                state->op.evaluate(state->x_span(), state->out);
                return state->residual_sink();
            });
        case Stage::EvaluateRing:
            return time_stage_calls(inner, [&](size_t i) noexcept {
                const auto& x = state->ring[i % state->ring.size()];
                state->op.evaluate(std::span<const double, CompiledShape::x_size>{x.data(), CompiledShape::x_size},
                                   state->out);
                return state->residual_sink();
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
        out["Nr"]                  = cxx_kernel_api::CompiledGrid::radial_nodes;
        out["Nt"]                  = cxx_kernel_api::CompiledGrid::theta_rows;
        out["x_size"]              = CompiledShape::x_size;
        out["active_count"]        = CompiledShape::active_count;
        out["L_max"]               = CompiledShape::L_max;
        out["M_max"]               = CompiledShape::M_max;
        out["K_max"]               = CompiledShape::K_max;
        out["source_sample_count"] = cxx_kernel_api::CompiledSource::sample_count;
        return out;
    }

    nlohmann::json build_metadata_json()
    {
        nlohmann::json out;
        out["git_sha"]              = VEQPY_CXX_STAGE_GIT_SHA;
        out["git_dirty"]            = VEQPY_CXX_STAGE_GIT_DIRTY != 0;
        out["compiler_id"]          = VEQPY_CXX_STAGE_CXX_COMPILER_ID;
        out["compiler_version"]     = VEQPY_CXX_STAGE_CXX_COMPILER_VERSION;
        out["build_type"]           = VEQPY_CXX_STAGE_BUILD_TYPE;
        out["fp_mode"]              = VEQPY_CXX_STAGE_FP_MODE;
        out["native_optimizations"] = VEQPY_CXX_STAGE_NATIVE_OPTIMIZATIONS != 0;
        out["thin_lto"]             = VEQPY_CXX_STAGE_THIN_LTO != 0;
        out["analysis_build"]       = VEQPY_CXX_STAGE_ANALYSIS_BUILD != 0;
        out["enzyme"]               = VEQPY_CXX_STAGE_ENZYME != 0;
        return out;
    }

    nlohmann::json run_benchmark(const Options& options)
    {
        nlohmann::json root;
        root["schema"]         = "veqpy.cxx.stage_benchmark.v2";
        root["unit"]           = "ns_per_call";
        root["build"]          = "current-source";
        root["build_metadata"] = build_metadata_json();
        root["stage_arg"]      = options.stage;
        root["repeat"]         = options.repeat;
        root["warmup"]         = options.warmup;
        root["inner"]          = options.inner;
        root["ring_size"]      = options.ring_size;
        root["topology"]       = topology_json();
        root["results"]        = nlohmann::json::array();

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
        std::cerr << "cxx_stage_benchmark: " << error.what() << '\n';
        return 1;
    }
}
