#pragma once

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

#include "config.h"
#include "geometry.h"
#include "grid.h"
#include "math.h"
#include "source/pf_psin_uniform_ip.h"
#include "profiles.h"
#include "residual.h"
#include "source.h"
#include "tensor.h"

namespace veqlib_stage_benchmark_cli
{

namespace
{
    using config::DefaultTopology;
    using geometry::radial_Kn;
    using geometry::radial_Kn_r;
    using geometry::radial_Ln_r;
    using geometry::radial_S_r;
    using geometry::radial_V_r;
    using geometry::surface_R;
    using grid::Grid;
    using grid::Legendre;
    using grid::Spectral;
    using math::cos;
    using math::sin;
    using source::PfPsinUniformIpOperator;
    using profiles::OptimizedProfileShapeFromCountsT;
    using residual::surface_G;
    using source::axis_fix_count;
    using source::UniformSourceShape;
    using std::size_t;
    using tensor::Matrix;
    using tensor::Vector;
    using tensor::uninitialized;

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
        using Operator = PfPsinUniformIpOperator<Shape, Grid, Source>;
    };

    constexpr auto bench_c_counts = DefaultTopology::c_family_counts;
    constexpr auto bench_s_counts = DefaultTopology::s_family_counts;

    using BenchTopology =
        PfPsinUniformIpTopology<DefaultTopology::Nr,
                                DefaultTopology::Nt,
                                51,
                                DefaultTopology::h_count,
                                DefaultTopology::v_count,
                                DefaultTopology::kappa_count,
                                DefaultTopology::psin_count,
                                DefaultTopology::F_count,
                                bench_c_counts,
                                bench_s_counts,
                                Legendre,
                                Spectral>;
    using BenchShape    = BenchTopology::Shape;
    using BenchGrid     = BenchTopology::Grid;
    using BenchSource   = BenchTopology::Source;
    using BenchOperator = BenchTopology::Operator;
    using PackedVector        = BenchOperator::PackedVector;
    using SourceRadialVector  = Vector<double, BenchGrid::radial_nodes>;
    using ResidualMomentRows  = BenchOperator::Residual::MomentRows;

    static_assert(BenchShape::L_max == DefaultTopology::L_max);
    static_assert(BenchShape::M_max == DefaultTopology::M_max);
    static_assert(BenchShape::K_max == DefaultTopology::K_max);

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
        GeometryPhase,
        GeometryPhaseSincos,
        GeometryPhaseSplitSincos,
        GeometryMetricNoStore,
        Geometry,
        SourceMaterialize,
        SourceCopyRegularize,
        SourceDpsin,
        SourceApsin,
        SourceDApsin,
        SourceDApsinBlock4,
        SourceInterpolatePair,
        SourceIntegrand,
        SourceAIntegrandRowdot,
        SourceAIntegrand,
        SourceNormalize,
        SourceDNormalized,
        SourceAlpha,
        SourceUpdate,
        ResidualUpdate,
        ResidualThetaReduce,
        ResidualRadialProject,
        ResidualPack,
        Evaluate,
        EvaluateRing,
    };

    struct Options
    {
        std::string stage  = "all";
        size_t      repeat = 30;
        size_t      warmup = 5;
        size_t      inner  = 1000;
        size_t      ring_size = 16;
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
        BenchOperator::RuntimeParams params{};
        params.a                                                     = 1.05 / 1.85;
        params.R0                                                    = 1.05;
        params.Z0                                                    = 0.0;
        params.B0                                                    = 3.0;
        params.Ip                                                    = 3.7699111867885415;
        params.fix_rho                                               = 0.05;
        params.profile_params.offsets[BenchShape::kappa_profile_id]  = 2.2;
        params.profile_params.offsets[BenchShape::c_profile_id<0>()] = 0.0;
        params.profile_params.offsets[BenchShape::s_profile_id<1>()] = 0.52359877559829887308;
        op.set_runtime_params(params);
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
        const auto& params = op.runtime_params();
        op.refresh_static_plan();
        op.workspace.profiles.refresh_active(x, params.profile_params);
    }

    void prepare_geometry(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        const auto& params = op.runtime_params();
        refresh_profiles(op, x);
        op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
    }

    void prepare_source_materialized(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
    }

    void prepare_source_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        const auto& params = op.runtime_params();
        prepare_geometry(op, x);
        op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
        op.workspace.source_runtime.update_pf_psin_uniform_ip(
            op.workspace.geometry, params.Ip, op.plan.n_axis_fix);
    }

    void prepare_residual_updated(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        prepare_source_updated(op, x);
        op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
    }

    void prepare_source_profile_root(BenchOperator& op, std::span<const double, BenchShape::x_size> x) noexcept
    {
        refresh_profiles(op, x);
        op.workspace.source_runtime.benchmark_copy_profile_psin_r(op.workspace.profiles);
        op.workspace.source_runtime.benchmark_regularize_psin_r(op.plan.n_axis_fix);
    }

    void prepare_source_integrand(BenchOperator&                              op,
                                  std::span<const double, BenchShape::x_size> x,
                                  SourceRadialVector&                         integrand) noexcept
    {
        const auto& params = op.runtime_params();
        prepare_source_materialized(op, x);
        op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
        op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(integrand, op.workspace.geometry);
    }

    void prepare_source_accumulated_integrand(BenchOperator&                              op,
                                              std::span<const double, BenchShape::x_size> x,
                                              SourceRadialVector&                         accumulated) noexcept
    {
        SourceRadialVector integrand{uninitialized};
        prepare_source_integrand(op, x, integrand);
        op.workspace.source_runtime.benchmark_A_integrand_into(accumulated, integrand);
    }

    void prepare_source_normalized_psin(BenchOperator&                              op,
                                        std::span<const double, BenchShape::x_size> x,
                                        SourceRadialVector&                         normalized) noexcept
    {
        SourceRadialVector accumulated{uninitialized};
        prepare_source_accumulated_integrand(op, x, accumulated);
        (void)op.workspace.source_runtime.benchmark_normalize_psin_r_into(
            normalized, accumulated, op.workspace.geometry, op.plan.n_axis_fix);
    }

    enum class GeometryProbeMode
    {
        Phase,
        PhaseSincos,
        PhaseSplitSincos,
        MetricNoStore,
    };

    template <GeometryProbeMode Mode>
    void run_geometry_probe(BenchOperator& op) noexcept
    {
        const auto& params = op.runtime_params();
        using Shape = BenchShape;

        constexpr size_t profile_value   = 0;
        constexpr size_t profile_radial  = 1;
        constexpr size_t profile_radial2 = 2;
        constexpr size_t c_limit         = BenchGrid::harmonic_rows;
        constexpr size_t s_limit         = BenchGrid::harmonic_rows;
        constexpr double probe_pi        = 3.141592653589793238462643383279502884;

        const auto& runtime_profiles = op.workspace.profiles;
        double      sink             = 0.0;

        for (size_t i = 0; i < BenchGrid::radial_nodes; ++i)
        {
            const double rho_i   = BenchGrid::nodes[i];
            const double c0_i    = runtime_profiles.c_family_fields(0, i, profile_value);
            const double c0_r_i  = runtime_profiles.c_family_fields(0, i, profile_radial);
            const double c0_rr_i = runtime_profiles.c_family_fields(0, i, profile_radial2);

            Matrix<double, c_limit, 3> c_fields{uninitialized};
            Matrix<double, s_limit, 3> s_fields{uninitialized};
            for (size_t order = 1; order < c_limit; ++order)
            {
                c_fields(order, profile_value) = runtime_profiles.c_family_fields(order, i, profile_value);
                c_fields(order, profile_radial) = runtime_profiles.c_family_fields(order, i, profile_radial);
                c_fields(order, profile_radial2) = runtime_profiles.c_family_fields(order, i, profile_radial2);
            }
            for (size_t order = 1; order < s_limit; ++order)
            {
                s_fields(order, profile_value) = runtime_profiles.s_family_fields(order, i, profile_value);
                s_fields(order, profile_radial) = runtime_profiles.s_family_fields(order, i, profile_radial);
                s_fields(order, profile_radial2) = runtime_profiles.s_family_fields(order, i, profile_radial2);
            }

            if constexpr (Mode == GeometryProbeMode::MetricNoStore)
            {
                const double h_i    = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_value);
                const double h_r_i  = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial);
                const double h_rr_i = runtime_profiles.profile_field(Shape::h_profile_id, i, profile_radial2);
                const double v_r_i  = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial);
                const double v_rr_i = runtime_profiles.profile_field(Shape::v_profile_id, i, profile_radial2);
                const double k_i    = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_value);
                const double k_r_i  = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial);
                const double k_rr_i = runtime_profiles.profile_field(Shape::kappa_profile_id, i, profile_radial2);

                double sum_J          = 0.0;
                double sum_JR         = 0.0;
                double sum_gttdivJR   = 0.0;
                double sum_gttdivJR_r = 0.0;
                double sum_JdivR      = 0.0;
                double sum_grtdivJR_t = 0.0;

                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_r_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_t_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rr_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_tt_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> sin_tb_values;
                alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> cos_tb_values;

                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    double tb_ij    = BenchGrid::theta[j] + c0_i;
                    double tb_r_ij  = c0_r_i;
                    double tb_t_ij  = 1.0;
                    double tb_rr_ij = c0_rr_i;
                    double tb_rt_ij = 0.0;
                    double tb_tt_ij = 0.0;

                    for (size_t order = 1; order < c_limit; ++order)
                    {
                        const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                        const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                        const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                        const double c_i       = c_fields(order, profile_value);
                        const double c_r_i     = c_fields(order, profile_radial);
                        const double c_rr_i    = c_fields(order, profile_radial2);

                        tb_ij += c_i * cos_kt;
                        tb_r_ij += c_r_i * cos_kt;
                        tb_t_ij -= c_i * k_sin_kt;
                        tb_rr_ij += c_rr_i * cos_kt;
                        tb_rt_ij -= c_r_i * k_sin_kt;
                        tb_tt_ij -= c_i * k2_cos_kt;
                    }

                    for (size_t order = 1; order < s_limit; ++order)
                    {
                        const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                        const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                        const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                        const double s_i       = s_fields(order, profile_value);
                        const double s_r_i     = s_fields(order, profile_radial);
                        const double s_rr_i    = s_fields(order, profile_radial2);

                        tb_ij += s_i * sin_kt;
                        tb_r_ij += s_r_i * sin_kt;
                        tb_t_ij += s_i * k_cos_kt;
                        tb_rr_ij += s_rr_i * sin_kt;
                        tb_rt_ij += s_r_i * k_cos_kt;
                        tb_tt_ij -= s_i * k2_sin_kt;
                    }

                    tb_values[j] = tb_ij;
                    tb_r_values[j] = tb_r_ij;
                    tb_t_values[j] = tb_t_ij;
                    tb_rr_values[j] = tb_rr_ij;
                    tb_rt_values[j] = tb_rt_ij;
                    tb_tt_values[j] = tb_tt_ij;
                }

#pragma clang loop vectorize(enable)
                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    const double tb_ij = tb_values[j];
                    geometry::detail::reduced_taylor_sincos(tb_ij, sin_tb_values[j], cos_tb_values[j]);
                }

                for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                {
                    const double sin_t = BenchGrid::sin_mtheta(1, j);
                    const double cos_t = BenchGrid::cos_mtheta(1, j);

                    const double tb_r_ij   = tb_r_values[j];
                    const double tb_t_ij   = tb_t_values[j];
                    const double tb_rr_ij  = tb_rr_values[j];
                    const double tb_rt_ij  = tb_rt_values[j];
                    const double tb_tt_ij  = tb_tt_values[j];
                    const double cos_tb_ij = cos_tb_values[j];
                    const double sin_tb_ij = sin_tb_values[j];

                    double R_ij = params.R0 + params.a * (h_i + rho_i * cos_tb_ij);
                    if (R_ij < 1.0e-6)
                        R_ij = 1.0e-6;

                    const double R_r_ij = params.a * (h_r_i + cos_tb_ij - rho_i * sin_tb_ij * tb_r_ij);
                    const double R_t_ij = -params.a * rho_i * sin_tb_ij * tb_t_ij;
                    const double R_rr_ij = params.a *
                        (h_rr_i - 2.0 * sin_tb_ij * tb_r_ij -
                         rho_i * (cos_tb_ij * tb_r_ij * tb_r_ij + sin_tb_ij * tb_rr_ij));
                    const double R_rt_ij = -params.a *
                        (sin_tb_ij * tb_t_ij +
                         rho_i * (cos_tb_ij * tb_r_ij * tb_t_ij + sin_tb_ij * tb_rt_ij));
                    const double R_tt_ij =
                        -params.a * rho_i * (cos_tb_ij * tb_t_ij * tb_t_ij + sin_tb_ij * tb_tt_ij);

                    const double Z_r_ij  = params.a * (v_r_i - (k_i + rho_i * k_r_i) * sin_t);
                    const double Z_t_ij  = -params.a * rho_i * k_i * cos_t;
                    const double Z_rr_ij = params.a * (v_rr_i - (2.0 * k_r_i + rho_i * k_rr_i) * sin_t);
                    const double Z_rt_ij = -params.a * (k_i + rho_i * k_r_i) * cos_t;
                    const double Z_tt_ij = params.a * rho_i * k_i * sin_t;

                    double J_ij = R_t_ij * Z_r_ij - R_r_ij * Z_t_ij;
                    if (J_ij < 1.0e-6)
                        J_ij = 1.0e-6;

                    const double J_r_ij =
                        -(R_rr_ij * Z_t_ij - R_rt_ij * Z_r_ij + R_r_ij * Z_rt_ij - R_t_ij * Z_rr_ij);
                    const double J_t_ij =
                        -(R_rt_ij * Z_t_ij - R_tt_ij * Z_r_ij + R_r_ij * Z_tt_ij - R_t_ij * Z_rt_ij);
                    const double JR_ij       = J_ij * R_ij;
                    const double JR_r_ij     = J_r_ij * R_ij + J_ij * R_r_ij;
                    const double JR_t_ij     = J_t_ij * R_ij + J_ij * R_t_ij;
                    const double JdivR_ij    = J_ij / R_ij;
                    const double grt_ij      = R_r_ij * R_t_ij + Z_r_ij * Z_t_ij;
                    const double grt_t_ij    =
                        R_rt_ij * R_t_ij + R_r_ij * R_tt_ij + Z_rt_ij * Z_t_ij + Z_r_ij * Z_tt_ij;
                    const double gtt_ij         = R_t_ij * R_t_ij + Z_t_ij * Z_t_ij;
                    const double gtt_r_ij       = 2.0 * (R_t_ij * R_rt_ij + Z_t_ij * Z_rt_ij);
                    const double inv_JR         = 1.0 / JR_ij;
                    const double grtdivJR_t_ij = (grt_t_ij - grt_ij * JR_t_ij * inv_JR) * inv_JR;
                    const double gttdivJR_ij   = gtt_ij * inv_JR;
                    const double gttdivJR_r_ij = gtt_r_ij * inv_JR - gtt_ij * JR_r_ij * inv_JR * inv_JR;
                    sum_J += J_ij;
                    sum_JR += JR_ij;
                    sum_gttdivJR += gttdivJR_ij;
                    sum_gttdivJR_r += gttdivJR_r_ij;
                    sum_JdivR += JdivR_ij;
                    sum_grtdivJR_t += grtdivJR_t_ij;
                }

                constexpr double theta_scale = 2.0 * probe_pi / static_cast<double>(BenchGrid::theta_rows);
                constexpr double mean_scale  = 1.0 / static_cast<double>(BenchGrid::theta_rows);
                op.workspace.geometry.radial_fields(radial_S_r, i) = sum_J * theta_scale;
                op.workspace.geometry.radial_fields(radial_V_r, i) = sum_JR * theta_scale * 2.0 * probe_pi;
                op.workspace.geometry.radial_fields(radial_Kn, i) = sum_gttdivJR * mean_scale;
                op.workspace.geometry.radial_fields(radial_Kn_r, i) = sum_gttdivJR_r * mean_scale;
                op.workspace.geometry.radial_fields(radial_Ln_r, i) = (sum_JdivR + sum_grtdivJR_t) * mean_scale;
            }
            else
            {
                if constexpr (Mode == GeometryProbeMode::PhaseSplitSincos)
                {
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_r_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_t_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rr_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_rt_values;
                    alignas(tensor::detail::simd_alignment) std::array<double, BenchGrid::theta_rows> tb_tt_values;

                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double tb_ij    = BenchGrid::theta[j] + c0_i;
                        double tb_r_ij  = c0_r_i;
                        double tb_t_ij  = 1.0;
                        double tb_rr_ij = c0_rr_i;
                        double tb_rt_ij = 0.0;
                        double tb_tt_ij = 0.0;

                        for (size_t order = 1; order < c_limit; ++order)
                        {
                            const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                            const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                            const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                            const double c_i       = c_fields(order, profile_value);
                            const double c_r_i     = c_fields(order, profile_radial);
                            const double c_rr_i    = c_fields(order, profile_radial2);

                            tb_ij += c_i * cos_kt;
                            tb_r_ij += c_r_i * cos_kt;
                            tb_t_ij -= c_i * k_sin_kt;
                            tb_rr_ij += c_rr_i * cos_kt;
                            tb_rt_ij -= c_r_i * k_sin_kt;
                            tb_tt_ij -= c_i * k2_cos_kt;
                        }

                        for (size_t order = 1; order < s_limit; ++order)
                        {
                            const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                            const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                            const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                            const double s_i       = s_fields(order, profile_value);
                            const double s_r_i     = s_fields(order, profile_radial);
                            const double s_rr_i    = s_fields(order, profile_radial2);

                            tb_ij += s_i * sin_kt;
                            tb_r_ij += s_r_i * sin_kt;
                            tb_t_ij += s_i * k_cos_kt;
                            tb_rr_ij += s_rr_i * sin_kt;
                            tb_rt_ij += s_r_i * k_cos_kt;
                            tb_tt_ij -= s_i * k2_sin_kt;
                        }

                        tb_values[j] = tb_ij;
                        tb_r_values[j] = tb_r_ij;
                        tb_t_values[j] = tb_t_ij;
                        tb_rr_values[j] = tb_rr_ij;
                        tb_rt_values[j] = tb_rt_ij;
                        tb_tt_values[j] = tb_tt_ij;
                    }

#pragma clang loop vectorize(enable)
                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double sin_tb = 0.0;
                        double cos_tb = 0.0;
                        geometry::detail::reduced_taylor_sincos(tb_values[j], sin_tb, cos_tb);
                        sink += sin_tb + cos_tb + tb_r_values[j] + tb_t_values[j] + tb_rr_values[j] + tb_rt_values[j] +
                                tb_tt_values[j];
                    }
                }
                else
                {
                    for (size_t j = 0; j < BenchGrid::theta_rows; ++j)
                    {
                        double tb_ij    = BenchGrid::theta[j] + c0_i;
                        double tb_r_ij  = c0_r_i;
                        double tb_t_ij  = 1.0;
                        double tb_rr_ij = c0_rr_i;
                        double tb_rt_ij = 0.0;
                        double tb_tt_ij = 0.0;

                        for (size_t order = 1; order < c_limit; ++order)
                        {
                            const double cos_kt    = BenchGrid::cos_mtheta(order, j);
                            const double k_sin_kt  = BenchGrid::m_sin_mtheta(order, j);
                            const double k2_cos_kt = BenchGrid::m2_cos_mtheta(order, j);
                            const double c_i       = c_fields(order, profile_value);
                            const double c_r_i     = c_fields(order, profile_radial);
                            const double c_rr_i    = c_fields(order, profile_radial2);

                            tb_ij += c_i * cos_kt;
                            tb_r_ij += c_r_i * cos_kt;
                            tb_t_ij -= c_i * k_sin_kt;
                            tb_rr_ij += c_rr_i * cos_kt;
                            tb_rt_ij -= c_r_i * k_sin_kt;
                            tb_tt_ij -= c_i * k2_cos_kt;
                        }

                        for (size_t order = 1; order < s_limit; ++order)
                        {
                            const double sin_kt    = BenchGrid::sin_mtheta(order, j);
                            const double k_cos_kt  = BenchGrid::m_cos_mtheta(order, j);
                            const double k2_sin_kt = BenchGrid::m2_sin_mtheta(order, j);
                            const double s_i       = s_fields(order, profile_value);
                            const double s_r_i     = s_fields(order, profile_radial);
                            const double s_rr_i    = s_fields(order, profile_radial2);

                            tb_ij += s_i * sin_kt;
                            tb_r_ij += s_r_i * sin_kt;
                            tb_t_ij += s_i * k_cos_kt;
                            tb_rr_ij += s_rr_i * sin_kt;
                            tb_rt_ij += s_r_i * k_cos_kt;
                            tb_tt_ij -= s_i * k2_sin_kt;
                        }

                        if constexpr (Mode == GeometryProbeMode::Phase)
                        {
                            sink += tb_ij + tb_r_ij + tb_t_ij + tb_rr_ij + tb_rt_ij + tb_tt_ij;
                        }
                        else
                        {
                            sink += sin(tb_ij) + cos(tb_ij) + tb_r_ij + tb_t_ij + tb_rr_ij + tb_rt_ij + tb_tt_ij;
                        }
                    }
                }
            }
        }

        if constexpr (Mode != GeometryProbeMode::MetricNoStore)
            op.workspace.geometry.radial_fields(radial_S_r, 0) = sink;
    }

    double consume_state(const BenchOperator&       op,
                         const PackedVector&        packed,
                         const SourceRadialVector&  source_scratch,
                         const ResidualMomentRows&  residual_moments) noexcept
    {
        return op.workspace.profiles.profile_field(BenchShape::psin_profile_id, 0, 0) +
               op.workspace.geometry.surface_field(surface_R, 0, 0) + op.workspace.source_runtime.alpha1 +
               op.workspace.residual.surface_field(surface_G, 0, 0) + packed[0] + source_scratch[0] +
               residual_moments(0, 0);
    }

    void run_stage_once(StageKind                                   stage,
                        BenchOperator&                              op,
                        std::span<const double, BenchShape::x_size> x,
                        PackedVector&                               packed,
                        SourceRadialVector&                         source_scratch,
                        SourceRadialVector&                         source_aux,
        ResidualMomentRows&                         residual_moments)
    {
        const auto& params = op.runtime_params();
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            op.workspace.profiles.refresh_fixed(params.profile_params);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesActive:
            op.workspace.profiles.refresh_active(x, params.profile_params);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::ProfilesAll:
            refresh_profiles(op, x);
            compiler_barrier(op.workspace.profiles.profile_fields.data());
            break;
        case StageKind::GeometryPhase:
            run_geometry_probe<GeometryProbeMode::Phase>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryPhaseSincos:
            run_geometry_probe<GeometryProbeMode::PhaseSincos>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryPhaseSplitSincos:
            run_geometry_probe<GeometryProbeMode::PhaseSplitSincos>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::GeometryMetricNoStore:
            run_geometry_probe<GeometryProbeMode::MetricNoStore>(op);
            compiler_barrier(op.workspace.geometry.radial_fields.data());
            break;
        case StageKind::Geometry:
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            compiler_barrier(op.workspace.geometry.surface_fields.data());
            break;
        case StageKind::SourceMaterialize:
            op.workspace.source_runtime.materialize_profile_owned_psin(op.workspace.profiles, op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.materialized_heat_input.data());
            break;
        case StageKind::SourceCopyRegularize:
            op.workspace.source_runtime.benchmark_copy_profile_psin_r(op.workspace.profiles);
            op.workspace.source_runtime.benchmark_regularize_psin_r(op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceDpsin:
            op.workspace.source_runtime.benchmark_D_psin_into_rr();
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceApsin:
            op.workspace.source_runtime.benchmark_A_psin_into(source_scratch);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceDApsin:
            op.workspace.source_runtime.benchmark_DA_psin_into(source_scratch, source_aux);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceDApsinBlock4:
            op.workspace.source_runtime.benchmark_DA_psin_block4_into(source_scratch, source_aux);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceInterpolatePair:
            op.workspace.source_runtime.benchmark_prepare_psin_queries();
            op.workspace.source_runtime.benchmark_interpolate_pair();
            compiler_barrier(op.workspace.source_runtime.materialized_heat_input.data());
            break;
        case StageKind::SourceIntegrand:
            op.workspace.source_runtime.benchmark_fill_pf_psin_integrand(source_scratch, op.workspace.geometry);
            compiler_barrier(source_scratch.data());
            break;
        case StageKind::SourceAIntegrand:
            op.workspace.source_runtime.benchmark_A_integrand_into(source_aux, source_scratch);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceAIntegrandRowdot:
            op.workspace.source_runtime.benchmark_A_integrand_rowdot_into(source_aux, source_scratch);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceNormalize:
            (void)op.workspace.source_runtime.benchmark_normalize_psin_r_into(
                source_aux, source_scratch, op.workspace.geometry, op.plan.n_axis_fix);
            compiler_barrier(source_aux.data());
            break;
        case StageKind::SourceDNormalized:
            op.workspace.source_runtime.benchmark_D_normalized_psin_into_rr(source_scratch);
            compiler_barrier(op.workspace.source_runtime.source_target_root_fields.data());
            break;
        case StageKind::SourceAlpha:
            op.workspace.source_runtime.benchmark_update_alpha_from_integral(op.workspace.geometry, params.Ip, 1.0);
            compiler_barrier(&op.workspace.source_runtime.alpha1);
            break;
        case StageKind::SourceUpdate:
            op.workspace.source_runtime.update_pf_psin_uniform_ip(
                op.workspace.geometry, params.Ip, op.plan.n_axis_fix);
            compiler_barrier(op.workspace.source_runtime.FFn_psin.data());
            break;
        case StageKind::ResidualUpdate:
            op.workspace.residual.update_compact(op.workspace.source_runtime, op.workspace.geometry);
            compiler_barrier(op.workspace.residual.surface_fields.data());
            break;
        case StageKind::ResidualThetaReduce:
            op.workspace.residual.benchmark_theta_reduce_into(residual_moments);
            compiler_barrier(residual_moments.data());
            break;
        case StageKind::ResidualRadialProject:
            op.workspace.residual.benchmark_radial_project_from(
                packed, residual_moments, params.a, params.R0, params.B0);
            compiler_barrier(packed.data());
            break;
        case StageKind::ResidualPack:
            op.workspace.residual.pack_into(packed, params.a, params.R0, params.B0);
            compiler_barrier(packed.data());
            break;
        case StageKind::Evaluate:
            op.evaluate(x, packed);
            compiler_barrier(packed.data());
            break;
        case StageKind::EvaluateRing:
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
        case StageKind::GeometryPhase:
            return "geometry_phase";
        case StageKind::GeometryPhaseSincos:
            return "geometry_phase_sincos";
        case StageKind::GeometryPhaseSplitSincos:
            return "geometry_phase_split_sincos";
        case StageKind::GeometryMetricNoStore:
            return "geometry_metric_no_store";
        case StageKind::Geometry:
            return "geometry";
        case StageKind::SourceMaterialize:
            return "source_materialize";
        case StageKind::SourceCopyRegularize:
            return "source_copy_regularize";
        case StageKind::SourceDpsin:
            return "source_D_psin";
        case StageKind::SourceApsin:
            return "source_A_psin";
        case StageKind::SourceDApsin:
            return "source_DA_psin";
        case StageKind::SourceDApsinBlock4:
            return "source_DA_psin_block4";
        case StageKind::SourceInterpolatePair:
            return "source_interpolate_pair";
        case StageKind::SourceIntegrand:
            return "source_integrand";
        case StageKind::SourceAIntegrand:
            return "source_A_integrand";
        case StageKind::SourceAIntegrandRowdot:
            return "source_A_integrand_rowdot";
        case StageKind::SourceNormalize:
            return "source_normalize";
        case StageKind::SourceDNormalized:
            return "source_D_normalized";
        case StageKind::SourceAlpha:
            return "source_alpha";
        case StageKind::SourceUpdate:
            return "source_update";
        case StageKind::ResidualUpdate:
            return "residual_update";
        case StageKind::ResidualThetaReduce:
            return "residual_theta_reduce";
        case StageKind::ResidualRadialProject:
            return "residual_radial_project";
        case StageKind::ResidualPack:
            return "residual_pack";
        case StageKind::Evaluate:
            return "evaluate";
        case StageKind::EvaluateRing:
            return "evaluate_ring";
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
        if (value == "geometry_phase")
            return StageKind::GeometryPhase;
        if (value == "geometry_phase_sincos")
            return StageKind::GeometryPhaseSincos;
        if (value == "geometry_phase_split_sincos")
            return StageKind::GeometryPhaseSplitSincos;
        if (value == "geometry_metric_no_store")
            return StageKind::GeometryMetricNoStore;
        if (value == "geometry")
            return StageKind::Geometry;
        if (value == "source_materialize")
            return StageKind::SourceMaterialize;
        if (value == "source_copy_regularize")
            return StageKind::SourceCopyRegularize;
        if (value == "source_D_psin")
            return StageKind::SourceDpsin;
        if (value == "source_A_psin")
            return StageKind::SourceApsin;
        if (value == "source_DA_psin")
            return StageKind::SourceDApsin;
        if (value == "source_DA_psin_block4")
            return StageKind::SourceDApsinBlock4;
        if (value == "source_interpolate_pair")
            return StageKind::SourceInterpolatePair;
        if (value == "source_integrand")
            return StageKind::SourceIntegrand;
        if (value == "source_A_integrand")
            return StageKind::SourceAIntegrand;
        if (value == "source_A_integrand_rowdot")
            return StageKind::SourceAIntegrandRowdot;
        if (value == "source_normalize")
            return StageKind::SourceNormalize;
        if (value == "source_D_normalized")
            return StageKind::SourceDNormalized;
        if (value == "source_alpha")
            return StageKind::SourceAlpha;
        if (value == "source_update")
            return StageKind::SourceUpdate;
        if (value == "residual_update")
            return StageKind::ResidualUpdate;
        if (value == "residual_theta_reduce")
            return StageKind::ResidualThetaReduce;
        if (value == "residual_radial_project")
            return StageKind::ResidualRadialProject;
        if (value == "residual_pack")
            return StageKind::ResidualPack;
        if (value == "evaluate")
            return StageKind::Evaluate;
        if (value == "evaluate_ring")
            return StageKind::EvaluateRing;
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
            StageKind::GeometryPhase,
            StageKind::GeometryPhaseSincos,
            StageKind::GeometryPhaseSplitSincos,
            StageKind::GeometryMetricNoStore,
            StageKind::Geometry,
            StageKind::SourceMaterialize,
            StageKind::SourceCopyRegularize,
            StageKind::SourceDpsin,
            StageKind::SourceApsin,
            StageKind::SourceDApsin,
            StageKind::SourceDApsinBlock4,
            StageKind::SourceInterpolatePair,
            StageKind::SourceIntegrand,
            StageKind::SourceAIntegrandRowdot,
            StageKind::SourceAIntegrand,
            StageKind::SourceNormalize,
            StageKind::SourceDNormalized,
            StageKind::SourceAlpha,
            StageKind::SourceUpdate,
            StageKind::ResidualUpdate,
            StageKind::ResidualThetaReduce,
            StageKind::ResidualRadialProject,
            StageKind::ResidualPack,
            StageKind::Evaluate,
            StageKind::EvaluateRing,
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
                std::cout << "usage: veqlib_main --mode stage [--stage all|profiles_fixed|profiles_active|"
                             "profiles_all|geometry_phase|geometry_phase_sincos|geometry_phase_split_sincos|"
                             "geometry_metric_no_store|geometry|source_materialize|source_copy_regularize|"
                             "source_D_psin|source_A_psin|source_DA_psin|source_DA_psin_block4|"
                             "source_interpolate_pair|source_integrand|"
                             "source_A_integrand|source_A_integrand_rowdot|source_normalize|source_D_normalized|source_alpha|"
                             "source_update|residual_update|residual_theta_reduce|residual_radial_project|residual_pack|"
                             "evaluate|evaluate_ring] [--repeat N] [--warmup N] "
                             "[--inner N] [--ring-size N]\n";
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
            else if (arg == "--ring-size")
                options.ring_size = parse_size_arg(arg, value, false);
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

    void prepare_for_stage(StageKind                                   stage,
                           BenchOperator&                              op,
                           std::span<const double, BenchShape::x_size> x,
                           SourceRadialVector&                         source_scratch,
                           SourceRadialVector&                         source_aux,
                           ResidualMomentRows&                         residual_moments)
    {
        const auto& params = op.runtime_params();
        switch (stage)
        {
        case StageKind::ProfilesFixed:
            break;
        case StageKind::ProfilesActive:
            op.refresh_static_plan();
            break;
        case StageKind::ProfilesAll:
            break;
        case StageKind::GeometryPhase:
        case StageKind::GeometryPhaseSincos:
        case StageKind::GeometryPhaseSplitSincos:
        case StageKind::GeometryMetricNoStore:
        case StageKind::Geometry:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceMaterialize:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceCopyRegularize:
            refresh_profiles(op, x);
            break;
        case StageKind::SourceDpsin:
        case StageKind::SourceApsin:
        case StageKind::SourceDApsin:
        case StageKind::SourceDApsinBlock4:
            prepare_source_profile_root(op, x);
            break;
        case StageKind::SourceInterpolatePair:
            prepare_source_materialized(op, x);
            break;
        case StageKind::SourceIntegrand:
            prepare_source_materialized(op, x);
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            break;
        case StageKind::SourceAIntegrand:
        case StageKind::SourceAIntegrandRowdot:
            prepare_source_integrand(op, x, source_scratch);
            break;
        case StageKind::SourceNormalize:
            prepare_source_accumulated_integrand(op, x, source_scratch);
            break;
        case StageKind::SourceDNormalized:
            prepare_source_normalized_psin(op, x, source_scratch);
            break;
        case StageKind::SourceAlpha:
            prepare_source_updated(op, x);
            break;
        case StageKind::SourceUpdate:
            prepare_source_materialized(op, x);
            op.workspace.geometry.update(params.a, params.R0, params.Z0, op.workspace.profiles);
            break;
        case StageKind::ResidualUpdate:
            prepare_source_updated(op, x);
            break;
        case StageKind::ResidualThetaReduce:
            prepare_residual_updated(op, x);
            break;
        case StageKind::ResidualRadialProject:
            prepare_residual_updated(op, x);
            op.workspace.residual.benchmark_theta_reduce_into(residual_moments);
            break;
        case StageKind::ResidualPack:
            prepare_residual_updated(op, x);
            break;
        case StageKind::Evaluate:
            break;
        case StageKind::EvaluateRing:
            break;
        }
    }

    std::vector<std::array<double, BenchShape::x_size>> make_state_ring(size_t ring_size)
    {
        std::vector<std::array<double, BenchShape::x_size>> states(ring_size);
        for (size_t state = 0; state < ring_size; ++state)
        {
            const double phase = static_cast<double>(state + 1) / static_cast<double>(ring_size);
            for (size_t i = 0; i < BenchShape::x_size; ++i)
            {
                const double mode = static_cast<double>((i % 5) + 1);
                states[state][i] = 0.02 * std::sin(mode * phase) + 0.01 * std::cos(mode + phase);
            }
        }
        return states;
    }

    nlohmann::json run_benchmark(StageKind stage, const Options& options)
    {
        auto op = std::make_unique<BenchOperator>();
        configure_operator(*op);
        std::array<double, BenchShape::x_size> x{};
        auto                                  state_ring = make_state_ring(options.ring_size);
        PackedVector                           packed{};
        SourceRadialVector                     source_scratch{};
        SourceRadialVector                     source_aux{};
        ResidualMomentRows                     residual_moments{};
        const auto                             x_values = x_span(x);

        prepare_for_stage(stage, *op, x_values, source_scratch, source_aux, residual_moments);

        for (size_t sample = 0; sample < options.warmup; ++sample)
            for (size_t i = 0; i < options.inner; ++i)
            {
                const auto input = stage == StageKind::EvaluateRing
                    ? x_span(state_ring[(sample * options.inner + i) % state_ring.size()])
                    : x_values;
                run_stage_once(stage, *op, input, packed, source_scratch, source_aux, residual_moments);
            }

        std::vector<double> samples;
        samples.reserve(options.repeat);
        using clock = std::chrono::steady_clock;
        for (size_t sample = 0; sample < options.repeat; ++sample)
        {
            const auto start = clock::now();
            for (size_t i = 0; i < options.inner; ++i)
            {
                const auto input = stage == StageKind::EvaluateRing
                    ? x_span(state_ring[(sample * options.inner + i) % state_ring.size()])
                    : x_values;
                run_stage_once(stage, *op, input, packed, source_scratch, source_aux, residual_moments);
            }
            const auto                                     stop    = clock::now();
            const std::chrono::duration<double, std::nano> elapsed = stop - start;
            samples.push_back(elapsed.count() / static_cast<double>(options.inner));
        }

        benchmark_sink += consume_state(*op, packed, source_scratch, residual_moments);
        const Stats stats = compute_stats(samples);

        return {
            {"stage", stage_name(stage)},
            {"repeat", options.repeat},
            {"warmup", options.warmup},
            {"inner", options.inner},
            {"ring_size", stage == StageKind::EvaluateRing ? options.ring_size : size_t{1}},
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

int run(int argc, char** argv)
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
        std::cerr << "veqlib_main --mode stage: " << error.what() << '\n';
        return 2;
    }
}

} // namespace veqlib_stage_benchmark_cli
