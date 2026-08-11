#pragma once

// Shared Cxx Kernel ABI integer codes.
//
// Keep these values stable: Python Kernel bindings, CMake topology defines, and
// the nanobind/C++ runtime ABI all communicate through these integer codes.

namespace cxx_abi
{
    inline constexpr int solver_method_powell             = 1;
    inline constexpr int solver_method_levenberg_marquardt = 2;
    inline constexpr int solver_method_newton_krylov       = 4;
    inline constexpr int solver_method_newton_raphson      = 5;

    inline constexpr int initial_policy_cold_zeros     = 1;
    inline constexpr int initial_policy_cold_geometric = 2;
    inline constexpr int initial_policy_cold           = 3;

    inline constexpr int continue_policy_cold_zeros     = 1;
    inline constexpr int continue_policy_cold_geometric = 2;
    inline constexpr int continue_policy_cold           = 3;
    inline constexpr int continue_policy_warm_fixed     = 4;
    inline constexpr int continue_policy_warm_predict   = 5;
    inline constexpr int continue_policy_warm_chord     = 6;
    inline constexpr int continue_policy_warm           = 7;

    inline constexpr int residual_normalization_none     = 0;
    inline constexpr int residual_normalization_fast     = 1;
    inline constexpr int residual_normalization_balanced = 2;
    inline constexpr int residual_normalization_safe     = 3;

    inline constexpr int source_route_pf  = 1;
    inline constexpr int source_route_pp  = 2;
    inline constexpr int source_route_pi  = 3;
    inline constexpr int source_route_pj1 = 4;
    inline constexpr int source_route_pj2 = 5;
    inline constexpr int source_route_pq  = 6;
    inline constexpr int source_route_pj3 = 7;

    inline constexpr int source_coordinate_rho  = 1;
    inline constexpr int source_coordinate_psin = 2;

    inline constexpr int source_constraint_null    = 0;
    inline constexpr int source_constraint_ip      = 1;
    inline constexpr int source_constraint_beta    = 2;
    inline constexpr int source_constraint_ip_beta = 3;

    inline constexpr int source_nodes_uniform = 1;
    inline constexpr int source_nodes_grid    = 2;

    inline constexpr int source_active_none = 0;
    inline constexpr int source_active_psin = 1;
    inline constexpr int source_active_F    = 2;

    inline constexpr int source_parameterization_identity  = 0;
    inline constexpr int source_parameterization_sqrt_psin = 1;
}

namespace cxx_kernel_api
{
    inline constexpr int SolverMethodPowell             = cxx_abi::solver_method_powell;
    inline constexpr int SolverMethodLevenbergMarquardt = cxx_abi::solver_method_levenberg_marquardt;
    inline constexpr int SolverMethodNewtonKrylov       = cxx_abi::solver_method_newton_krylov;
    inline constexpr int SolverMethodNewtonRaphson      = cxx_abi::solver_method_newton_raphson;

    inline constexpr int InitialPolicyColdZeros     = cxx_abi::initial_policy_cold_zeros;
    inline constexpr int InitialPolicyColdGeometric = cxx_abi::initial_policy_cold_geometric;
    inline constexpr int InitialPolicyCold          = cxx_abi::initial_policy_cold;

    inline constexpr int ContinuePolicyColdZeros     = cxx_abi::continue_policy_cold_zeros;
    inline constexpr int ContinuePolicyColdGeometric = cxx_abi::continue_policy_cold_geometric;
    inline constexpr int ContinuePolicyCold          = cxx_abi::continue_policy_cold;
    inline constexpr int ContinuePolicyWarmFixed     = cxx_abi::continue_policy_warm_fixed;
    inline constexpr int ContinuePolicyWarmPredict   = cxx_abi::continue_policy_warm_predict;
    inline constexpr int ContinuePolicyWarmChord     = cxx_abi::continue_policy_warm_chord;
    inline constexpr int ContinuePolicyWarm          = cxx_abi::continue_policy_warm;

    inline constexpr int ResidualNormalizationNone     = cxx_abi::residual_normalization_none;
    inline constexpr int ResidualNormalizationFast     = cxx_abi::residual_normalization_fast;
    inline constexpr int ResidualNormalizationBalanced = cxx_abi::residual_normalization_balanced;
    inline constexpr int ResidualNormalizationSafe     = cxx_abi::residual_normalization_safe;
}

namespace operators::detail
{
    inline constexpr int source_route_pf  = cxx_abi::source_route_pf;
    inline constexpr int source_route_pp  = cxx_abi::source_route_pp;
    inline constexpr int source_route_pi  = cxx_abi::source_route_pi;
    inline constexpr int source_route_pj1 = cxx_abi::source_route_pj1;
    inline constexpr int source_route_pj2 = cxx_abi::source_route_pj2;
    inline constexpr int source_route_pq  = cxx_abi::source_route_pq;
    inline constexpr int source_route_pj3 = cxx_abi::source_route_pj3;

    inline constexpr int source_coordinate_rho  = cxx_abi::source_coordinate_rho;
    inline constexpr int source_coordinate_psin = cxx_abi::source_coordinate_psin;

    inline constexpr int source_constraint_null    = cxx_abi::source_constraint_null;
    inline constexpr int source_constraint_ip      = cxx_abi::source_constraint_ip;
    inline constexpr int source_constraint_beta    = cxx_abi::source_constraint_beta;
    inline constexpr int source_constraint_ip_beta = cxx_abi::source_constraint_ip_beta;

    inline constexpr int source_nodes_uniform = cxx_abi::source_nodes_uniform;
    inline constexpr int source_nodes_grid    = cxx_abi::source_nodes_grid;

    inline constexpr int source_active_none = cxx_abi::source_active_none;
    inline constexpr int source_active_psin = cxx_abi::source_active_psin;
    inline constexpr int source_active_F    = cxx_abi::source_active_F;

    inline constexpr int source_parameterization_identity  = cxx_abi::source_parameterization_identity;
    inline constexpr int source_parameterization_sqrt_psin = cxx_abi::source_parameterization_sqrt_psin;
}
