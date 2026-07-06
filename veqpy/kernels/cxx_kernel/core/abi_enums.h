#pragma once

// Shared VEQlib ABI integer codes.
//
// Keep these values stable: Python facade code, CMake topology defines, and the
// nanobind/C++ runtime ABI all communicate with these integer codes.

namespace veqlib_abi
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

namespace veqlib_kernel_api
{
    inline constexpr int SolverMethodPowell             = veqlib_abi::solver_method_powell;
    inline constexpr int SolverMethodLevenbergMarquardt = veqlib_abi::solver_method_levenberg_marquardt;
    inline constexpr int SolverMethodNewtonKrylov       = veqlib_abi::solver_method_newton_krylov;
    inline constexpr int SolverMethodNewtonRaphson      = veqlib_abi::solver_method_newton_raphson;

    inline constexpr int InitialPolicyColdZeros     = veqlib_abi::initial_policy_cold_zeros;
    inline constexpr int InitialPolicyColdGeometric = veqlib_abi::initial_policy_cold_geometric;
    inline constexpr int InitialPolicyCold          = veqlib_abi::initial_policy_cold;

    inline constexpr int ContinuePolicyColdZeros     = veqlib_abi::continue_policy_cold_zeros;
    inline constexpr int ContinuePolicyColdGeometric = veqlib_abi::continue_policy_cold_geometric;
    inline constexpr int ContinuePolicyCold          = veqlib_abi::continue_policy_cold;
    inline constexpr int ContinuePolicyWarmFixed     = veqlib_abi::continue_policy_warm_fixed;
    inline constexpr int ContinuePolicyWarmPredict   = veqlib_abi::continue_policy_warm_predict;
    inline constexpr int ContinuePolicyWarmChord     = veqlib_abi::continue_policy_warm_chord;
    inline constexpr int ContinuePolicyWarm          = veqlib_abi::continue_policy_warm;

    inline constexpr int ResidualNormalizationNone     = veqlib_abi::residual_normalization_none;
    inline constexpr int ResidualNormalizationFast     = veqlib_abi::residual_normalization_fast;
    inline constexpr int ResidualNormalizationBalanced = veqlib_abi::residual_normalization_balanced;
    inline constexpr int ResidualNormalizationSafe     = veqlib_abi::residual_normalization_safe;
}

namespace operators::detail
{
    inline constexpr int source_route_pf  = veqlib_abi::source_route_pf;
    inline constexpr int source_route_pp  = veqlib_abi::source_route_pp;
    inline constexpr int source_route_pi  = veqlib_abi::source_route_pi;
    inline constexpr int source_route_pj1 = veqlib_abi::source_route_pj1;
    inline constexpr int source_route_pj2 = veqlib_abi::source_route_pj2;
    inline constexpr int source_route_pq  = veqlib_abi::source_route_pq;

    inline constexpr int source_coordinate_rho  = veqlib_abi::source_coordinate_rho;
    inline constexpr int source_coordinate_psin = veqlib_abi::source_coordinate_psin;

    inline constexpr int source_constraint_null    = veqlib_abi::source_constraint_null;
    inline constexpr int source_constraint_ip      = veqlib_abi::source_constraint_ip;
    inline constexpr int source_constraint_beta    = veqlib_abi::source_constraint_beta;
    inline constexpr int source_constraint_ip_beta = veqlib_abi::source_constraint_ip_beta;

    inline constexpr int source_nodes_uniform = veqlib_abi::source_nodes_uniform;
    inline constexpr int source_nodes_grid    = veqlib_abi::source_nodes_grid;

    inline constexpr int source_active_none = veqlib_abi::source_active_none;
    inline constexpr int source_active_psin = veqlib_abi::source_active_psin;
    inline constexpr int source_active_F    = veqlib_abi::source_active_F;

    inline constexpr int source_parameterization_identity  = veqlib_abi::source_parameterization_identity;
    inline constexpr int source_parameterization_sqrt_psin = veqlib_abi::source_parameterization_sqrt_psin;
}
