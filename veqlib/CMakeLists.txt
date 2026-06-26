cmake_minimum_required(VERSION 3.24)

project(veqlib LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_ENZYME "Load the Enzyme clang plugin and probe autodiff" ON)
option(VEQLIB_ENABLE_STRICT_CONVERSIONS "Reject implicit narrowing and signedness conversions with clang warnings" ON)
option(VEQLIB_ENABLE_NATIVE_OPTIMIZATIONS "Enable native CPU and loop optimization flags" ON)
option(VEQLIB_ENABLE_THIN_LTO "Enable clang ThinLTO and lld for optimized builds" ON)
option(VEQLIB_ANALYSIS_BUILD "Build with source-correlated optimization diagnostics" OFF)
option(VEQLIB_FORCE_OPENBLAS_SINGLE_THREAD "Force OpenBLAS to one runtime thread inside veqlib targets" ON)
option(VEQLIB_ENABLE_PYTHON_BINDINGS "Build the nanobind Python extension for VEQlib" ON)
set(VEQLIB_FP_MODE "RELAXED" CACHE STRING "Floating-point mode: STRICT, FMA, or RELAXED")
set_property(CACHE VEQLIB_FP_MODE PROPERTY STRINGS STRICT FMA RELAXED)
set(VEQLIB_ENZYME_PLUGIN "" CACHE FILEPATH "Path to ClangEnzyme shared library")
set(VEQLIB_ENZYME_INCLUDE_DIR "$ENV{HOME}/opt/Enzyme/enzyme/include" CACHE PATH "Path to Enzyme C++ headers")
set(VEQLIB_GCEM_ROOT "$ENV{HOME}/opt/gcem-install" CACHE PATH "GCEM install prefix")
set(VEQLIB_NB_DOMAIN "veqlib_default" CACHE STRING "nanobind type registry domain for this extension")
set(
    VEQLIB_PREBUILT_NANOBIND_STATIC
    ""
    CACHE FILEPATH
    "Optional prebuilt libnanobind-static.a reused across topology-specific kernel builds"
)
set(VEQLIB_CONSTEXPR_STEP_LIMIT "67108864")
set(VEQLIB_CONSTEXPR_STEPS "${VEQLIB_CONSTEXPR_STEP_LIMIT}" CACHE STRING "Clang constexpr evaluation step budget")
set(VEQ_NR "8" CACHE STRING "Default topology radial grid node count")
set(VEQ_NT "16" CACHE STRING "Default topology poloidal grid node count")
set(VEQ_SOURCE_SAMPLE_COUNT "51" CACHE STRING "Default topology source sample count")
set(VEQ_H_PROFILE_COUNT "10" CACHE STRING "h profile coefficient count; 0 disables the profile")
set(VEQ_V_PROFILE_COUNT "10" CACHE STRING "v profile coefficient count; 0 disables the profile")
set(VEQ_KAPPA_PROFILE_COUNT "10" CACHE STRING "kappa profile coefficient count; 0 disables the profile")
set(VEQ_PSIN_PROFILE_COUNT "10" CACHE STRING "psin profile coefficient count; 0 disables the profile")
set(VEQ_F_PROFILE_COUNT "10" CACHE STRING "F profile coefficient count; 0 disables the profile")
set(
    VEQ_COS_PROFILE_COUNTS
    "10;10;10;10;10;10;10;10;10;10"
    CACHE STRING
    "Cosine profile coefficient counts: c0;c1;..., with 0 disabling a profile"
)
set(
    VEQ_SIN_PROFILE_COUNTS
    "10;10;10;10;10;10;10;10;10;10"
    CACHE STRING
    "Sine profile coefficient counts: s1;s2;..., with 0 disabling a profile"
)
set(
    VEQ_PROFILE_KMAX_LIMIT
    "1024"
    CACHE STRING
    "Upper bound for K_max; config uses min(M_max, this value)"
)
set(
    VEQ_BOUNDARY_M_MAX
    "AUTO"
    CACHE STRING
    "Boundary/geometric Fourier M_max; AUTO derives it from active c/s profile counts"
)

string(TOUPPER "${VEQLIB_FP_MODE}" VEQLIB_FP_MODE_NORMALIZED)
if(NOT VEQLIB_FP_MODE_NORMALIZED MATCHES "^(STRICT|FMA|RELAXED)$")
    message(FATAL_ERROR "VEQLIB_FP_MODE must be STRICT, FMA, or RELAXED; got '${VEQLIB_FP_MODE}'")
endif()
set(VEQLIB_FP_MODE "${VEQLIB_FP_MODE_NORMALIZED}" CACHE STRING "Floating-point mode: STRICT, FMA, or RELAXED" FORCE)

if(VEQLIB_CONSTEXPR_STEPS GREATER VEQLIB_CONSTEXPR_STEP_LIMIT)
    message(FATAL_ERROR "VEQLIB_CONSTEXPR_STEPS must not exceed ${VEQLIB_CONSTEXPR_STEP_LIMIT}")
endif()
if(NOT VEQLIB_NB_DOMAIN MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "VEQLIB_NB_DOMAIN must be a valid C/C++ identifier, got '${VEQLIB_NB_DOMAIN}'")
endif()

function(veqlib_require_nonnegative_int_var source_var)
    if(NOT "${${source_var}}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${source_var} must be a non-negative integer, got '${${source_var}}'")
    endif()
endfunction()

function(veqlib_require_int_at_least source_var minimum)
    veqlib_require_nonnegative_int_var("${source_var}")
    if("${${source_var}}" LESS "${minimum}")
        message(FATAL_ERROR "${source_var} must be at least ${minimum}, got '${${source_var}}'")
    endif()
endfunction()

function(veqlib_prepare_nonnegative_int_list
    source_var
    out_values
    out_size
    out_max
    out_highest_order
    first_order
)
    string(REGEX REPLACE "[ \t\r\n]+" "" normalized "${${source_var}}")
    string(REPLACE "," ";" normalized "${normalized}")
    if("${normalized}" STREQUAL "")
        set(values)
    else()
        set(values ${normalized})
    endif()

    set(max_count 0)
    set(highest_order -1)
    set(last_nonzero_index -1)
    set(index 0)
    foreach(value IN LISTS values)
        if(NOT "${value}" MATCHES "^[0-9]+$")
            message(FATAL_ERROR "${source_var} must contain non-negative integers, got '${value}'")
        endif()
        if("${value}" GREATER "${max_count}")
            set(max_count "${value}")
        endif()
        if(NOT "${value}" EQUAL 0)
            math(EXPR highest_order "${index} + ${first_order}")
            set(last_nonzero_index "${index}")
        endif()
        math(EXPR index "${index} + 1")
    endforeach()

    if(last_nonzero_index LESS 0)
        set(trimmed_values)
    else()
        math(EXPR retained_count "${last_nonzero_index} + 1")
        list(SUBLIST values 0 "${retained_count}" trimmed_values)
    endif()

    string(REPLACE ";" ", " header_values "${trimmed_values}")
    list(LENGTH trimmed_values value_count)

    set("${out_values}" "${header_values}" PARENT_SCOPE)
    set("${out_size}" "${value_count}" PARENT_SCOPE)
    set("${out_max}" "${max_count}" PARENT_SCOPE)
    set("${out_highest_order}" "${highest_order}" PARENT_SCOPE)
endfunction()

veqlib_require_int_at_least(VEQ_NR 4)
veqlib_require_int_at_least(VEQ_NT 4)
veqlib_require_int_at_least(VEQ_SOURCE_SAMPLE_COUNT 1)
veqlib_require_nonnegative_int_var(VEQ_H_PROFILE_COUNT)
veqlib_require_nonnegative_int_var(VEQ_V_PROFILE_COUNT)
veqlib_require_nonnegative_int_var(VEQ_KAPPA_PROFILE_COUNT)
veqlib_require_nonnegative_int_var(VEQ_PSIN_PROFILE_COUNT)
veqlib_require_nonnegative_int_var(VEQ_F_PROFILE_COUNT)
veqlib_require_int_at_least(VEQ_PROFILE_KMAX_LIMIT 2)
if(NOT VEQ_BOUNDARY_M_MAX STREQUAL "AUTO")
    veqlib_require_int_at_least(VEQ_BOUNDARY_M_MAX 1)
endif()

veqlib_prepare_nonnegative_int_list(
    VEQ_COS_PROFILE_COUNTS
    VEQ_COS_PROFILE_COUNTS_VALUES
    VEQ_COS_PROFILE_COUNTS_SIZE
    VEQ_COS_PROFILE_COUNT_MAX
    VEQ_C_MAX
    0
)
veqlib_prepare_nonnegative_int_list(
    VEQ_SIN_PROFILE_COUNTS
    VEQ_SIN_PROFILE_COUNTS_VALUES
    VEQ_SIN_PROFILE_COUNTS_SIZE
    VEQ_SIN_PROFILE_COUNT_MAX
    VEQ_S_MAX
    1
)

set(VEQ_CORE_PROFILE_COUNT_MAX "${VEQ_H_PROFILE_COUNT}")
foreach(profile_count IN ITEMS
    "${VEQ_V_PROFILE_COUNT}"
    "${VEQ_KAPPA_PROFILE_COUNT}"
    "${VEQ_PSIN_PROFILE_COUNT}"
    "${VEQ_F_PROFILE_COUNT}"
)
    if("${profile_count}" GREATER "${VEQ_CORE_PROFILE_COUNT_MAX}")
        set(VEQ_CORE_PROFILE_COUNT_MAX "${profile_count}")
    endif()
endforeach()

set(VEQ_PROFILE_COUNT_MAX "${VEQ_CORE_PROFILE_COUNT_MAX}")
foreach(profile_count IN ITEMS
    "${VEQ_COS_PROFILE_COUNT_MAX}"
    "${VEQ_SIN_PROFILE_COUNT_MAX}"
)
    if("${profile_count}" GREATER "${VEQ_PROFILE_COUNT_MAX}")
        set(VEQ_PROFILE_COUNT_MAX "${profile_count}")
    endif()
endforeach()

math(EXPR VEQ_L_MAX "${VEQ_PROFILE_COUNT_MAX} - 1")
if(VEQ_L_MAX LESS 1)
    set(VEQ_L_MAX 1)
endif()

if(VEQ_C_MAX LESS 0)
    set(VEQ_C_MAX 0)
endif()
if(VEQ_S_MAX LESS 0)
    set(VEQ_S_MAX 0)
endif()

set(VEQ_ACTIVE_M_MAX "${VEQ_C_MAX}")
if(VEQ_S_MAX GREATER VEQ_ACTIVE_M_MAX)
    set(VEQ_ACTIVE_M_MAX "${VEQ_S_MAX}")
endif()
if(VEQ_ACTIVE_M_MAX LESS 1)
    message(FATAL_ERROR "Derived active M_max must be at least 1; enable at least one positive c/s order")
endif()

if(VEQ_BOUNDARY_M_MAX STREQUAL "AUTO")
    set(VEQ_M_MAX "${VEQ_ACTIVE_M_MAX}")
else()
    set(VEQ_M_MAX "${VEQ_BOUNDARY_M_MAX}")
    if(VEQ_M_MAX LESS VEQ_ACTIVE_M_MAX)
        message(
            FATAL_ERROR
            "VEQ_BOUNDARY_M_MAX (${VEQ_M_MAX}) must be >= active profile M (${VEQ_ACTIVE_M_MAX})"
        )
    endif()
endif()

set(VEQ_RHO_ORDER_BASE "${VEQ_M_MAX}")
if(VEQ_RHO_ORDER_BASE LESS 2)
    set(VEQ_RHO_ORDER_BASE 2)
endif()
set(VEQ_K_MAX "${VEQ_RHO_ORDER_BASE}")
if(VEQ_PROFILE_KMAX_LIMIT LESS VEQ_K_MAX)
    set(VEQ_K_MAX "${VEQ_PROFILE_KMAX_LIMIT}")
endif()
if(VEQ_K_MAX LESS 2)
    message(FATAL_ERROR "Derived K_max must be at least 2")
endif()

set(VEQLIB_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VEQLIB_GENERATED_INCLUDE_DIR}")
configure_file(
    "${PROJECT_SOURCE_DIR}/config.h.in"
    "${VEQLIB_GENERATED_INCLUDE_DIR}/config.h"
    @ONLY
)

function(veqlib_require_clang)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "veqlib requires clang++; configure with -DCMAKE_CXX_COMPILER=clang++")
    endif()
endfunction()

function(veqlib_link_dependencies target)
    find_package(nlohmann_json CONFIG REQUIRED)
    target_link_libraries("${target}" PRIVATE nlohmann_json::nlohmann_json)

    if(VEQLIB_GCEM_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${VEQLIB_GCEM_ROOT}")
    endif()
    find_package(gcem CONFIG REQUIRED)
    target_link_libraries("${target}" PRIVATE gcem)

    find_path(CMINPACK_INCLUDE_DIR NAMES cminpack.h PATH_SUFFIXES cminpack-1 REQUIRED)
    find_library(CMINPACK_LIBRARY NAMES cminpack REQUIRED)
    target_include_directories("${target}" PRIVATE "${CMINPACK_INCLUDE_DIR}")
    target_link_libraries("${target}" PRIVATE "${CMINPACK_LIBRARY}")

    find_path(LAPACKE_INCLUDE_DIR NAMES lapacke.h REQUIRED)
    find_library(LAPACKE_LIBRARY NAMES lapacke REQUIRED)
    find_library(LAPACK_LIBRARY NAMES lapack REQUIRED)
    find_library(BLAS_LIBRARY NAMES openblas REQUIRED)
    target_include_directories("${target}" PRIVATE "${LAPACKE_INCLUDE_DIR}")
    target_link_libraries(
        "${target}"
        PRIVATE
            "${LAPACKE_LIBRARY}"
            "${LAPACK_LIBRARY}"
            "${BLAS_LIBRARY}"
    )

    if(VEQLIB_FORCE_OPENBLAS_SINGLE_THREAD)
        target_compile_definitions("${target}" PRIVATE VEQLIB_FORCE_OPENBLAS_SINGLE_THREAD=1)
    endif()
endfunction()

function(veqlib_find_enzyme out_var)
    if(VEQLIB_ENZYME_PLUGIN AND EXISTS "${VEQLIB_ENZYME_PLUGIN}")
        set("${out_var}" "${VEQLIB_ENZYME_PLUGIN}" PARENT_SCOPE)
        return()
    endif()

    if(DEFINED ENV{ENZYME_PLUGIN} AND EXISTS "$ENV{ENZYME_PLUGIN}")
        set("${out_var}" "$ENV{ENZYME_PLUGIN}" PARENT_SCOPE)
        return()
    endif()

    string(REGEX MATCH "^[0-9]+" clang_major "${CMAKE_CXX_COMPILER_VERSION}")
    set(enzyme_names ClangEnzyme.so)
    set(enzyme_path_suffixes lib)
    if(clang_major)
        list(PREPEND enzyme_names "ClangEnzyme-${clang_major}.so")
        list(PREPEND enzyme_path_suffixes "llvm-${clang_major}/lib")
    endif()

    find_file(
        found_enzyme_plugin
        NAMES ${enzyme_names}
        HINTS
            ENV ENZYME_PLUGIN_PATH
        PATHS
            "${PROJECT_SOURCE_DIR}/../.."
            "$ENV{HOME}/opt/Enzyme/enzyme/build-llvm18/Enzyme"
            /usr/lib
            /usr/local/lib
            /opt/homebrew/lib
        PATH_SUFFIXES ${enzyme_path_suffixes}
    )
    set("${out_var}" "${found_enzyme_plugin}" PARENT_SCOPE)
endfunction()

function(veqlib_enable_enzyme target)
    if(NOT ENABLE_ENZYME)
        return()
    endif()

    veqlib_find_enzyme(enzyme_plugin)
    if(NOT enzyme_plugin)
        message(FATAL_ERROR "ClangEnzyme plugin not found; set VEQLIB_ENZYME_PLUGIN=/path/to/ClangEnzyme-18.so")
    endif()

    get_filename_component(enzyme_plugin_name "${enzyme_plugin}" NAME)
    if(NOT enzyme_plugin_name MATCHES "^ClangEnzyme")
        message(FATAL_ERROR "veqlib's direct clang++ workflow requires ClangEnzyme, got ${enzyme_plugin}")
    endif()

    message(STATUS "Using Enzyme plugin: ${enzyme_plugin}")
    target_compile_options("${target}" PRIVATE "-fplugin=${enzyme_plugin}")
    target_compile_definitions("${target}" PRIVATE ENABLE_ENZYME=1)
    if(VEQLIB_ENZYME_INCLUDE_DIR AND EXISTS "${VEQLIB_ENZYME_INCLUDE_DIR}")
        target_include_directories("${target}" PRIVATE "${VEQLIB_ENZYME_INCLUDE_DIR}")
    endif()
endfunction()

function(veqlib_enable_strict_conversions target)
    if(NOT VEQLIB_ENABLE_STRICT_CONVERSIONS)
        return()
    endif()

    target_compile_options(
        "${target}"
        PRIVATE
            -Wconversion
            -Wsign-conversion
            -Wsign-compare
            -Wconstant-conversion
            -Wfloat-conversion
            -Wimplicit-int-conversion
            -Wimplicit-float-conversion
            -Wliteral-conversion
            -Wshorten-64-to-32
            -Werror=conversion
            -Werror=sign-conversion
            -Werror=sign-compare
            -Werror=constant-conversion
            -Werror=float-conversion
            -Werror=implicit-int-conversion
            -Werror=implicit-float-conversion
            -Werror=literal-conversion
            -Werror=shorten-64-to-32
    )
endfunction()

function(veqlib_enable_constexpr_budget target)
    target_compile_options("${target}" PRIVATE "-fconstexpr-steps=${VEQLIB_CONSTEXPR_STEPS}")
endfunction()

function(veqlib_enable_kernel_optimizations target)
    set(release_like "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")

    target_compile_options(
        "${target}"
        PRIVATE
            $<${release_like}:-O3>
    )
    target_compile_definitions("${target}" PRIVATE "VEQLIB_FP_MODE_${VEQLIB_FP_MODE}=1")

    if(VEQLIB_ENABLE_NATIVE_OPTIMIZATIONS)
        target_compile_options(
            "${target}"
            PRIVATE
                $<${release_like}:-march=native>
                $<${release_like}:-mtune=native>
                $<${release_like}:-mprefer-vector-width=256>
                $<${release_like}:-fstrict-aliasing>
                $<${release_like}:-fomit-frame-pointer>
                $<${release_like}:-funroll-loops>
                $<${release_like}:-fvectorize>
                $<${release_like}:-fslp-vectorize>
                $<${release_like}:-ffunction-sections>
                $<${release_like}:-fdata-sections>
        )
        target_link_options(
            "${target}"
            PRIVATE
                $<${release_like}:-Wl,-O3>
                $<${release_like}:-Wl,--gc-sections>
        )
    endif()

    if(VEQLIB_FP_MODE STREQUAL "STRICT")
        target_compile_options(
            "${target}"
            PRIVATE
                $<${release_like}:-fno-fast-math>
                $<${release_like}:-ffp-contract=off>
        )
    elseif(VEQLIB_FP_MODE STREQUAL "FMA")
        target_compile_options(
            "${target}"
            PRIVATE
                $<${release_like}:-fno-fast-math>
                $<${release_like}:-ffp-contract=fast>
        )
    elseif(VEQLIB_FP_MODE STREQUAL "RELAXED")
        target_compile_options(
            "${target}"
            PRIVATE
                $<${release_like}:-ffast-math>
                $<${release_like}:-ffp-contract=fast>
                $<${release_like}:-funsafe-math-optimizations>
                $<${release_like}:-fno-math-errno>
                $<${release_like}:-fno-trapping-math>
                $<${release_like}:-fno-signed-zeros>
                $<${release_like}:-freciprocal-math>
                $<${release_like}:-ffinite-math-only>
                $<${release_like}:-fapprox-func>
        )
    endif()

    if(VEQLIB_ENABLE_THIN_LTO AND NOT VEQLIB_ANALYSIS_BUILD)
        target_compile_options(
            "${target}"
            PRIVATE
                $<${release_like}:-flto=thin>
        )
        target_link_options(
            "${target}"
            PRIVATE
                $<${release_like}:-flto=thin>
                $<${release_like}:-fuse-ld=lld>
        )
    endif()

    if(VEQLIB_ANALYSIS_BUILD)
        target_compile_options(
            "${target}"
            PRIVATE
                -g
                -fno-omit-frame-pointer
                -fdebug-info-for-profiling
                -fno-unroll-loops
                "-Rpass=(loop-vectorize|slp-vectorizer)"
                "-Rpass-missed=(loop-vectorize|slp-vectorizer)"
                "-Rpass-analysis=(loop-vectorize|slp-vectorizer)"
                -fsave-optimization-record=yaml
                -Wframe-larger-than=4096
                -Wlarge-by-value-copy=256
                -fstack-usage
        )
    endif()
endfunction()

veqlib_require_clang()

set(VEQLIB_COMMON_SOURCES nonlinear.cpp linalg.cpp)

function(veqlib_configure_target target)
    target_include_directories(
        "${target}"
        PRIVATE
            "${VEQLIB_GENERATED_INCLUDE_DIR}"
            "${PROJECT_SOURCE_DIR}"
    )
    set_target_properties(
        "${target}"
        PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
    veqlib_enable_enzyme("${target}")
    veqlib_enable_strict_conversions("${target}")
    veqlib_enable_constexpr_budget("${target}")
    veqlib_enable_kernel_optimizations("${target}")
    veqlib_link_dependencies("${target}")
endfunction()

add_executable(veqlib_stage_benchmark stage_benchmark.cpp ${VEQLIB_COMMON_SOURCES})
veqlib_configure_target(veqlib_stage_benchmark)

if(VEQLIB_ENABLE_PYTHON_BINDINGS)
    if(NOT Python_EXECUTABLE AND EXISTS "${PROJECT_SOURCE_DIR}/../.venv/bin/python")
        set(Python_EXECUTABLE "${PROJECT_SOURCE_DIR}/../.venv/bin/python" CACHE FILEPATH "Python interpreter")
    endif()
    find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

    if(NOT nanobind_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
            RESULT_VARIABLE VEQLIB_NANOBIND_CMAKE_RESULT
            OUTPUT_VARIABLE VEQLIB_NANOBIND_CMAKE_DIR
            ERROR_VARIABLE VEQLIB_NANOBIND_CMAKE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT VEQLIB_NANOBIND_CMAKE_RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "nanobind is required when VEQLIB_ENABLE_PYTHON_BINDINGS=ON. "
                "Python '${Python_EXECUTABLE}' failed to report --cmake_dir: "
                "${VEQLIB_NANOBIND_CMAKE_ERROR}"
            )
        endif()
        set(nanobind_DIR "${VEQLIB_NANOBIND_CMAKE_DIR}" CACHE PATH "nanobind CMake package directory")
    endif()
    find_package(nanobind CONFIG REQUIRED)

    if(VEQLIB_PREBUILT_NANOBIND_STATIC)
        if(NOT EXISTS "${VEQLIB_PREBUILT_NANOBIND_STATIC}")
            message(
                FATAL_ERROR
                "VEQLIB_PREBUILT_NANOBIND_STATIC does not exist: ${VEQLIB_PREBUILT_NANOBIND_STATIC}"
            )
        endif()
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m nanobind --include_dir
            RESULT_VARIABLE VEQLIB_NANOBIND_INCLUDE_RESULT
            OUTPUT_VARIABLE VEQLIB_NANOBIND_INCLUDE_DIR
            ERROR_VARIABLE VEQLIB_NANOBIND_INCLUDE_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT VEQLIB_NANOBIND_INCLUDE_RESULT EQUAL 0)
            message(
                FATAL_ERROR
                "Python '${Python_EXECUTABLE}' failed to report nanobind --include_dir: "
                "${VEQLIB_NANOBIND_INCLUDE_ERROR}"
            )
        endif()
        get_filename_component(VEQLIB_NANOBIND_ROOT "${VEQLIB_NANOBIND_INCLUDE_DIR}" DIRECTORY)

        add_library(veqlib_ext MODULE bindings.cpp ${VEQLIB_COMMON_SOURCES})
        nanobind_compile_options(veqlib_ext)
        nanobind_link_options(veqlib_ext)
        set_target_properties(veqlib_ext PROPERTIES LINKER_LANGUAGE CXX)
        nanobind_extension(veqlib_ext)
        nanobind_disable_stack_protector(veqlib_ext)
        nanobind_opt_size(veqlib_ext)
        nanobind_strip(veqlib_ext)
        nanobind_set_visibility(veqlib_ext)
        target_include_directories(
            veqlib_ext
            SYSTEM
            PRIVATE
                "${Python_INCLUDE_DIRS}"
                "${VEQLIB_NANOBIND_INCLUDE_DIR}"
                "${VEQLIB_NANOBIND_ROOT}/ext/robin_map/include"
        )
        target_link_libraries(veqlib_ext PRIVATE "${VEQLIB_PREBUILT_NANOBIND_STATIC}" Python::Module)
    else()
        nanobind_add_module(veqlib_ext NB_SUPPRESS_WARNINGS bindings.cpp ${VEQLIB_COMMON_SOURCES})
    endif()
    veqlib_configure_target(veqlib_ext)
    target_compile_definitions(veqlib_ext PRIVATE "NB_DOMAIN=${VEQLIB_NB_DOMAIN}")
endif()

enable_testing()
add_test(
    NAME veqlib_stage_benchmark_smoke
    COMMAND
        veqlib_stage_benchmark
        --stage
        all
        --repeat
        1
        --warmup
        1
        --inner
        1
        --ring-size
        2
)
if(VEQLIB_ENABLE_PYTHON_BINDINGS)
    add_test(
        NAME veqlib_python_binding
        COMMAND
            "${Python_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/binding_smoke.py"
    )
    set_tests_properties(
        veqlib_python_binding
        PROPERTIES
            ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:veqlib_ext>"
    )
endif()
