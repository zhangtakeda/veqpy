cmake_minimum_required(VERSION 3.24)

project(veqlib LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_ENZYME "Load the Enzyme clang plugin and probe autodiff" ON)
option(VEQLIB_ENABLE_STRICT_CONVERSIONS "Reject implicit narrowing and signedness conversions with clang warnings" ON)
option(VEQLIB_ENABLE_NATIVE_OPTIMIZATIONS "Enable native CPU, fast-math, and loop optimization flags" ON)
option(VEQLIB_ENABLE_THIN_LTO "Enable clang ThinLTO and lld for optimized builds" ON)
option(VEQLIB_FORCE_OPENBLAS_SINGLE_THREAD "Force OpenBLAS to one runtime thread inside veqlib targets" ON)
set(VEQLIB_ENZYME_PLUGIN "" CACHE FILEPATH "Path to ClangEnzyme shared library")
set(VEQLIB_GCEM_ROOT "$ENV{HOME}/opt/gcem-install" CACHE PATH "GCEM install prefix")
set(VEQLIB_CONSTEXPR_STEP_LIMIT "67108864")
set(VEQLIB_CONSTEXPR_STEPS "${VEQLIB_CONSTEXPR_STEP_LIMIT}" CACHE STRING "Clang constexpr evaluation step budget")

if(VEQLIB_CONSTEXPR_STEPS GREATER VEQLIB_CONSTEXPR_STEP_LIMIT)
    message(FATAL_ERROR "VEQLIB_CONSTEXPR_STEPS must not exceed ${VEQLIB_CONSTEXPR_STEP_LIMIT}")
endif()

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
    target_compile_options(
        "${target}"
        PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-O3>
    )

    if(VEQLIB_ENABLE_NATIVE_OPTIMIZATIONS)
        target_compile_options(
            "${target}"
            PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-march=native>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-mtune=native>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-mprefer-vector-width=256>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-ffast-math>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-ffp-contract=fast>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-funsafe-math-optimizations>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fno-math-errno>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fno-trapping-math>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fno-signed-zeros>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-freciprocal-math>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-ffinite-math-only>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fapprox-func>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fstrict-aliasing>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fomit-frame-pointer>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-funroll-loops>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fvectorize>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fslp-vectorize>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-ffunction-sections>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fdata-sections>
        )
        target_link_options(
            "${target}"
            PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-Wl,-O3>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-Wl,--gc-sections>
        )
    endif()

    if(VEQLIB_ENABLE_THIN_LTO)
        target_compile_options(
            "${target}"
            PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-flto=thin>
        )
        target_link_options(
            "${target}"
            PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-flto=thin>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-fuse-ld=lld>
        )
    endif()
endfunction()

veqlib_require_clang()

set(VEQLIB_COMMON_SOURCES linalg.cpp)

function(veqlib_configure_executable target)
    target_include_directories("${target}" PRIVATE "${PROJECT_SOURCE_DIR}")
    set_target_properties(
        "${target}"
        PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
    veqlib_enable_strict_conversions("${target}")
    veqlib_enable_constexpr_budget("${target}")
    veqlib_enable_kernel_optimizations("${target}")
    veqlib_link_dependencies("${target}")
endfunction()

add_executable(veqlib_probe main.cpp ${VEQLIB_COMMON_SOURCES})
veqlib_configure_executable(veqlib_probe)
veqlib_enable_enzyme(veqlib_probe)

add_executable(veqlib_temp_validation temp_validation.cpp ${VEQLIB_COMMON_SOURCES})
veqlib_configure_executable(veqlib_temp_validation)

enable_testing()
add_test(NAME veqlib_probe COMMAND veqlib_probe)
add_test(NAME veqlib_temp_validation COMMAND veqlib_temp_validation)
