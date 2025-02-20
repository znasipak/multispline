# =============================================
# == multispline project CMake configuration ==
# =============================================

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.15...3.31)

# ---- Main project definition ----
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# ---- Add options ----
set(MULTISPLINE_MARCH native CACHE STRING
                                 "Value of the -march compiler option if supported by compiler")

# Define the splinecy target
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cython")

add_custom_command(
    OUTPUT "cython/spline_wrap.cpp"
    COMMENT
    "Using Cython to build 'spline_wrap.cpp' from 'spline_wrap.pyx'."
    COMMAND
      Python::Interpreter ARGS -m cython
      "${CMAKE_CURRENT_SOURCE_DIR}/cython/spline_wrap.pyx" --output-file
      "${CMAKE_CURRENT_BINARY_DIR}/cython/spline_wrap.cpp" -3 -+
      --module-name "splinecy" -I "${CMAKE_CURRENT_SOURCE_DIR}/cpp/include"
    DEPENDS "cython/spline_wrap.pyx"
    VERBATIM
)

python_add_library(splinecy MODULE cython/spline_wrap.cpp cpp/src/spline.cpp
                     WITH_SOABI)
target_include_directories(splinecy PRIVATE ${Python_NumPy_INCLUDE_DIR} "cpp/include")
set_property(TARGET splinecy PROPERTY CXX_STANDARD 11)

include(CheckCXXCompilerFlag)
set(MARCH_OPT "-march=${MULTISPLINE_MARCH}")
check_cxx_compiler_flag("${MARCH_OPT}" COMPILER_SUPPORTS_MULTISPLINE_MARCH)
if(COMPILER_SUPPORTS_MULTISPLINE_MARCH)
  target_compile_options(splinecy PRIVATE "${MARCH_OPT}")
else()
  message(WARNING "The CXX compiler does not support option '${MARCH_OPT}'. It will be ignored.")
endif()

install(TARGETS splinecy DESTINATION "./")
