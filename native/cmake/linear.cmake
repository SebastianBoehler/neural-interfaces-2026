# Exact native OpenBLAS dependency used by the qualified sklearn reference.
# This library contains optimized C/Fortran kernels, not a Python/SciPy callback.
if(NOT DEFINED NI26_LBFGSB_BLAS_LIBRARY OR NOT EXISTS "${NI26_LBFGSB_BLAS_LIBRARY}")
  message(FATAL_ERROR "Set NI26_LBFGSB_BLAS_LIBRARY to the existing SciPy-prefixed LP64 native OpenBLAS library")
endif()
enable_language(C)
get_filename_component(NI26_LINEAR_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(ni26_lbfgsb STATIC "${NI26_LINEAR_ROOT}/vendor/scipy_lbfgsb/lbfgsb.c")
set_target_properties(ni26_lbfgsb PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ni26_lbfgsb PRIVATE BLAS_SYMBOL_PREFIX=scipy_)
target_sources(ni26_base PRIVATE "${NI26_LINEAR_ROOT}/src/linear.cpp"
  "${NI26_LINEAR_ROOT}/src/lbfgsb.cpp" "${NI26_LINEAR_ROOT}/src/scaler.cpp")
target_link_libraries(ni26_base PRIVATE ni26_lbfgsb PUBLIC "${NI26_LBFGSB_BLAS_LIBRARY}")
get_filename_component(NI26_LBFGSB_BLAS_DIRECTORY "${NI26_LBFGSB_BLAS_LIBRARY}" DIRECTORY)
set_property(TARGET ni26_base APPEND PROPERTY BUILD_RPATH "${NI26_LBFGSB_BLAS_DIRECTORY}")
add_executable(ni26_linear_source "${NI26_LINEAR_ROOT}/tests/linear_source.cpp")
target_link_libraries(ni26_linear_source PRIVATE ni26_base "${NI26_LBFGSB_BLAS_LIBRARY}")
