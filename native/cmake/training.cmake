# The source fits use the same native OpenBLAS symbols as the reference.
# The worker links only ni26_base and ni26_runtime.
if(NOT DEFINED NI26_LBFGSB_BLAS_LIBRARY OR NOT EXISTS "${NI26_LBFGSB_BLAS_LIBRARY}")
  message(FATAL_ERROR "Set NI26_LBFGSB_BLAS_LIBRARY to the existing SciPy-prefixed LP64 native OpenBLAS library")
endif()
enable_language(C)
add_library(ni26_lbfgsb STATIC vendor/scipy_lbfgsb/lbfgsb.c)
set_target_properties(ni26_lbfgsb PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ni26_lbfgsb PRIVATE BLAS_SYMBOL_PREFIX=scipy_)

add_library(ni26_initialization STATIC src/initialization.cpp)
target_link_libraries(ni26_initialization PUBLIC ni26_base)
target_compile_options(ni26_initialization PRIVATE -O2 -Wall -Wextra)
add_library(ni26_training STATIC
  src/source.cpp src/preparation.cpp src/sampling.cpp src/sampling_rng.cpp src/sampling_choice.cpp
  src/linear.cpp src/lbfgsb.cpp src/scaler.cpp src/pca.cpp
  src/core_training.cpp src/core_training_window.cpp
  src/session_bn.cpp src/training_fbc.cpp src/training.cpp src/training_source.cpp)
target_link_libraries(ni26_training PUBLIC ni26_runtime
  PRIVATE ni26_initialization ni26_lbfgsb "${NI26_LBFGSB_BLAS_LIBRARY}")
target_compile_options(ni26_training PRIVATE -O2 -Wall -Wextra)

add_executable(ni26_core_train tools/core_train.cpp)
add_executable(ni26_train_raw tools/train_raw.cpp)
add_executable(ni26_core_training_source tests/core_training_source.cpp)
add_executable(ni26_neural_training_source tests/neural_training_source.cpp)
add_executable(ni26_linear_source tests/linear_source.cpp)
add_executable(ni26_pca_source tests/pca_source.cpp)
add_executable(ni26_preparation_source tests/preparation_source.cpp)
foreach(target ni26_core_train ni26_train_raw ni26_core_training_source ni26_neural_training_source
    ni26_linear_source ni26_pca_source ni26_preparation_source)
  target_link_libraries(${target} PRIVATE ni26_training)
endforeach()

add_executable(ni26_init_raw tools/init_raw.cpp)
target_link_libraries(ni26_init_raw PRIVATE ni26_initialization)
add_executable(ni26_initialization_source tests/initialization_source.cpp)
target_link_libraries(ni26_initialization_source PRIVATE ni26_initialization)
