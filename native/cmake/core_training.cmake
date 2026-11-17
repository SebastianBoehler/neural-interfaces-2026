# Include after the base target and linear.cmake. No Python training driver.
get_filename_component(NI26_CORE_TRAINING_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
target_sources(ni26_base PRIVATE "${NI26_CORE_TRAINING_ROOT}/src/core_training.cpp"
  "${NI26_CORE_TRAINING_ROOT}/src/core_training_window.cpp")
add_executable(ni26_core_train "${NI26_CORE_TRAINING_ROOT}/tools/core_train.cpp")
target_link_libraries(ni26_core_train PRIVATE ni26_base)
# Prepared existing-reference source test; no fit is run during the build.
add_executable(ni26_core_training_source "${NI26_CORE_TRAINING_ROOT}/tests/core_training_source.cpp")
target_link_libraries(ni26_core_training_source PRIVATE ni26_base)
