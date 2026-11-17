# Include after linear.cmake: reuse its explicit qualified LP64 OpenBLAS provider.
get_filename_component(NI26_PCA_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
target_sources(ni26_base PRIVATE "${NI26_PCA_ROOT}/src/pca.cpp")
add_executable(ni26_pca_source "${NI26_PCA_ROOT}/tests/pca_source.cpp")
target_link_libraries(ni26_pca_source PRIVATE ni26_base)
