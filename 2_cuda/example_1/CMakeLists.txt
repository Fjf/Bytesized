# Install CUDA kernels as a library to not have to compile the entire project with nvcc, but only the kernel section.
add_library(kernels SHARED kernels.cu kernels.cuh)
set_target_properties(
        kernels
        PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

target_compile_options(
        kernels
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -Xcompiler
        -fPIC
        -shared
        -dc
        -O3
        >)
target_link_libraries(kernels CUDA::cudart_static)

# Create executable and link the kernels for installation.
add_executable(example_1 main.cpp)
set_target_properties(example_1 PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(example_1 kernels)

# Install the executable to ensure the CUDA toolkit stubs are not linked at runtime.
set_target_properties(example_1 PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)
set_target_properties(example_1 PROPERTIES INSTALL_RPATH "${CMAKE_SOURCE_DIR}/build/example_1")
install(TARGETS example_1
        CONFIGURATIONS Release
        RUNTIME DESTINATION bin)
