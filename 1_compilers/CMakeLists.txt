add_executable(ex1 kernels.cpp)

add_executable(ex2 example_2.cpp)

add_executable(ex2_fast example_2.cpp)
set_target_properties(ex2_fast PROPERTIES COMPILE_FLAGS "-Ofast")

add_executable(ex3 example_3.cpp)
add_executable(ex3_fast example_3.cpp)
set_target_properties(ex3_fast PROPERTIES COMPILE_FLAGS "-Ofast")

add_executable(ex4 example_4.cpp)
add_executable(ex4_fast example_4.cpp)
set_target_properties(ex4_fast PROPERTIES COMPILE_FLAGS "-Ofast")
