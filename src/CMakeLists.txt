# Generic test that uses conan libs
add_executable(measure_LU main.cpp)
target_link_libraries(
    measure_LU
    PRIVATE project_options
            project_warnings
            stopwatch
            tridiagLU)
