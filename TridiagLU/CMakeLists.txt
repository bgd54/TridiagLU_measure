add_library(
    tridiagLU
    SHARED
    src/TridiagLU/tridiagLU.cpp
    src/TridiagLU/tridiagLUGS.cpp
    src/TridiagLU/tridiagIterJacobi.cpp
    src/TridiagLU/tridiagLUInit.cpp
    src/TridiagLU/tridiagScaLPK.cpp
    src/TridiagLU/blocktridiagLU.cpp
    src/TridiagLU/blocktridiagIterJacobi.cpp)

target_link_libraries(tridiagLU PRIVATE project_options)

target_include_directories(tridiagLU PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
