cmake_minimum_required(VERSION 3.10)
project(KGraphProject CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${BIN_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

include_directories(include)

find_package(Boost REQUIRED COMPONENTS timer program_options)
find_package(OpenMP REQUIRED)
link_libraries(Boost::timer Boost::program_options OpenMP::OpenMP_CXX)

set(SRC_DIR src)
set(MAIN_DIR main)
set(KGRAPH_SRC ${SRC_DIR}/kgraph.cpp ${SRC_DIR}/MemoryUtils.cpp)
set(KGRAPH_RN_SRC ${SRC_DIR}/kgraph_RN.cpp ${SRC_DIR}/MemoryUtils.cpp)
set(KGRAPH_RS_SRC ${SRC_DIR}/kgraph_RS.cpp ${SRC_DIR}/MemoryUtils.cpp)
set(KGRAPH_BJ_SRC ${SRC_DIR}/kgraph_BJ.cpp ${SRC_DIR}/MemoryUtils.cpp)
set(KGRAPH_SQ16_SRC ${SRC_DIR}/kgraph_SQ16.cpp ${SRC_DIR}/MemoryUtils.cpp)
set(KGRAPH_PLUS_SRC ${SRC_DIR}/kgraph_PLUS.cpp ${SRC_DIR}/MemoryUtils.cpp)

add_executable(kgraph ${MAIN_DIR}/main.cpp ${KGRAPH_SRC})
add_executable(kgraph_RN ${MAIN_DIR}/main.cpp ${KGRAPH_RN_SRC})
add_executable(kgraph_RS ${MAIN_DIR}/main.cpp ${KGRAPH_RS_SRC})
add_executable(kgraph_BJ ${MAIN_DIR}/main.cpp ${KGRAPH_BJ_SRC})
add_executable(kgraph_SQ16 ${MAIN_DIR}/main_SQ16.cpp ${KGRAPH_SQ16_SRC})
add_executable(kgraph_PLUS ${MAIN_DIR}/main_SQ16.cpp ${KGRAPH_PLUS_SRC})

target_compile_definitions(kgraph_SQ16 PRIVATE SHORT)
target_compile_definitions(kgraph_PLUS PRIVATE SHORT)

foreach(target kgraph kgraph_RN kgraph_RS kgraph_BJ kgraph_SQ16 kgraph_PLUS)
    target_link_libraries(${target} Boost::timer Boost::program_options OpenMP::OpenMP_CXX)
endforeach()


