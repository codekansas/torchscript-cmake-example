cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(example-cpp-backend LANGUAGES CXX C)

set(LIB_NAME cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMODULE_NAME=lib${LIB_NAME}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-reorder")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(PythonLibs REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python
             PATHS "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")

add_library(${LIB_NAME} SHARED example.cpp)

target_compile_features(${LIB_NAME} PRIVATE cxx_std_17)
set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${LIB_NAME}
                      PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")
target_compile_options(
  ${LIB_NAME} PUBLIC -Wall -Wno-sign-compare -Wno-unused-function
                     -Wno-unknown-pragmas)
target_include_directories(
  ${LIB_NAME}
  PUBLIC ${TORCH_INCLUDE_DIRS}
  PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(
  ${LIB_NAME}
  PUBLIC ${PYTHON_LIBRARIES}
  PUBLIC ${TORCH_LIBRARIES}
  PUBLIC ${TORCH_PYTHON_LIBRARY})

message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
