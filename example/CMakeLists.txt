cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(example-cpp-backend LANGUAGES CXX C)

set(CMAKE_VERBOSE_MAKEFILE on)
set(LIB_NAME cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTORCH_EXTENSION_NAME=${LIB_NAME}")
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(PythonLibs REQUIRED)

find_library(TORCH_PYTHON_LIBRARY torch_python
             PATHS "${TORCH_INSTALL_PREFIX}/lib")

add_library(${LIB_NAME} SHARED example.cpp)

set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(${LIB_NAME} PROPERTIES PREFIX "")

if(APPLE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set_target_properties(${LIB_NAME} PROPERTIES LINK_FLAGS
                                               "-Wl,-undefined,dynamic_lookup")
elseif(WIN32)
  message(FATAL_ERROR "Windows is not supported")
endif()

target_include_directories(
  ${LIB_NAME}
  PUBLIC ${TORCH_INCLUDE_DIRS}
  PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(
  ${LIB_NAME}
  PUBLIC ${TORCH_LIBRARIES}
  PUBLIC ${TORCH_PYTHON_LIBRARY})
link_directories(${PYTHON_LIBRARIES})
