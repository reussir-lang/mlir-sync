cmake_minimum_required(VERSION 3.28)
project(Sync LANGUAGES C CXX VERSION 0.1.0)

option(SYNC_ENABLE_TESTS "Build sync dialect integration tests" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(MLIR_TABLEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_TABLEGEN_OUTPUT_DIR}
)

include_directories(SYSTEM
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tool)

if(SYNC_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
