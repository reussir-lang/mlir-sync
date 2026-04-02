message(STATUS "finding tools in ${LLVM_TOOLS_BINARY_DIR}")

function(find_llvm_tool var tool)
  find_program(${var}
    NAMES "${tool}${CMAKE_EXECUTABLE_SUFFIX}"
    HINTS ${LLVM_TOOLS_BINARY_DIR}
    REQUIRED)
  message(STATUS "${tool} path: ${${var}}")
endfunction()

find_llvm_tool(filecheck_path FileCheck)
find_llvm_tool(mlir_translate_path mlir-translate)
find_llvm_tool(not_path not)
find_program(clang_path NAMES clang REQUIRED)
find_program(cargo_path NAMES cargo REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
message(STATUS "Python3 path: ${Python3_EXECUTABLE}")
message(STATUS "clang path: ${clang_path}")
message(STATUS "cargo path: ${cargo_path}")

set(sync_cargo_target_dir ${CMAKE_BINARY_DIR}/cargo)
set(sync_runtime_lib_path
  ${sync_cargo_target_dir}/release/${CMAKE_STATIC_LIBRARY_PREFIX}mlir_sync${CMAKE_STATIC_LIBRARY_SUFFIX})

add_custom_command(
  OUTPUT ${sync_runtime_lib_path}
  COMMAND ${CMAKE_COMMAND} -E env
          CARGO_TARGET_DIR=${sync_cargo_target_dir}
          ${cargo_path} build -p mlir_sync --release
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS
    ${CMAKE_SOURCE_DIR}/Cargo.toml
    ${CMAKE_SOURCE_DIR}/Cargo.lock
    ${CMAKE_SOURCE_DIR}/crates/mlir_sync/Cargo.toml
    ${CMAKE_SOURCE_DIR}/crates/mlir_sync/src/lib.rs
    ${CMAKE_SOURCE_DIR}/crates/mlir_sync/src/mutex.rs
    ${CMAKE_SOURCE_DIR}/crates/portable_futex/Cargo.toml
    ${CMAKE_SOURCE_DIR}/crates/portable_futex/build.rs
    ${CMAKE_SOURCE_DIR}/crates/portable_futex/src/lib.rs
  VERBATIM
)

add_custom_target(sync-runtime DEPENDS ${sync_runtime_lib_path})

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

add_custom_target(check-sync
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_lit.py -j 1
          ${CMAKE_CURRENT_BINARY_DIR} -v
  DEPENDS sync-opt sync-runtime
)
