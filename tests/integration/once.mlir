// RUN: %sync-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %sync-opt %s --convert-sync-to-std | %FileCheck %s --check-prefix=STD
// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm | %FileCheck %s --check-prefix=LOWER

module {
  func.func @execute_marks_complete() -> i1 {
    %once = memref.alloca() : memref<!sync.once>
    sync.once.init %once : memref<!sync.once>
    sync.once.execute %once : memref<!sync.once> {
      sync.yield
    }
    %completed1 = sync.once.completed %once : memref<!sync.once>
    return %completed1 : i1
  }

  func.func @once_execute_wrapper(%counter: memref<i32>) {
    %once = memref.alloca() : memref<!sync.once>
    %c1 = arith.constant 1 : i32
    sync.once.init %once : memref<!sync.once>
    sync.once.execute %once : memref<!sync.once> {
      %value = memref.load %counter[] : memref<i32>
      %next = arith.addi %value, %c1 : i32
      memref.store %next, %counter[] : memref<i32>
      sync.yield
    }
    sync.once.execute %once : memref<!sync.once> {
      %value = memref.load %counter[] : memref<i32>
      %next = arith.addi %value, %c1 : i32
      memref.store %next, %counter[] : memref<i32>
      sync.yield
    }
    return
  }
}

// ROUNDTRIP: func.func @execute_marks_complete() -> i1
// ROUNDTRIP: sync.once.init
// ROUNDTRIP: sync.once.completed
// ROUNDTRIP: func.func @once_execute_wrapper
// ROUNDTRIP: sync.once.execute

// STD-DAG: func.func private @mlir_sync_call_once_slow_path_prologue(!ptr.ptr<#ptr.generic_space>) -> i1 attributes {no_inline, passthrough = ["cold", "nounwind", "noinline"]}
// STD-DAG: func.func private @mlir_sync_call_once_slow_path_epilogue(!ptr.ptr<#ptr.generic_space>) attributes {no_inline, passthrough = ["cold", "nounwind", "noinline"]}
// STD-LABEL: func.func @execute_marks_complete() -> i1 {
// STD: %[[ONCE:.+]] = memref.alloca() : memref<!sync.once>
// STD: sync.once.init %[[ONCE]] : memref<!sync.once>
// STD: %[[DONE0:.+]] = sync.once.completed %[[ONCE]] : memref<!sync.once>
// STD: scf.if %[[DONE0]] {
// STD: } else {
// STD: %[[ONCE_CAST:.+]] = memref.memory_space_cast %[[ONCE]] : memref<!sync.once> to memref<!sync.once, #ptr.generic_space>
// STD: %[[PTR:.+]] = ptr.to_ptr %[[ONCE_CAST]] : memref<!sync.once, #ptr.generic_space> -> <#ptr.generic_space>
// STD: %[[EXECUTE:.+]] = {{(func\.)?call}} @mlir_sync_call_once_slow_path_prologue(%[[PTR]]) : (!ptr.ptr<#ptr.generic_space>) -> i1
// STD: scf.if %[[EXECUTE]] {
// STD: {{(func\.)?call}} @mlir_sync_call_once_slow_path_epilogue(%[[PTR]]) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: }
// STD: %[[DONE1:.+]] = sync.once.completed %[[ONCE]] : memref<!sync.once>
// STD: return %[[DONE1]] : i1
// STD-LABEL: func.func @once_execute_wrapper(
// STD: %[[ONCE2:.+]] = memref.alloca() : memref<!sync.once>
// STD: sync.once.init %[[ONCE2]] : memref<!sync.once>
// STD: %[[IS_DONE2:.+]] = sync.once.completed %[[ONCE2]] : memref<!sync.once>
// STD: scf.if %[[IS_DONE2]] {
// STD: } else {
// STD: %[[ONCE2_CAST:.+]] = memref.memory_space_cast %[[ONCE2]] : memref<!sync.once> to memref<!sync.once, #ptr.generic_space>
// STD: %[[PTR2:.+]] = ptr.to_ptr %[[ONCE2_CAST]] : memref<!sync.once, #ptr.generic_space> -> <#ptr.generic_space>
// STD: %[[RUN2:.+]] = {{(func\.)?call}} @mlir_sync_call_once_slow_path_prologue(%[[PTR2]]) : (!ptr.ptr<#ptr.generic_space>) -> i1
// STD: scf.if %[[RUN2]] {
// STD: memref.load
// STD: memref.store
// STD: {{(func\.)?call}} @mlir_sync_call_once_slow_path_epilogue(%[[PTR2]]) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: }

// LOWER-DAG: llvm.func @mlir_sync_call_once_slow_path_prologue(!llvm.ptr) -> i1 attributes {passthrough = ["cold", "nounwind", "noinline"], sym_visibility = "private"}
// LOWER-DAG: llvm.func @mlir_sync_call_once_slow_path_epilogue(!llvm.ptr) attributes {passthrough = ["cold", "nounwind", "noinline"], sym_visibility = "private"}
// LOWER-LABEL: llvm.func @execute_marks_complete() -> i1 {
// LOWER: llvm.store
// LOWER: llvm.load
// LOWER: llvm.icmp
// LOWER: llvm.call @mlir_sync_call_once_slow_path_prologue
// LOWER: llvm.call @mlir_sync_call_once_slow_path_epilogue
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @once_execute_wrapper(
// LOWER: llvm.store
// LOWER: llvm.load
// LOWER: llvm.icmp
// LOWER: llvm.call @mlir_sync_call_once_slow_path_prologue
// LOWER: llvm.call @mlir_sync_call_once_slow_path_epilogue
// LOWER: llvm.return
