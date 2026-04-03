// RUN: %sync-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %sync-opt %s --convert-sync-to-std | %FileCheck %s --check-prefix=STD
// RUN: %sync-opt %s --attach-host-llvm-layout --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm | %FileCheck %s --check-prefix=LLVM

module {
  func.func @combining_lock_capture_shape(%external: i32) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %lock = memref.alloca() : memref<!sync.combining_lock<i64>>

    sync.combining_lock.init %lock : memref<!sync.combining_lock<i64>>, %c0_i64 : i64
    sync.combining_lock.critical_section %lock {combine_limit = 7 : i64}
      : memref<!sync.combining_lock<i64>> {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      %external_i64 = arith.extsi %external : i32 to i64
      %sum = arith.addi %value, %external_i64 : i64
      %next = arith.addi %sum, %c1_i64 : i64
      memref.store %next, %payload[] : memref<i64>
      sync.yield
    }
    return
  }

  func.func @combining_lock_default_limit() {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %lock = memref.alloca() : memref<!sync.combining_lock<i64>>

    sync.combining_lock.init %lock : memref<!sync.combining_lock<i64>>, %c0_i64 : i64
    sync.combining_lock.critical_section %lock : memref<!sync.combining_lock<i64>> {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      %next = arith.addi %value, %c1_i64 : i64
      memref.store %next, %payload[] : memref<i64>
      sync.yield
    }
    return
  }
}

// ROUNDTRIP: sync.combining_lock.init
// ROUNDTRIP: sync.combining_lock.critical_section %{{.*}} {combine_limit = 7 : i64}

// STD: func.func private @__sync_combining_lock_slow_{{[0-9]+}}(%arg0: !ptr.ptr<#ptr.generic_space>) attributes {llvm.linkage = #llvm.linkage<internal>, no_inline, passthrough = ["cold"]}
// STD: %[[RECOVERED_PAYLOAD:.+]]:3 = "sync.combining_lock.recover"(%{{.*}}) : (!ptr.ptr<#ptr.generic_space>) -> (memref<i64>, i32, i64)
// STD-LABEL: func.func @combining_lock_capture_shape
// STD-DAG: %[[PAYLOAD:.+]] = sync.combining_lock.get_payload %{{.*}} : memref<!sync.combining_lock<i64>> -> memref<i64>
// STD-DAG: %[[TRUE:.+]] = arith.constant true
// STD-DAG: %[[FALSE:.+]] = arith.constant false
// STD: %[[HAS_TAIL:.+]] = sync.combining_lock.has_tail
// STD: %[[SHOULD_SLOW:.+]] = scf.if %[[HAS_TAIL]] -> (i1)
// STD: scf.yield %[[TRUE]] : i1
// STD: %[[TRY_ACQUIRE:.+]] = sync.combining_lock.try_acquire
// STD: %[[NO_FAST_PATH:.+]] = scf.if %[[TRY_ACQUIRE]] -> (i1)
// STD: scf.yield %[[FALSE]] : i1
// STD: scf.yield %[[TRUE]] : i1
// STD: scf.yield %[[NO_FAST_PATH]] : i1
// STD: scf.if %[[SHOULD_SLOW]]
// STD: %[[NODE:.+]], %[[RAW_NODE:.+]] = "sync.combining_lock.capture"(%{{.*}}, %[[PAYLOAD]], %arg0, %c1_i64) : ((!ptr.ptr<#ptr.generic_space>) -> (), memref<i64>, i32, i64) -> (memref<!sync.combining_lock_node<i32, i64>, #ptr.generic_space>, !ptr.ptr<#ptr.generic_space>)
// STD: func.call @mlir_sync_combining_lock_attach_slow_path
// STD: sync.combining_lock.capture_end %[[RAW_NODE]] : !ptr.ptr<#ptr.generic_space>
// STD: sync.combining_lock.release

// STD-LABEL: func.func @combining_lock_default_limit
// STD: arith.constant -1 : i64

// LLVM: llvm.func internal @__sync_combining_lock_slow_{{[0-9]+}}(%arg0: !llvm.ptr) attributes {passthrough = ["cold"], sym_visibility = "private"}
// LLVM-LABEL: llvm.func @combining_lock_capture_shape
// LLVM: llvm.atomicrmw xchg
// LLVM: llvm.alloca
// LLVM: llvm.intr.lifetime.start
// LLVM: llvm.store %{{.*}}, %{{.*}} invariant_group : !llvm.ptr, !llvm.ptr
// LLVM: llvm.call @mlir_sync_combining_lock_attach_slow_path
// LLVM: llvm.intr.lifetime.end

// LLVM-LABEL: llvm.func @combining_lock_default_limit
// LLVM: llvm.mlir.constant(-1 : i64)
