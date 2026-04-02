// RUN: %sync-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %sync-opt %s --convert-sync-to-std | %FileCheck %s --check-prefix=STD
// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-sync-to-llvm --convert-cf-to-llvm | %FileCheck %s --check-prefix=LOWER

module {
  func.func @try_lock_once() -> i1 {
    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    %locked = sync.raw_mutex.try_lock %mutex : memref<!sync.raw_mutex>
    return %locked : i1
  }

  func.func @lock_unlock_once() {
    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.lock %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.unlock %mutex : memref<!sync.raw_mutex>
    return
  }
}

// ROUNDTRIP: func.func @try_lock_once() -> i1
// ROUNDTRIP: %{{.*}} = memref.alloca() : memref<!sync.raw_mutex>
// ROUNDTRIP: sync.raw_mutex.init
// ROUNDTRIP: %{{.*}} = sync.raw_mutex.try_lock
// ROUNDTRIP: func.func @lock_unlock_once()
// ROUNDTRIP: sync.raw_mutex.lock
// ROUNDTRIP: sync.raw_mutex.unlock

// STD-DAG: func.func private @mlir_sync_mutex_lock_slow_path(!ptr.ptr<#ptr.generic_space>)
// STD-DAG: func.func private @mlir_sync_mutex_unlock_slow_path(!ptr.ptr<#ptr.generic_space>)
// STD: func.func @try_lock_once() -> i1
// STD: %[[MUTEX:.+]] = memref.alloca() : memref<!sync.raw_mutex>
// STD: sync.raw_mutex.init %[[MUTEX]]
// STD: %[[LOCKED:.+]] = sync.raw_mutex.try_lock %[[MUTEX]]
// STD: return %[[LOCKED]] : i1
// STD: func.func @lock_unlock_once()
// STD: %[[LOCK_ACQUIRED:.+]] = sync.raw_mutex.try_lock %[[MUTEX]]
// STD: scf.if %[[LOCK_ACQUIRED]] {
// STD: } else {
// STD: %[[LOCK_CAST:.+]] = memref.memory_space_cast %[[MUTEX]] : memref<!sync.raw_mutex> to memref<!sync.raw_mutex, #ptr.generic_space>
// STD: %[[PTR:.+]] = ptr.to_ptr %[[LOCK_CAST]] : memref<!sync.raw_mutex, #ptr.generic_space> -> <#ptr.generic_space>
// STD: {{(func\.)?call}} @mlir_sync_mutex_lock_slow_path(%[[PTR]]) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: %[[NEEDS_WAKE:.+]] = sync.raw_mutex.unlock_fast %[[MUTEX]] : memref<!sync.raw_mutex>
// STD: scf.if %[[NEEDS_WAKE]] {
// STD: %[[UNLOCK_CAST:.+]] = memref.memory_space_cast %[[MUTEX]] : memref<!sync.raw_mutex> to memref<!sync.raw_mutex, #ptr.generic_space>
// STD: %[[SLOW_PTR:.+]] = ptr.to_ptr %[[UNLOCK_CAST]] : memref<!sync.raw_mutex, #ptr.generic_space> -> <#ptr.generic_space>
// STD: {{(func\.)?call}} @mlir_sync_mutex_unlock_slow_path(%[[SLOW_PTR]]) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }

// LOWER-DAG: llvm.func @mlir_sync_mutex_lock_slow_path(!llvm.ptr)
// LOWER-DAG: llvm.func @mlir_sync_mutex_unlock_slow_path(!llvm.ptr)
// LOWER: llvm.func @try_lock_once() -> i1 {
// LOWER: llvm.store
// LOWER: llvm.cmpxchg
// LOWER: llvm.extractvalue
// LOWER: llvm.return
// LOWER: llvm.func @lock_unlock_once()
// LOWER: llvm.cmpxchg
// LOWER: llvm.addrspacecast
// LOWER: llvm.call @mlir_sync_mutex_lock_slow_path
// LOWER: llvm.atomicrmw xchg
// LOWER: llvm.cond_br
// LOWER: llvm.call @mlir_sync_mutex_unlock_slow_path
