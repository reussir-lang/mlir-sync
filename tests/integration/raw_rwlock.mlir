// RUN: %sync-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %sync-opt %s --convert-sync-to-std | %FileCheck %s --check-prefix=STD
// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm | %FileCheck %s --check-prefix=LOWER

module {
  func.func @try_read_then_unlock() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_read_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }

  func.func @read_lock_unlock_once() {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_unlock %rwlock : memref<!sync.raw_rwlock>
    return
  }

  func.func @try_write_then_unlock() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_write_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }

  func.func @write_lock_unlock_once() {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_unlock %rwlock : memref<!sync.raw_rwlock>
    return
  }
}

// ROUNDTRIP: func.func @try_read_then_unlock() -> i1
// ROUNDTRIP: sync.raw_rwlock.init
// ROUNDTRIP: %{{.*}} = sync.raw_rwlock.try_read_lock
// ROUNDTRIP: sync.raw_rwlock.read_unlock
// ROUNDTRIP: func.func @read_lock_unlock_once()
// ROUNDTRIP: sync.raw_rwlock.read_lock
// ROUNDTRIP: sync.raw_rwlock.read_unlock
// ROUNDTRIP: func.func @try_write_then_unlock() -> i1
// ROUNDTRIP: %{{.*}} = sync.raw_rwlock.try_write_lock
// ROUNDTRIP: sync.raw_rwlock.write_unlock
// ROUNDTRIP: func.func @write_lock_unlock_once()
// ROUNDTRIP: sync.raw_rwlock.write_lock
// ROUNDTRIP: sync.raw_rwlock.write_unlock

// STD-DAG: func.func private @mlir_sync_rwlock_read_lock_slow_path(!ptr.ptr<#ptr.generic_space>)
// STD-DAG: func.func private @mlir_sync_rwlock_write_lock_slow_path(!ptr.ptr<#ptr.generic_space>)
// STD-DAG: func.func private @mlir_sync_rwlock_unlock_slow_path(!ptr.ptr<#ptr.generic_space>, i32)
// STD-LABEL: func.func @try_read_then_unlock() -> i1 {
// STD: %[[TRYREAD:.+]]:2 = scf.while
// STD: %[[STATE:.+]] = sync.raw_rwlock.load_state %{{.*}} : memref<!sync.raw_rwlock>
// STD: %[[CANREAD:.+]] = arith.andi %{{.*}}, %{{.*}} : i1
// STD: %[[STEP:.+]]:2 = scf.if %[[CANREAD]] -> (i1, i1) {
// STD: %[[NEXT:.+]] = arith.addi %[[STATE]], %{{.*}} : i32
// STD: %[[CAS:.+]] = sync.raw_rwlock.cmpxchg_state %{{.*}}, %[[STATE]], %[[NEXT]] : memref<!sync.raw_rwlock>
// STD: scf.yield %[[CAS]], %[[CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: %[[UNLOCK:.+]] = sync.raw_rwlock.read_unlock_fast %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }
// STD: return %[[TRYREAD]]#1 : i1
// STD-LABEL: func.func @read_lock_unlock_once() {
// STD: %[[READLOCK:.+]]:2 = scf.while
// STD: %[[STATE:.+]] = sync.raw_rwlock.load_state %{{.*}} : memref<!sync.raw_rwlock>
// STD: %[[CANREAD:.+]] = arith.andi %{{.*}}, %{{.*}} : i1
// STD: %[[STEP:.+]]:2 = scf.if %[[CANREAD]] -> (i1, i1) {
// STD: %[[NEXT:.+]] = arith.addi %[[STATE]], %{{.*}} : i32
// STD: %[[CAS:.+]] = sync.raw_rwlock.cmpxchg_state %{{.*}}, %[[STATE]], %[[NEXT]] : memref<!sync.raw_rwlock>
// STD: scf.yield %[[CAS]], %[[CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: scf.if %[[READLOCK]]#1 {
// STD: } else {
// STD: func.call @mlir_sync_rwlock_read_lock_slow_path(%{{.*}}) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: %[[UNLOCK:.+]] = sync.raw_rwlock.read_unlock_fast %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }
// STD-LABEL: func.func @try_write_then_unlock() -> i1 {
// STD: %[[TRYWRITE:.+]]:2 = scf.while
// STD: %[[STATE:.+]] = sync.raw_rwlock.load_state %{{.*}} : memref<!sync.raw_rwlock>
// STD: %[[STEP:.+]]:2 = scf.if %{{.*}} -> (i1, i1) {
// STD: %[[CAS:.+]] = sync.raw_rwlock.cmpxchg_state %{{.*}}, %{{.*}}, %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.yield %[[CAS]], %[[CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: %[[UNLOCK:.+]] = sync.raw_rwlock.write_unlock_fast %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }
// STD: return %[[TRYWRITE]]#1 : i1
// STD-LABEL: func.func @write_lock_unlock_once() {
// STD: %[[WRITELOCK:.+]]:2 = scf.while
// STD: %[[STATE:.+]] = sync.raw_rwlock.load_state %{{.*}} : memref<!sync.raw_rwlock>
// STD: %[[STEP:.+]]:2 = scf.if %{{.*}} -> (i1, i1) {
// STD: %[[CAS:.+]] = sync.raw_rwlock.cmpxchg_state %{{.*}}, %{{.*}}, %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.yield %[[CAS]], %[[CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: scf.if %[[WRITELOCK]]#1 {
// STD: } else {
// STD: func.call @mlir_sync_rwlock_write_lock_slow_path(%{{.*}}) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: %[[UNLOCK:.+]] = sync.raw_rwlock.write_unlock_fast %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }

// LOWER-DAG: llvm.func @mlir_sync_rwlock_read_lock_slow_path(!llvm.ptr)
// LOWER-DAG: llvm.func @mlir_sync_rwlock_write_lock_slow_path(!llvm.ptr)
// LOWER-DAG: llvm.func @mlir_sync_rwlock_unlock_slow_path(!llvm.ptr, i32)
// LOWER-LABEL: llvm.func @try_read_then_unlock() -> i1 {
// LOWER: llvm.load
// LOWER: llvm.cmpxchg
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @read_lock_unlock_once() {
// LOWER: llvm.cmpxchg
// LOWER: llvm.call @mlir_sync_rwlock_read_lock_slow_path
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @try_write_then_unlock() -> i1 {
// LOWER: llvm.cmpxchg
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @write_lock_unlock_once() {
// LOWER: llvm.cmpxchg
// LOWER: llvm.call @mlir_sync_rwlock_write_lock_slow_path
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.return
