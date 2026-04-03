// RUN: %sync-opt %s | %FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %sync-opt %s --convert-sync-to-std | %FileCheck %s --check-prefix=STD
// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm | %FileCheck %s --check-prefix=LOWER

module {
  func.func @project_and_try_write() -> i1 {
    %c0_i64 = arith.constant 0 : i64
    %rwlock = memref.alloca() : memref<!sync.rwlock<i64>>
    sync.rwlock.init %rwlock : memref<!sync.rwlock<i64>>, %c0_i64 : i64
    %raw = sync.rwlock.get_raw_rwlock %rwlock
      : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
    %payload = sync.rwlock.get_payload %rwlock
      : memref<!sync.rwlock<i64>> -> memref<i64>
    %locked = sync.raw_rwlock.try_write_lock %raw : memref<!sync.raw_rwlock>
    scf.if %locked {
      sync.raw_rwlock.write_unlock %raw : memref<!sync.raw_rwlock>
    }
    %value = memref.load %payload[] : memref<i64>
    %is_zero = arith.cmpi eq, %value, %c0_i64 : i64
    return %is_zero : i1
  }

  func.func @rwlock_sections() -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %rwlock = memref.alloca() : memref<!sync.rwlock<i64>>
    sync.rwlock.init %rwlock : memref<!sync.rwlock<i64>>, %c0_i64 : i64

    sync.rwlock.write_critical_section %rwlock : memref<!sync.rwlock<i64>> {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      %next = arith.addi %value, %c1_i64 : i64
      memref.store %next, %payload[] : memref<i64>
      sync.yield
    }

    %result = sync.rwlock.read_critical_section %rwlock
      : memref<!sync.rwlock<i64>> -> i64 {
    ^bb0(%payload: memref<i64>):
      %value = memref.load %payload[] : memref<i64>
      sync.yield %value : i64
    }
    return %result : i64
  }
}

// ROUNDTRIP: func.func @project_and_try_write() -> i1
// ROUNDTRIP: sync.rwlock.init
// ROUNDTRIP: sync.rwlock.get_raw_rwlock
// ROUNDTRIP: sync.rwlock.get_payload
// ROUNDTRIP: %{{.*}} = sync.raw_rwlock.try_write_lock
// ROUNDTRIP: func.func @rwlock_sections() -> i64
// ROUNDTRIP: sync.rwlock.write_critical_section
// ROUNDTRIP: sync.rwlock.read_critical_section

// STD-LABEL: func.func @project_and_try_write() -> i1 {
// STD: %[[RAW_INIT:.+]] = sync.rwlock.get_raw_rwlock %{{.*}} : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
// STD: sync.raw_rwlock.init %[[RAW_INIT]] : memref<!sync.raw_rwlock>
// STD: %[[PAYLOAD_INIT:.+]] = sync.rwlock.get_payload %{{.*}} : memref<!sync.rwlock<i64>> -> memref<i64>
// STD: memref.store %{{.*}}, %[[PAYLOAD_INIT]][] : memref<i64>
// STD: %[[RAW:.+]] = sync.rwlock.get_raw_rwlock %{{.*}} : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
// STD: %[[PAYLOAD:.+]] = sync.rwlock.get_payload %{{.*}} : memref<!sync.rwlock<i64>> -> memref<i64>
// STD: %[[TRYWRITE:.+]]:2 = scf.while
// STD: %[[STATE:.+]] = sync.raw_rwlock.load_state %[[RAW]] : memref<!sync.raw_rwlock>
// STD: %[[STEP:.+]]:2 = scf.if %{{.*}} -> (i1, i1) {
// STD: %[[CAS:.+]] = sync.raw_rwlock.cmpxchg_state %[[RAW]], %{{.*}}, %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.yield %[[CAS]], %[[CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: scf.if %[[TRYWRITE]]#1 {
// STD: %[[UNLOCK:.+]] = sync.raw_rwlock.write_unlock_fast %[[RAW]] : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }
// STD: }
// STD: %{{.*}} = memref.load %[[PAYLOAD]][] : memref<i64>
// STD-LABEL: func.func @rwlock_sections() -> i64 {
// STD: %[[RAW_INIT:.+]] = sync.rwlock.get_raw_rwlock %{{.*}} : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
// STD: sync.raw_rwlock.init %[[RAW_INIT]] : memref<!sync.raw_rwlock>
// STD: %[[PAYLOAD_INIT:.+]] = sync.rwlock.get_payload %{{.*}} : memref<!sync.rwlock<i64>> -> memref<i64>
// STD: memref.store %{{.*}}, %[[PAYLOAD_INIT]][] : memref<i64>
// STD: %[[WRITE_RAW:.+]] = sync.rwlock.get_raw_rwlock %{{.*}} : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
// STD: %[[WRITE_LOOP:.+]]:2 = scf.while
// STD: %[[WRITE_STATE:.+]] = sync.raw_rwlock.load_state %[[WRITE_RAW]] : memref<!sync.raw_rwlock>
// STD: %[[WRITE_STEP:.+]]:2 = scf.if %{{.*}} -> (i1, i1) {
// STD: %[[WRITE_CAS:.+]] = sync.raw_rwlock.cmpxchg_state %[[WRITE_RAW]], %{{.*}}, %{{.*}} : memref<!sync.raw_rwlock>
// STD: scf.yield %[[WRITE_CAS]], %[[WRITE_CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: scf.if %[[WRITE_LOOP]]#1 {
// STD: } else {
// STD: func.call @mlir_sync_rwlock_write_lock_slow_path(%{{.*}}) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: %[[WRITE_PAYLOAD:.+]] = sync.rwlock.get_payload %{{.*}} : memref<!sync.rwlock<i64>> -> memref<i64>
// STD: memref.store %{{.*}}, %[[WRITE_PAYLOAD]][] : memref<i64>
// STD: %[[WRITE_UNLOCK:.+]] = sync.raw_rwlock.write_unlock_fast %[[WRITE_RAW]] : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[WRITE_UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }
// STD: %[[READ_RAW:.+]] = sync.rwlock.get_raw_rwlock %{{.*}} : memref<!sync.rwlock<i64>> -> memref<!sync.raw_rwlock>
// STD: %[[READ_LOOP:.+]]:2 = scf.while
// STD: %[[READ_STATE:.+]] = sync.raw_rwlock.load_state %[[READ_RAW]] : memref<!sync.raw_rwlock>
// STD: %[[READ_CAN:.+]] = arith.andi %{{.*}}, %{{.*}} : i1
// STD: %[[READ_STEP:.+]]:2 = scf.if %[[READ_CAN]] -> (i1, i1) {
// STD: %[[READ_NEXT:.+]] = arith.addi %[[READ_STATE]], %{{.*}} : i32
// STD: %[[READ_CAS:.+]] = sync.raw_rwlock.cmpxchg_state %[[READ_RAW]], %[[READ_STATE]], %[[READ_NEXT]] : memref<!sync.raw_rwlock>
// STD: scf.yield %[[READ_CAS]], %[[READ_CAS]] : i1, i1
// STD: } else {
// STD: scf.yield %{{.*}}, %{{.*}} : i1, i1
// STD: }
// STD: scf.if %[[READ_LOOP]]#1 {
// STD: } else {
// STD: func.call @mlir_sync_rwlock_read_lock_slow_path(%{{.*}}) : (!ptr.ptr<#ptr.generic_space>) -> ()
// STD: }
// STD: %[[READ_PAYLOAD:.+]] = sync.rwlock.get_payload %{{.*}} : memref<!sync.rwlock<i64>> -> memref<i64>
// STD: %{{.*}} = memref.load %[[READ_PAYLOAD]][] : memref<i64>
// STD: %[[READ_UNLOCK:.+]] = sync.raw_rwlock.read_unlock_fast %[[READ_RAW]] : memref<!sync.raw_rwlock>
// STD: scf.if %{{.*}} {
// STD: func.call @mlir_sync_rwlock_unlock_slow_path(%{{.*}}, %[[READ_UNLOCK]]) : (!ptr.ptr<#ptr.generic_space>, i32) -> ()
// STD: }

// LOWER-LABEL: llvm.func @project_and_try_write() -> i1 {
// LOWER: llvm.store
// LOWER: llvm.store
// LOWER: llvm.cmpxchg
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.load
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @rwlock_sections() -> i64 {
// LOWER: llvm.cmpxchg
// LOWER: llvm.call @mlir_sync_rwlock_write_lock_slow_path
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.cmpxchg
// LOWER: llvm.call @mlir_sync_rwlock_read_lock_slow_path
// LOWER: llvm.atomicrmw sub
// LOWER: llvm.call @mlir_sync_rwlock_unlock_slow_path
// LOWER: llvm.return
