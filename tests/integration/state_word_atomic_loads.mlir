// RUN: %sync-opt %s --convert-to-llvm | %FileCheck %s --check-prefix=LOWER
// RUN: %sync-opt %s --convert-to-llvm | %mlir-translate --mlir-to-llvmir | %FileCheck %s --check-prefix=LLVMIR

// Loads of lock state words race with the cmpxchg/atomicrmw updates of other
// threads, so they must lower to atomic loads: a plain load is a data race
// under the LLVM memory model.

module {
  // The rwlock fast paths only read the state word to compute the operands of
  // the acquire cmpxchg that actually takes the lock, so monotonic suffices.
  func.func @raw_rwlock_load_state(%rwlock: memref<!sync.raw_rwlock>) -> i32 {
    %state = sync.raw_rwlock.load_state %rwlock : memref<!sync.raw_rwlock>
    return %state : i32
  }

  // A completed once lets the caller skip the slow path and read whatever the
  // once body initialized, so the check must acquire.
  func.func @once_completed(%once: memref<!sync.once>) -> i1 {
    %completed = sync.once.completed %once : memref<!sync.once>
    return %completed : i1
  }
}

// LOWER-LABEL: llvm.func @raw_rwlock_load_state(
// LOWER: %{{.+}} = llvm.load %{{.+}} atomic monotonic {alignment = 4 : i64} : !llvm.ptr -> i32
// LOWER-NOT: llvm.load
// LOWER: llvm.return
// LOWER-LABEL: llvm.func @once_completed(
// LOWER: %{{.+}} = llvm.load %{{.+}} atomic acquire {alignment = 4 : i64} : !llvm.ptr -> i32
// LOWER-NOT: llvm.load
// LOWER: llvm.return

// LLVMIR-LABEL: define i32 @raw_rwlock_load_state(
// LLVMIR: load atomic i32, ptr %{{.+}} monotonic, align 4
// LLVMIR-NOT: load i32
// LLVMIR: ret i32
// LLVMIR-LABEL: define i1 @once_completed(
// LLVMIR: load atomic i32, ptr %{{.+}} acquire, align 4
// LLVMIR-NOT: load i32
// LLVMIR: ret i1
