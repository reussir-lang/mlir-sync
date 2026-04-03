// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 %S/raw_rwlock_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @rwlock_try_read_returns_true() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_read_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }

  func.func @rwlock_try_write_returns_true() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_write_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }

  func.func @rwlock_write_then_try_read_returns_false() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_lock %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_read_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.write_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }

  func.func @rwlock_read_then_try_write_returns_false() -> i1 {
    %rwlock = memref.alloca() : memref<!sync.raw_rwlock>
    sync.raw_rwlock.init %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_lock %rwlock : memref<!sync.raw_rwlock>
    %locked = sync.raw_rwlock.try_write_lock %rwlock : memref<!sync.raw_rwlock>
    sync.raw_rwlock.read_unlock %rwlock : memref<!sync.raw_rwlock>
    return %locked : i1
  }
}
