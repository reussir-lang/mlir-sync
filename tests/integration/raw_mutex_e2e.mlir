// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-sync-to-llvm --convert-cf-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 %S/raw_mutex_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @mutex_try_lock_returns_true() -> i1 {
    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    %locked = sync.raw_mutex.try_lock %mutex : memref<!sync.raw_mutex>
    return %locked : i1
  }

  func.func @mutex_relock_returns_false() -> i1 {
    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.lock %mutex : memref<!sync.raw_mutex>
    %locked = sync.raw_mutex.try_lock %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.unlock %mutex : memref<!sync.raw_mutex>
    return %locked : i1
  }

  func.func @mutex_lock_unlock_smoke() {
    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.lock %mutex : memref<!sync.raw_mutex>
    sync.raw_mutex.unlock %mutex : memref<!sync.raw_mutex>
    return
  }
}
