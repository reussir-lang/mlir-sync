// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 %S/once_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @once_initially_incomplete() -> i1 {
    %once = memref.alloca() : memref<!sync.once>
    sync.once.init %once : memref<!sync.once>
    %completed = sync.once.completed %once : memref<!sync.once>
    return %completed : i1
  }

  func.func @once_execute_marks_complete() -> i1 {
    %once = memref.alloca() : memref<!sync.once>
    sync.once.init %once : memref<!sync.once>
    sync.once.execute %once : memref<!sync.once> {
      sync.yield
    }
    %after = sync.once.completed %once : memref<!sync.once>
    return %after : i1
  }

  func.func @once_execute_runs_once() -> i32 {
    %once = memref.alloca() : memref<!sync.once>
    %counter = memref.alloca() : memref<i32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    sync.once.init %once : memref<!sync.once>
    memref.store %c0, %counter[] : memref<i32>

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

    %result = memref.load %counter[] : memref<i32>
    return %result : i32
  }
}
