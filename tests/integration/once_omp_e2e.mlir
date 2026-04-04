// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 -fopenmp %S/once_omp_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @once_parallel_runs_once() -> i32 {
    %num_threads = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %delay = arith.constant 10000 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %once = memref.alloca() : memref<!sync.once>
    %counter = memref.alloca() : memref<i32>
    sync.once.init %once : memref<!sync.once>
    memref.store %c0_i32, %counter[] : memref<i32>

    omp.parallel num_threads(%num_threads : i32) {
      sync.once.execute %once : memref<!sync.once> {
        scf.for %i = %c0 to %delay step %c1 {
        }
        %value = memref.load %counter[] : memref<i32>
        %next = arith.addi %value, %c1_i32 : i32
        memref.store %next, %counter[] : memref<i32>
        sync.yield
      }
      omp.terminator
    }

    %result = memref.load %counter[] : memref<i32>
    return %result : i32
  }
}
