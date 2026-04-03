// RUN: %sync-opt %s --convert-sync-to-std --convert-scf-to-cf --convert-to-llvm > %t
// RUN: %mlir-translate --mlir-to-llvmir %t | %clang -Wno-override-module -O0 -c -x ir -o %t.o -
// RUN: %clang -std=c11 -O0 -fopenmp %S/raw_mutex_omp_e2e_main.c %t.o %sync-runtime-lib -o %t.exe
// RUN: %t.exe

module {
  func.func @mutex_parallel_increment() -> i32 {
    %num_threads = arith.constant 4 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %iters = arith.constant 1000 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %mutex = memref.alloca() : memref<!sync.raw_mutex>
    %counter = memref.alloca() : memref<i32>
    sync.raw_mutex.init %mutex : memref<!sync.raw_mutex>
    memref.store %c0_i32, %counter[] : memref<i32>

    omp.parallel num_threads(%num_threads : i32) {
      scf.for %i = %c0 to %iters step %c1 {
        sync.raw_mutex.lock %mutex : memref<!sync.raw_mutex>
        %value = memref.load %counter[] : memref<i32>
        %next = arith.addi %value, %c1_i32 : i32
        memref.store %next, %counter[] : memref<i32>
        sync.raw_mutex.unlock %mutex : memref<!sync.raw_mutex>
      }
      omp.terminator
    }

    %result = memref.load %counter[] : memref<i32>
    return %result : i32
  }
}
