#ifndef SYNC_IR_SYNCOPS_TD
#define SYNC_IR_SYNCOPS_TD

include "Sync/IR/SyncDialect.td"
include "Sync/IR/SyncTypes.td"
include "mlir/IR/OpBase.td"

class SyncOp<string mnemonic, list<Trait> traits = []>
    : Op<SyncDialect, mnemonic, traits>;

class SyncRawMutexOp<string mnemonic, list<Trait> traits = []>
    : SyncOp<"raw_mutex." # mnemonic, traits>;

def SyncRawMutexInitOp : SyncRawMutexOp<"init"> {
  let summary = "Initialize a raw mutex";
  let description = [{
    `sync.raw_mutex.init` initializes a zero-ranked memref containing a
    `!sync.raw_mutex` to the unlocked state.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex)";
  let hasVerifier = 1;
}

def SyncRawMutexTryLockOp : SyncRawMutexOp<"try_lock"> {
  let summary = "Try to acquire a raw mutex";
  let description = [{
    `sync.raw_mutex.try_lock` attempts to acquire a raw mutex and returns `true`
    on success.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let results = (outs I1:$acquired);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex)";
  let hasVerifier = 1;
}

def SyncRawMutexLockOp : SyncRawMutexOp<"lock"> {
  let summary = "Acquire a raw mutex";
  let description = [{
    `sync.raw_mutex.lock` acquires a raw mutex, using the runtime slow path when
    contention is observed.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex)";
  let hasVerifier = 1;
}

def SyncRawMutexUnlockFastOp : SyncRawMutexOp<"unlock_fast"> {
  let summary = "Release a raw mutex fast path";
  let description = [{
    `sync.raw_mutex.unlock_fast` releases a raw mutex and returns `true` when
    the slow path wake-up is required.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let results = (outs I1:$needsWake);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex)";
  let hasVerifier = 1;
}

def SyncRawMutexUnlockOp : SyncRawMutexOp<"unlock"> {
  let summary = "Release a raw mutex";
  let description = [{
    `sync.raw_mutex.unlock` releases a raw mutex and wakes a waiter via the
    runtime slow path when the mutex was contended.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex)";
  let hasVerifier = 1;
}

#endif // SYNC_IR_SYNCOPS_TD
