#ifndef SYNC_IR_SYNCOPS_TD
#define SYNC_IR_SYNCOPS_TD

include "Sync/IR/SyncDialect.td"
include "Sync/IR/SyncTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SyncOp<string mnemonic, list<Trait> traits = []>
    : Op<SyncDialect, mnemonic, traits>;

class SyncRawMutexOp<string mnemonic, list<Trait> traits = []>
    : SyncOp<"raw_mutex." # mnemonic, traits>;

class SyncMutexOp<string mnemonic, list<Trait> traits = []>
    : SyncOp<"mutex." # mnemonic, traits>;

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

def SyncYieldOp : SyncOp<"yield", [HasParent<"SyncMutexCriticalSectionOp">,
                                   Terminator]> {
  let summary = "Terminate a sync region";
  let description = [{
    `sync.yield` terminates the body region of `sync.mutex.critical_section`.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
}

def SyncMutexGetRawMutexOp : SyncMutexOp<"get_raw_mutex"> {
  let summary = "Project the raw mutex header";
  let description = [{
    `sync.mutex.get_raw_mutex` projects a zero-ranked memref containing
    `!sync.mutex<T>` to a zero-ranked memref containing the raw mutex header.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let results = (outs MemRefRankOf<[AnyType], [0]>:$rawMutex);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex) `->` type($rawMutex)";
  let hasVerifier = 1;
}

def SyncMutexGetPayloadOp : SyncMutexOp<"get_payload"> {
  let summary = "Project the inline payload";
  let description = [{
    `sync.mutex.get_payload` projects a zero-ranked memref containing
    `!sync.mutex<T>` to a zero-ranked memref containing the payload `T`.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let results = (outs MemRefRankOf<[AnyType], [0]>:$payload);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex) `->` type($payload)";
  let hasVerifier = 1;
}

def SyncMutexCriticalSectionOp : SyncMutexOp<"critical_section", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"SyncYieldOp">
    ]> {
  let summary = "Execute a critical section over a mutex payload";
  let description = [{
    `sync.mutex.critical_section` acquires the mutex, executes the region with a
    zero-ranked memref to the payload, and releases the mutex on exit.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex) (`->` type($results)^)? regions";
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // SYNC_IR_SYNCOPS_TD
