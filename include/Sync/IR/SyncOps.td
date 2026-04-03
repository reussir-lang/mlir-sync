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

class SyncCombiningLockOp<string mnemonic, list<Trait> traits = []>
    : SyncOp<"combining_lock." # mnemonic, traits>;

def SyncMutexInitOp : SyncMutexOp<"init"> {
  let summary = "Initialize a typed mutex";
  let description = [{
    `sync.mutex.init` initializes the raw mutex header of a zero-ranked memref
    containing `!sync.mutex<T>` to the unlocked state and optionally stores an
    initial payload value.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$mutex,
                       Optional<AnyType>:$initialValue);
  let assemblyFormat = "$mutex attr-dict `:` type($mutex) (`,` $initialValue^ `:` type($initialValue))?";
  let hasVerifier = 1;
}

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

def SyncYieldOp : SyncOp<"yield", [ParentOneOf<["SyncMutexCriticalSectionOp",
                                                "SyncCombiningLockCriticalSectionOp"]>,
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

def SyncCombiningLockInitOp : SyncCombiningLockOp<"init"> {
  let summary = "Initialize a combining lock";
  let description = [{
    `sync.combining_lock.init` initializes a zero-ranked memref containing
    `!sync.combining_lock<T>` and optionally stores an initial payload value.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$lock,
                       Optional<AnyType>:$initialValue);
  let assemblyFormat = "$lock attr-dict `:` type($lock) (`,` $initialValue^ `:` type($initialValue))?";
  let hasVerifier = 1;
}

def SyncCombiningLockHasTailOp
    : SyncCombiningLockOp<"has_tail"> {
  let summary = "Check whether the combining lock has a tail";
  let description = [{
    `sync.combining_lock.has_tail` returns true when the runtime tail pointer is
    non-null.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$lock);
  let results = (outs I1:$hasTail);
  let assemblyFormat = "$lock attr-dict `:` type($lock)";
  let hasVerifier = 1;
}

def SyncCombiningLockTryAcquireOp
    : SyncCombiningLockOp<"try_acquire"> {
  let summary = "Attempt to acquire the combining lock status bit";
  let description = [{
    `sync.combining_lock.try_acquire` performs the acquire test-and-set on the
    runtime status flag and returns true on success.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$lock);
  let results = (outs I1:$acquired);
  let assemblyFormat = "$lock attr-dict `:` type($lock)";
  let hasVerifier = 1;
}

def SyncCombiningLockReleaseOp : SyncCombiningLockOp<"release"> {
  let summary = "Release a combining lock fast path";
  let description = [{
    `sync.combining_lock.release` stores `false` into the lock status with
    release ordering.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$lock);
  let assemblyFormat = "$lock attr-dict `:` type($lock)";
  let hasVerifier = 1;
}

def SyncCombiningLockGetPayloadOp : SyncCombiningLockOp<"get_payload"> {
  let summary = "Project a combining lock or node payload";
  let description = [{
    `sync.combining_lock.get_payload` projects either a combining lock payload
    or the payload-plus-capture tuple materialized in a slow-path node.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$source);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$source attr-dict `:` type($source) `->` type($results)";
  let hasVerifier = 1;
}

def SyncCombiningLockRecoverOp : SyncCombiningLockOp<"recover"> {
  let summary = "Recover payload and captures from a raw slow-path node pointer";
  let description = [{
    `sync.combining_lock.recover` is an internal helper used by outlined
    slow-path wrappers to reconstruct the payload memref and captured SSA values
    from the raw callback pointer.
  }];
  let arguments = (ins AnyType:$node);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def SyncCombiningLockCaptureOp : SyncCombiningLockOp<"capture"> {
  let summary = "Materialize a combining lock slow-path node";
  let description = [{
    `sync.combining_lock.capture` creates a stack-allocated slow-path node that
    stores a callback, a payload pointer, and the captured SSA values. It
    returns both a zero-ranked memref view of the node and the raw stack pointer
    so the lifetime can be ended on the original allocation value.
  }];
  let arguments = (ins AnyType:$callee,
                       MemRefRankOf<[AnyType], [0]>:$payload,
                       Variadic<AnyType>:$captures);
  let results = (outs MemRefRankOf<[AnyType], [0]>:$node,
                      AnyType:$rawNode);
  let hasVerifier = 1;
}

def SyncCombiningLockCaptureEndOp : SyncCombiningLockOp<"capture_end"> {
  let summary = "End the lifetime of a captured slow-path node";
  let description = [{
    `sync.combining_lock.capture_end` is an internal helper that marks the end
    of the stack lifetime created by `sync.combining_lock.capture`.
  }];
  let arguments = (ins AnyType:$rawNode);
  let assemblyFormat = "$rawNode attr-dict `:` type($rawNode)";
  let hasVerifier = 1;
}

def SyncCombiningLockCriticalSectionOp : SyncCombiningLockOp<"critical_section", [
      RecursiveMemoryEffects,
      SingleBlockImplicitTerminator<"SyncYieldOp">
    ]> {
  let summary = "Execute a combining-lock critical section";
  let description = [{
    `sync.combining_lock.critical_section` tries the combining-lock fast path
    inline and outlines a slow-path callback when contention is observed.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [0]>:$lock,
                       OptionalAttr<I64Attr>:$combine_limit);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$lock attr-dict `:` type($lock) regions";
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

#endif // SYNC_IR_SYNCOPS_TD
