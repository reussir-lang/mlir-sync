#ifndef SYNC_IR_SYNCTYPES_TD
#define SYNC_IR_SYNCTYPES_TD

include "Sync/IR/SyncDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class SyncType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<SyncDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def SyncRawMutexType
    : SyncType<"RawMutex", "raw_mutex", [MemRefElementTypeInterface]> {
  let summary = "Opaque raw mutex state";
  let description = [{
    `!sync.raw_mutex` models the storage for the portable mutex implementation
    in the runtime library. The lowering maps it exactly to an `i32`.
  }];
  let assemblyFormat = "";
}

def SyncRawRwLockType
    : SyncType<"RawRwLock", "raw_rwlock", [MemRefElementTypeInterface]> {
  let summary = "Opaque raw rwlock state";
  let description = [{
    `!sync.raw_rwlock` models the storage for the portable futex rwlock
    implementation in the runtime library. The LLVM lowering maps it to the two
    `i32` words used by the runtime state machine.
  }];
  let assemblyFormat = "";
}

def SyncOnceType
    : SyncType<"Once", "once", [MemRefElementTypeInterface]> {
  let summary = "Opaque once state";
  let description = [{
    `!sync.once` models the storage for the portable futex once
    implementation in the runtime library. The LLVM lowering maps it exactly
    to an `i32` state word.
  }];
  let assemblyFormat = "";
}

def SyncMutexType
    : SyncType<"Mutex", "mutex", [MemRefElementTypeInterface]> {
  let summary = "Mutex with inline payload storage";
  let description = [{
    `!sync.mutex<T>` stores a raw mutex header together with a payload of type
    `T`. The LLVM lowering maps it to a structure containing an `i32` mutex
    header followed by the lowered payload type.
  }];
  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def SyncRwLockType
    : SyncType<"RwLock", "rwlock", [MemRefElementTypeInterface]> {
  let summary = "Reader-writer lock with inline payload storage";
  let description = [{
    `!sync.rwlock<T>` stores a raw rwlock header together with a payload of type
    `T`. The LLVM lowering maps it to a structure containing the lowered raw
    rwlock header followed by the lowered payload type.
  }];
  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def SyncCombiningLockType
    : SyncType<"CombiningLock", "combining_lock",
               [MemRefElementTypeInterface]> {
  let summary = "Combining lock with inline payload storage";
  let description = [{
    `!sync.combining_lock<T>` models the runtime `Lock<T>` value used by the
    combining lock implementation. The LLVM lowering maps it to a structure
    whose first field is the raw lock header and whose second field is the
    lowered payload type.
  }];
  let parameters = (ins "Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def SyncCombiningLockNodeType
    : SyncType<"CombiningLockNode", "combining_lock_node",
               [MemRefElementTypeInterface]> {
  let summary = "Combining lock slow-path node";
  let description = [{
    `!sync.combining_lock_node<T1, T2, ...>` models the stack-allocated node
    passed to the runtime combining-lock slow path. The lowered layout stores
    the runtime node header, a raw pointer to the protected payload, and then
    the captured SSA values in order.
  }];
  let parameters = (ins ArrayRefParameter<"mlir::Type", "capture types">:$captureTypes);
  let assemblyFormat = "`<` $captureTypes `>`";
  let genVerifyDecl = 1;
}

#endif // SYNC_IR_SYNCTYPES_TD
