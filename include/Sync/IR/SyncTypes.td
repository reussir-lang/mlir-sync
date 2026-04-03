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

#endif // SYNC_IR_SYNCTYPES_TD
