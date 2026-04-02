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

#endif // SYNC_IR_SYNCTYPES_TD
