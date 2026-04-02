#ifndef SYNC_IR_SYNCDIALECT_TD
#define SYNC_IR_SYNCDIALECT_TD

include "mlir/IR/OpBase.td"

def SyncDialect : Dialect {
  let name = "sync";
  let cppNamespace = "::mlir::sync";
  let useDefaultTypePrinterParser = 1;
}

#endif // SYNC_IR_SYNCDIALECT_TD
