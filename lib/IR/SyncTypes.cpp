#include "Sync/IR/SyncTypes.h"

#include <mlir/IR/BuiltinTypes.h>

namespace mlir::sync {

mlir::LogicalResult MutexType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (!mlir::BaseMemRefType::isValidElementType(valueType))
    return emitError() << "mutex payload type must be a valid memref element type";
  return mlir::success();
}

} // namespace mlir::sync
