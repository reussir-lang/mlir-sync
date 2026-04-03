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

mlir::LogicalResult RwLockType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (!mlir::BaseMemRefType::isValidElementType(valueType))
    return emitError()
           << "rwlock payload type must be a valid memref element type";
  return mlir::success();
}

mlir::LogicalResult CombiningLockType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (!mlir::BaseMemRefType::isValidElementType(valueType))
    return emitError()
           << "combining lock payload type must be a valid memref element type";
  return mlir::success();
}

mlir::LogicalResult CombiningLockNodeType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> captureTypes) {
  for (mlir::Type captureType : captureTypes) {
    if (llvm::isa<mlir::FunctionType>(captureType))
      return emitError()
             << "combining lock node captures cannot include function types";
  }
  return mlir::success();
}

} // namespace mlir::sync
