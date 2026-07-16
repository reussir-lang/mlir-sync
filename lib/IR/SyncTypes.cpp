#include "Sync/IR/SyncTypes.h"

#include <algorithm>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Alignment.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/TypeSize.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace mlir::sync {
namespace {

struct StorageLayout {
  llvm::TypeSize size;
  llvm::Align alignment;
};

StorageLayout getTypeLayout(mlir::Type type,
                            const mlir::DataLayout &dataLayout) {
  return {dataLayout.getTypeSize(type),
          llvm::Align(dataLayout.getTypeABIAlignment(type))};
}

StorageLayout getStructLayout(llvm::ArrayRef<StorageLayout> fields) {
  // Apply ordinary ABI struct padding, including tail padding. Nested runtime
  // headers are passed as one field so their tail padding remains observable.
  uint64_t size = 0;
  llvm::Align alignment{1};
  for (StorageLayout field : fields) {
    if (!field.size.isFixed())
      llvm::report_fatal_error("sync storage fields must have a fixed size");
    size = llvm::alignTo(size, field.alignment);
    size += field.size.getFixedValue();
    alignment = std::max(alignment, field.alignment);
  }
  return {llvm::TypeSize::getFixed(llvm::alignTo(size, alignment)), alignment};
}

StorageLayout getStructLayout(llvm::ArrayRef<mlir::Type> fields,
                              const mlir::DataLayout &dataLayout) {
  llvm::SmallVector<StorageLayout> fieldLayouts;
  fieldLayouts.reserve(fields.size());
  for (mlir::Type field : fields)
    fieldLayouts.push_back(getTypeLayout(field, dataLayout));
  return getStructLayout(fieldLayouts);
}

mlir::IntegerType getIntegerType(mlir::MLIRContext *context,
                                 unsigned bitWidth) {
  return mlir::IntegerType::get(context, bitWidth);
}

StorageLayout getStorageLayout(RawMutexType type,
                               const mlir::DataLayout &dataLayout) {
  return getTypeLayout(getIntegerType(type.getContext(), 32), dataLayout);
}

StorageLayout getStorageLayout(RawRwLockType type,
                               const mlir::DataLayout &dataLayout) {
  auto i32Type = getIntegerType(type.getContext(), 32);
  return getStructLayout({i32Type, i32Type}, dataLayout);
}

StorageLayout getStorageLayout(OnceType type,
                               const mlir::DataLayout &dataLayout) {
  return getTypeLayout(getIntegerType(type.getContext(), 32), dataLayout);
}

StorageLayout getStorageLayout(MutexType type,
                               const mlir::DataLayout &dataLayout) {
  return getStructLayout(
      {getTypeLayout(RawMutexType::get(type.getContext()), dataLayout),
       getTypeLayout(type.getValueType(), dataLayout)});
}

StorageLayout getStorageLayout(RwLockType type,
                               const mlir::DataLayout &dataLayout) {
  return getStructLayout(
      {getTypeLayout(RawRwLockType::get(type.getContext()), dataLayout),
       getTypeLayout(type.getValueType(), dataLayout)});
}

StorageLayout getCombiningRawLockLayout(mlir::MLIRContext *context,
                                        const mlir::DataLayout &dataLayout) {
  // This is the `{ptr, i8}` raw header used by the LLVM type converter.
  auto pointerType = mlir::LLVM::LLVMPointerType::get(context);
  auto i8Type = getIntegerType(context, 8);
  return getStructLayout({pointerType, i8Type}, dataLayout);
}

StorageLayout getStorageLayout(CombiningLockType type,
                               const mlir::DataLayout &dataLayout) {
  return getStructLayout(
      {getCombiningRawLockLayout(type.getContext(), dataLayout),
       getTypeLayout(type.getValueType(), dataLayout)});
}

StorageLayout getStorageLayout(CombiningLockNodeType type,
                               const mlir::DataLayout &dataLayout) {
  auto pointerType = mlir::LLVM::LLVMPointerType::get(type.getContext());
  llvm::SmallVector<mlir::Type> fields{getIntegerType(type.getContext(), 32),
                                       pointerType, pointerType, pointerType};
  llvm::append_range(fields, type.getCaptureTypes());
  return getStructLayout(fields, dataLayout);
}

} // namespace

#define DEFINE_SYNC_DATA_LAYOUT(TypeName)                                      \
  llvm::TypeSize TypeName::getTypeSizeInBits(                                  \
      const mlir::DataLayout &dataLayout,                                      \
      [[maybe_unused]] mlir::DataLayoutEntryListRef params) const {            \
    StorageLayout layout = getStorageLayout(*this, dataLayout);                \
    return layout.size * 8;                                                    \
  }                                                                            \
                                                                               \
  uint64_t TypeName::getABIAlignment(                                          \
      const mlir::DataLayout &dataLayout,                                      \
      [[maybe_unused]] mlir::DataLayoutEntryListRef params) const {            \
    return getStorageLayout(*this, dataLayout).alignment.value();              \
  }

DEFINE_SYNC_DATA_LAYOUT(RawMutexType)
DEFINE_SYNC_DATA_LAYOUT(RawRwLockType)
DEFINE_SYNC_DATA_LAYOUT(OnceType)
DEFINE_SYNC_DATA_LAYOUT(MutexType)
DEFINE_SYNC_DATA_LAYOUT(RwLockType)
DEFINE_SYNC_DATA_LAYOUT(CombiningLockType)
DEFINE_SYNC_DATA_LAYOUT(CombiningLockNodeType)

#undef DEFINE_SYNC_DATA_LAYOUT

mlir::LogicalResult
MutexType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  mlir::Type valueType) {
  if (!mlir::BaseMemRefType::isValidElementType(valueType))
    return emitError()
           << "mutex payload type must be a valid memref element type";
  return mlir::success();
}

mlir::LogicalResult
RwLockType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
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
