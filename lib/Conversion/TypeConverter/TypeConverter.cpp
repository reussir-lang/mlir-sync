#include "Sync/Conversion/TypeConverter.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace mlir::sync {
namespace {

mlir::LowerToLLVMOptions getLowerOptions(mlir::ModuleOp moduleOp) {
  llvm::StringRef dataLayoutString;
  auto dataLayoutAttr = moduleOp->getAttrOfType<mlir::StringAttr>(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName());
  if (dataLayoutAttr)
    dataLayoutString = dataLayoutAttr.getValue();

  auto options = mlir::LowerToLLVMOptions(moduleOp.getContext());
  auto llvmDataLayout = llvm::DataLayout(dataLayoutString);
  if (llvmDataLayout.getPointerSizeInBits(0) == 32)
    options.overrideIndexBitwidth(32);
  options.dataLayout = llvmDataLayout;
  return options;
}

mlir::Type getOpaquePtrType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

mlir::LLVM::LLVMStructType getRawRwLockLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {mlir::IntegerType::get(context, 32),
                mlir::IntegerType::get(context, 32)});
}

mlir::LLVM::LLVMStructType getCombiningRawLockLLVMType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, {getOpaquePtrType(context), mlir::IntegerType::get(context, 8)});
}

} // namespace

void populateSyncToLLVMTypeConversions(mlir::LLVMTypeConverter &converter) {
  converter.addConversion([](RawMutexType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  converter.addConversion([](RawRwLockType type) -> mlir::Type {
    return getRawRwLockLLVMType(type.getContext());
  });
  converter.addConversion([](OnceType type) -> mlir::Type {
    return mlir::IntegerType::get(type.getContext(), 32);
  });
  converter.addConversion([&converter](MutexType type) -> mlir::Type {
    auto i32Type = mlir::IntegerType::get(type.getContext(), 32);
    mlir::Type payloadType = converter.convertType(type.getValueType());
    if (!payloadType)
      return {};
    return mlir::LLVM::LLVMStructType::getLiteral(
        type.getContext(), {i32Type, payloadType});
  });
  converter.addConversion([&converter](RwLockType type) -> mlir::Type {
    mlir::Type payloadType = converter.convertType(type.getValueType());
    if (!payloadType)
      return {};
    return mlir::LLVM::LLVMStructType::getLiteral(
        type.getContext(), {getRawRwLockLLVMType(type.getContext()), payloadType});
  });
  converter.addConversion([&converter](CombiningLockType type) -> mlir::Type {
    mlir::Type payloadType = converter.convertType(type.getValueType());
    if (!payloadType)
      return {};
    return mlir::LLVM::LLVMStructType::getLiteral(
        type.getContext(),
        {getCombiningRawLockLLVMType(type.getContext()), payloadType});
  });
  converter.addConversion(
      [&converter](CombiningLockNodeType type) -> mlir::Type {
        llvm::SmallVector<mlir::Type> fields{
            mlir::IntegerType::get(type.getContext(), 32),
            getOpaquePtrType(type.getContext()),
            getOpaquePtrType(type.getContext()),
            getOpaquePtrType(type.getContext())};
        for (mlir::Type captureType : type.getCaptureTypes()) {
          mlir::Type loweredCapture = converter.convertType(captureType);
          if (!loweredCapture)
            return mlir::Type{};
          fields.push_back(loweredCapture);
        }
        return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), fields);
      });
}

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp moduleOp)
    : mlir::LLVMTypeConverter(moduleOp.getContext(), getLowerOptions(moduleOp)) {
  populateSyncToLLVMTypeConversions(*this);
}

} // namespace mlir::sync
