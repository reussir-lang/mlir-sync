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

} // namespace

void populateSyncToLLVMTypeConversions(mlir::LLVMTypeConverter &converter) {
  converter.addConversion([](RawMutexType type) -> mlir::Type {
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
}

LLVMTypeConverter::LLVMTypeConverter(mlir::ModuleOp moduleOp)
    : mlir::LLVMTypeConverter(moduleOp.getContext(), getLowerOptions(moduleOp)) {
  populateSyncToLLVMTypeConversions(*this);
}

} // namespace mlir::sync
