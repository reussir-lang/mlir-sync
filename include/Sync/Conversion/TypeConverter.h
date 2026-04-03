#pragma once

#ifndef SYNC_CONVERSION_TYPECONVERTER_H
#define SYNC_CONVERSION_TYPECONVERTER_H

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/BuiltinOps.h>

#include "Sync/IR/SyncTypes.h"

namespace mlir::sync {

void populateSyncToLLVMTypeConversions(mlir::LLVMTypeConverter &converter);

class LLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  explicit LLVMTypeConverter(mlir::ModuleOp moduleOp);
};

} // namespace mlir::sync

#endif // SYNC_CONVERSION_TYPECONVERTER_H
