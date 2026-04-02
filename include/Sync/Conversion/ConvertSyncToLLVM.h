#pragma once

#ifndef SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
#define SYNC_CONVERSION_CONVERTSYNCTOLLVM_H

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Sync/Conversion/TypeConverter.h"
#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

#define GEN_PASS_DECL_CONVERTSYNCTOLLVMPASS
#include "Sync/Conversion/Passes.h.inc"

void populateConvertSyncToLLVMConversionPatterns(
    LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

} // namespace mlir::sync

#endif // SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
