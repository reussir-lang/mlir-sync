#pragma once

#ifndef SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
#define SYNC_CONVERSION_CONVERTSYNCTOLLVM_H

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Sync/Conversion/TypeConverter.h"
#include "Sync/IR/SyncOps.h"

namespace mlir::sync {

/// Calling-convention marker for functions the sync lowering creates: the
/// func-to-llvm conversion pins every function it converts to the C calling
/// convention, so the requested convention rides this discardable attribute
/// and is folded into the converted `llvm.func` by the sync patterns.
inline constexpr ::llvm::StringLiteral kSyncCConvAttr = "sync.cconv";

/// Same contract for LLVM function attributes: the requested `passthrough`
/// strings ride this marker and are folded into the converted `llvm.func`'s
/// inherent attribute.
inline constexpr ::llvm::StringLiteral kSyncPassthroughAttr =
    "sync.passthrough";

#define GEN_PASS_DECL_CONVERTSYNCTOLLVMPASS
#include "Sync/Conversion/Passes.h.inc"

void configureConvertSyncToLLVMConversionLegality(
    mlir::ConversionTarget &target);

void populateConvertSyncToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns);

void registerConvertSyncToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace mlir::sync

#endif // SYNC_CONVERSION_CONVERTSYNCTOLLVM_H
