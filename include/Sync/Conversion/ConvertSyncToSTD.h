#pragma once

#ifndef SYNC_CONVERSION_CONVERTSYNCTOSTD_H
#define SYNC_CONVERSION_CONVERTSYNCTOSTD_H

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::sync {

#define GEN_PASS_DECL_CONVERTSYNCTOSTDPASS
#include "Sync/Conversion/Passes.h.inc"

void populateConvertSyncToSTDPatterns(mlir::RewritePatternSet &patterns);

} // namespace mlir::sync

#endif // SYNC_CONVERSION_CONVERTSYNCTOSTD_H
