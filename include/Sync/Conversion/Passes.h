#pragma once

#ifndef SYNC_CONVERSION_PASSES_H
#define SYNC_CONVERSION_PASSES_H

#include <mlir/Pass/Pass.h>

namespace mlir::sync {

#define GEN_PASS_DECL
#include "Sync/Conversion/Passes.h.inc"

} // namespace mlir::sync

#endif // SYNC_CONVERSION_PASSES_H
