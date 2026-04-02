#pragma once

#ifndef SYNC_IR_SYNCOPS_H
#define SYNC_IR_SYNCOPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include "Sync/IR/SyncTypes.h"

#define GET_OP_CLASSES
#include "Sync/IR/SyncOps.h.inc"

#endif // SYNC_IR_SYNCOPS_H
