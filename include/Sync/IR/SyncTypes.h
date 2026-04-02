#pragma once

#ifndef SYNC_IR_SYNCTYPES_H
#define SYNC_IR_SYNCTYPES_H

#include <mlir/IR/BuiltinTypes.h>

#include "Sync/IR/SyncDialect.h"

#define GET_TYPEDEF_CLASSES
#include "Sync/IR/SyncTypes.h.inc"

#endif // SYNC_IR_SYNCTYPES_H
