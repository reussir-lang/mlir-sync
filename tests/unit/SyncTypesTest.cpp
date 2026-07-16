#include <cstdint>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Parser/Parser.h>

#include "Sync/Conversion/TypeConverter.h"
#include "Sync/IR/SyncDialect.h"
#include "Sync/IR/SyncTypes.h"

namespace {

struct ExpectedLayout {
  llvm::StringRef name;
  mlir::Type type;
  uint64_t size;
  uint64_t abiAlignment;
  uint64_t preferredAlignment;
};

bool checkLayout(const mlir::DataLayout &dataLayout,
                 mlir::sync::LLVMTypeConverter &typeConverter,
                 const ExpectedLayout &expected) {
  llvm::TypeSize size = dataLayout.getTypeSize(expected.type);
  uint64_t abiAlignment = dataLayout.getTypeABIAlignment(expected.type);
  uint64_t preferredAlignment =
      dataLayout.getTypePreferredAlignment(expected.type);
  bool success = size.isFixed() && size.getFixedValue() == expected.size &&
                 abiAlignment == expected.abiAlignment &&
                 preferredAlignment == expected.preferredAlignment;

  if (!success) {
    llvm::errs() << expected.name << " (" << expected.type
                 << "): expected size " << expected.size << " and alignment "
                 << expected.abiAlignment << '/' << expected.preferredAlignment
                 << ", got size ";
    if (size.isFixed())
      llvm::errs() << size.getFixedValue();
    else
      llvm::errs() << "scalable(" << size.getKnownMinValue() << ')';
    llvm::errs() << " and alignment " << abiAlignment << '/'
                 << preferredAlignment << '\n';
  }

  mlir::Type convertedType = typeConverter.convertType(expected.type);
  if (!convertedType) {
    llvm::errs() << "failed to convert " << expected.name << " ("
                 << expected.type << ") to LLVM\n";
    return false;
  }
  llvm::TypeSize convertedSize = dataLayout.getTypeSize(convertedType);
  uint64_t convertedABIAlignment =
      dataLayout.getTypeABIAlignment(convertedType);
  uint64_t convertedPreferredAlignment =
      dataLayout.getTypePreferredAlignment(convertedType);
  if (size != convertedSize || abiAlignment != convertedABIAlignment ||
      preferredAlignment != convertedPreferredAlignment) {
    llvm::errs() << expected.name << " (" << expected.type
                 << ") disagrees with its LLVM storage type " << convertedType
                 << ": sync alignment " << abiAlignment << '/'
                 << preferredAlignment << ", LLVM alignment "
                 << convertedABIAlignment << '/' << convertedPreferredAlignment
                 << '\n';
    success = false;
  }
  return success;
}

bool checkLayouts(mlir::MLIRContext &context, llvm::StringRef moduleText,
                  llvm::ArrayRef<ExpectedLayout> expectedLayouts) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleText, &context);
  if (!module) {
    llvm::errs() << "failed to parse test data layout\n";
    return false;
  }

  mlir::DataLayout dataLayout(*module);
  mlir::sync::LLVMTypeConverter typeConverter(*module);
  bool success = true;
  for (const ExpectedLayout &expected : expectedLayouts)
    success &= checkLayout(dataLayout, typeConverter, expected);
  return success;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::DLTIDialect, mlir::LLVM::LLVMDialect,
                      mlir::sync::SyncDialect>();

  auto i8Type = mlir::IntegerType::get(&context, 8);
  auto i64Type = mlir::IntegerType::get(&context, 64);
  auto rawMutexType = mlir::sync::RawMutexType::get(&context);
  auto rawRwLockType = mlir::sync::RawRwLockType::get(&context);
  auto onceType = mlir::sync::OnceType::get(&context);
  auto mutexI8Type = mlir::sync::MutexType::get(&context, i8Type);
  auto mutexI64Type = mlir::sync::MutexType::get(&context, i64Type);
  auto rwLockI8Type = mlir::sync::RwLockType::get(&context, i8Type);
  auto rwLockI64Type = mlir::sync::RwLockType::get(&context, i64Type);
  auto combiningI8Type = mlir::sync::CombiningLockType::get(&context, i8Type);
  auto combiningI64Type = mlir::sync::CombiningLockType::get(&context, i64Type);
  auto emptyNodeType = mlir::sync::CombiningLockNodeType::get(&context, {});
  llvm::SmallVector<mlir::Type> captureTypes{i8Type, i64Type};
  auto captureNodeType =
      mlir::sync::CombiningLockNodeType::get(&context, captureTypes);
  auto nestedMutexType = mlir::sync::MutexType::get(&context, combiningI8Type);

  constexpr llvm::StringLiteral layout64 = R"mlir(
    module attributes {
      dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
        #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
        #dlti.dl_entry<i64, dense<[64, 128]> : vector<2xi64>>,
        #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 128, 64]> : vector<4xi64>>
      >
    } {
    }
  )mlir";
  llvm::SmallVector<ExpectedLayout> expected64{
      {"raw mutex", rawMutexType, 4, 4, 4},
      {"raw rwlock", rawRwLockType, 8, 4, 4},
      {"once", onceType, 4, 4, 4},
      {"mutex<i8>", mutexI8Type, 8, 4, 4},
      {"mutex<i64>", mutexI64Type, 16, 8, 8},
      {"rwlock<i8>", rwLockI8Type, 12, 4, 4},
      {"rwlock<i64>", rwLockI64Type, 16, 8, 8},
      {"combining_lock<i8>", combiningI8Type, 24, 8, 8},
      {"combining_lock<i64>", combiningI64Type, 24, 8, 8},
      {"combining_lock_node<>", emptyNodeType, 32, 8, 8},
      {"combining_lock_node<i8, i64>", captureNodeType, 48, 8, 8},
      {"mutex<combining_lock<i8>>", nestedMutexType, 32, 8, 8},
  };

  constexpr llvm::StringLiteral layout32 = R"mlir(
    module attributes {
      dlti.dl_spec = #dlti.dl_spec<
        #dlti.dl_entry<i8, dense<8> : vector<2xi64>>,
        #dlti.dl_entry<i32, dense<32> : vector<2xi64>>,
        #dlti.dl_entry<i64, dense<[64, 128]> : vector<2xi64>>,
        #dlti.dl_entry<!llvm.ptr, dense<[32, 32, 64, 32]> : vector<4xi64>>
      >
    } {
    }
  )mlir";
  llvm::SmallVector<ExpectedLayout> expected32{
      {"32-bit combining_lock<i8>", combiningI8Type, 12, 4, 4},
      {"32-bit combining_lock<i64>", combiningI64Type, 16, 8, 8},
      {"32-bit combining_lock_node<>", emptyNodeType, 16, 4, 4},
      {"32-bit combining_lock_node<i8, i64>", captureNodeType, 32, 8, 8},
  };

  return checkLayouts(context, layout64, expected64) &&
                 checkLayouts(context, layout32, expected32)
             ? 0
             : 1;
}
