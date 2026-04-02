import lit.formats
import os

config.name = "Sync"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, "test")

config.substitutions.append((r"%clang", config.clang_path))
config.substitutions.append((r"%sync-opt", config.sync_opt_path))
config.substitutions.append((r"%mlir-translate", config.mlir_translate_path))
config.substitutions.append((r"%sync-runtime-lib", config.sync_runtime_lib_path))
config.substitutions.append((r"%FileCheck", config.filecheck_path))
config.substitutions.append((r"%not", config.not_path))
