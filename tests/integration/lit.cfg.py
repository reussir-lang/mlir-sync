import lit.formats
import os
import platform

config.name = "Sync"
# lit's internal shell mishandles Windows extended-length temp paths; keep the
# external (msys) shell there and the internal shell everywhere else.
if platform.system() == "Windows":
    config.test_format = lit.formats.ShTest(force_execute_external=True)
else:
    config.test_format = lit.formats.ShTest()

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_output_root, "test")

# Toolchain wrappers that inject flags through the environment (e.g. Nix's
# clang wrapper reading NIX_CFLAGS_COMPILE/NIX_LDFLAGS, which are only honored
# together with the wrapper's NIX_CC_WRAPPER_* role markers) need these
# variables to survive lit's environment scrubbing.
for _var, _value in os.environ.items():
    if _var.startswith("NIX_") or _var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        config.environment[_var] = _value

sync_runtime_link = config.sync_runtime_lib_path
if getattr(config, "host_system", "") == "Windows":
    sync_runtime_link += " -lsynchronization"

config.substitutions.append((r"%clang", config.clang_path))
config.substitutions.append((r"%sync-opt", config.sync_opt_path))
config.substitutions.append((r"%mlir-translate", config.mlir_translate_path))
config.substitutions.append((r"%sync-runtime-lib", sync_runtime_link))
config.substitutions.append((r"%FileCheck", config.filecheck_path))
config.substitutions.append((r"%not", config.not_path))
