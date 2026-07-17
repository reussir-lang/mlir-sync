#![no_std]
#![cfg_attr(feature = "nightly", feature(core_intrinsics, rust_cold_cc))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

/// Declares a `#[no_mangle]` runtime entry point. With the `nightly` feature
/// the symbol is exported with the `rust-cold` ABI, which lowers to the LLVM
/// `preserve_most` calling convention expected by the compiler backend when
/// it is built with `MLIR_SYNC_ENABLE_NIGHTLY_FEATURE`; otherwise the symbol
/// falls back to the plain C ABI.
macro_rules! runtime_export {
    ($(#[$meta:meta])* $vis:vis unsafe fn $name:ident($($params:tt)*) $(-> $ret:ty)? $body:block) => {
        #[cfg(feature = "nightly")]
        $(#[$meta])*
        #[unsafe(no_mangle)]
        #[cold]
        $vis unsafe extern "rust-cold" fn $name($($params)*) $(-> $ret)? $body

        #[cfg(not(feature = "nightly"))]
        $(#[$meta])*
        #[unsafe(no_mangle)]
        #[cold]
        $vis unsafe extern "C" fn $name($($params)*) $(-> $ret)? $body
    };
}
pub(crate) use runtime_export;

pub mod combining_lock;
pub mod mutex;
pub mod once;
pub mod rwlock;

// Only standalone runtime builds (e.g. the staticlib linked into the MLIR
// integration tests) provide a panic handler; when used as a Rust library
// dependency the final binary supplies one (usually via std).
#[cfg(all(feature = "panic-handler", not(any(test, miri))))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(feature = "panic-handler", not(any(test, miri))))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
