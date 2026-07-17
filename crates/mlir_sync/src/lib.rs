#![no_std]

/// Declares a `#[no_mangle]` runtime entry point with the plain C ABI. The
/// compiler backend keeps its lock fast paths free of caller-saved register
/// spills by calling these through internal `preserve_most` trampolines it
/// generates itself, so the exported symbols stay on the stable C convention.
macro_rules! runtime_export {
    ($(#[$meta:meta])* $vis:vis unsafe fn $name:ident($($params:tt)*) $(-> $ret:ty)? $body:block) => {
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
