#![no_std]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

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
