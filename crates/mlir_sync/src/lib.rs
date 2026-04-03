#![no_std]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]
#![cfg_attr(feature = "nightly", allow(internal_features))]

pub mod combining_lock;
pub mod mutex;
pub mod rwlock;

#[cfg(not(any(test, miri)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(not(any(test, miri)))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
