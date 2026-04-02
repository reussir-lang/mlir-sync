#![no_std]
pub mod combining_lock;
pub mod mutex;

#[cfg(not(any(test, miri)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(all(target_os = "macos", not(any(test, miri))))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}
