#![no_std]
pub mod combining_lock;
pub mod mutex;

#[cfg(not(any(test, miri)))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
