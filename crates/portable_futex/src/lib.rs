#![no_std]
#![cfg_attr(
    all(feature = "nightly", target_family = "wasm"),
    feature(stdarch_wasm_atomic_wait)
)]

#[cfg(any(test, miri))]
extern crate std;

use core::ops::Deref;
use core::ptr::NonNull;
use core::sync::atomic::{AtomicU32, Ordering};

#[cfg(all(
    feature = "nightly",
    not(miri),
    target_family = "wasm",
    target_feature = "atomics",
    target_arch = "wasm32"
))]
use core::arch::wasm32 as wasm;
#[cfg(all(
    feature = "nightly",
    not(miri),
    target_family = "wasm",
    target_feature = "atomics",
    target_arch = "wasm64"
))]
use core::arch::wasm64 as wasm;
#[cfg(all(not(miri), target_os = "macos"))]
use core::ffi::{c_int, c_void};
#[cfg(all(not(miri), target_os = "linux"))]
use linux_raw_sys::general::{FUTEX_PRIVATE_FLAG, FUTEX_WAIT, FUTEX_WAKE};
#[cfg(miri)]
use std::sync::Mutex;
#[cfg(miri)]
use std::thread::{self, Thread};
#[cfg(miri)]
use std::vec::Vec;
#[cfg(all(not(miri), target_os = "linux"))]
use syscalls::{Errno, Sysno};
#[cfg(all(not(miri), target_os = "windows"))]
use winapi::{
    shared::basetsd::SIZE_T,
    um::{
        synchapi::{WaitOnAddress, WakeByAddressAll, WakeByAddressSingle},
        winbase::INFINITE,
    },
};

#[cfg(all(not(miri), target_os = "macos"))]
const OS_SYNC_WAIT_ON_ADDRESS_NONE: u32 = 0;
#[cfg(all(not(miri), target_os = "macos"))]
const OS_SYNC_WAKE_BY_ADDRESS_NONE: u32 = 0;
#[cfg(all(not(miri), target_os = "macos"))]
const EINTR: i32 = 4;
#[cfg(all(not(miri), target_os = "macos"))]
const EFAULT: i32 = 14;
#[cfg(all(not(miri), target_os = "macos"))]
const ENOMEM: i32 = 12;

#[cfg(all(not(miri), target_os = "macos"))]
unsafe extern "C" {
    fn os_sync_wait_on_address(addr: *mut c_void, value: u64, size: usize, flags: u32) -> c_int;
    fn os_sync_wake_by_address_any(addr: *mut c_void, size: usize, flags: u32) -> c_int;
    fn os_sync_wake_by_address_all(addr: *mut c_void, size: usize, flags: u32) -> c_int;
    fn __error() -> *mut c_int;
}

#[cfg(all(not(miri), target_os = "macos"))]
#[inline]
fn last_os_error() -> i32 {
    unsafe { *__error() }
}

#[cfg(all(not(miri), target_os = "linux"))]
#[inline]
unsafe fn linux_futex_wait(uaddr: *const AtomicU32, expected: u32) -> Result<(), Errno> {
    unsafe {
        syscalls::syscall!(
            Sysno::futex,
            uaddr,
            FUTEX_WAIT | FUTEX_PRIVATE_FLAG,
            expected,
            core::ptr::null::<u8>(),
            core::ptr::null::<u8>(),
            0u32
        )
    }
    .map(|_| ())
}

#[cfg(all(not(miri), target_os = "linux"))]
#[inline]
unsafe fn linux_futex_wake(uaddr: *const AtomicU32, wake_count: u32) -> Result<usize, Errno> {
    unsafe {
        syscalls::syscall!(
            Sysno::futex,
            uaddr,
            FUTEX_WAKE | FUTEX_PRIVATE_FLAG,
            wake_count,
            core::ptr::null::<u8>(),
            core::ptr::null::<u8>(),
            0u32
        )
    }
}

#[cfg_attr(not(miri), repr(transparent))]
pub struct Futex {
    word: AtomicU32,
    #[cfg(miri)]
    waiters: Mutex<Vec<Thread>>,
}

impl Futex {
    pub const fn new(init: u32) -> Self {
        Self {
            word: AtomicU32::new(init),
            #[cfg(miri)]
            waiters: Mutex::new(Vec::new()),
        }
    }

    #[inline]
    /// Waits while the futex word is equal to `expected`.
    ///
    /// # Safety
    ///
    /// `this` must point to a valid [`Futex`].
    pub unsafe fn wait(this: NonNull<Self>, expected: u32) {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        loop {
            if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                return;
            }

            match unsafe {
                wasm::memory_atomic_wait32(
                    (*this.as_ptr()).word.as_ptr().cast(),
                    expected as i32,
                    -1,
                )
            } {
                0 | 1 | 2 => continue,
                _ => return,
            }
        }

        #[cfg(miri)]
        loop {
            if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                return;
            }

            {
                let mut waiters = unsafe { &(*this.as_ptr()).waiters }
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                    return;
                }
                waiters.push(thread::current());
            }

            thread::park();
        }

        #[cfg(all(
            not(miri),
            not(any(target_os = "linux", target_os = "windows", target_os = "macos")),
            not(all(
                feature = "nightly",
                target_family = "wasm",
                target_feature = "atomics",
                any(target_arch = "wasm32", target_arch = "wasm64")
            ))
        ))]
        {
            while unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } == expected {
                core::hint::spin_loop();
            }
        }

        #[cfg(all(not(miri), target_os = "linux"))]
        loop {
            if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                return;
            }
            match unsafe { linux_futex_wait(core::ptr::addr_of!((*this.as_ptr()).word), expected) }
            {
                Ok(()) | Err(Errno::EINTR) | Err(Errno::EAGAIN) => continue,
                Err(_) => return,
            }
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        loop {
            if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                return;
            }

            let wait_succeeded = unsafe {
                WaitOnAddress(
                    (*this.as_ptr()).word.as_ptr().cast(),
                    core::ptr::from_ref(&expected).cast_mut().cast(),
                    core::mem::size_of::<u32>() as SIZE_T,
                    INFINITE,
                )
            } != 0;

            if !wait_succeeded {
                return;
            }
        }

        #[cfg(all(not(miri), target_os = "macos"))]
        loop {
            if unsafe { (*this.as_ptr()).word.load(Ordering::Acquire) } != expected {
                return;
            }

            let ret = unsafe {
                os_sync_wait_on_address(
                    (*this.as_ptr()).word.as_ptr().cast(),
                    expected as u64,
                    core::mem::size_of::<u32>(),
                    OS_SYNC_WAIT_ON_ADDRESS_NONE,
                )
            };
            if ret >= 0 {
                continue;
            }

            match last_os_error() {
                EINTR | EFAULT | ENOMEM => continue,
                _ => return,
            }
        }
    }

    #[inline]
    /// Wakes at most one waiter blocked on this futex.
    ///
    /// # Safety
    ///
    /// `this` must point to a valid [`Futex`].
    pub unsafe fn wake_one(this: NonNull<Self>) -> bool {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        return unsafe {
            wasm::memory_atomic_notify((*this.as_ptr()).word.as_ptr().cast(), 1) != 0
        };

        #[cfg(miri)]
        {
            let waiter = {
                let mut guard = unsafe { &(*this.as_ptr()).waiters }
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(waiter) = guard.pop() else {
                    return false;
                };
                waiter
            };
            waiter.unpark();
            return true;
        }

        #[cfg(all(
            not(miri),
            not(any(target_os = "linux", target_os = "windows", target_os = "macos")),
            not(all(
                feature = "nightly",
                target_family = "wasm",
                target_feature = "atomics",
                any(target_arch = "wasm32", target_arch = "wasm64")
            ))
        ))]
        return false;

        #[cfg(all(not(miri), target_os = "linux"))]
        {
            unsafe {
                linux_futex_wake(core::ptr::addr_of!((*this.as_ptr()).word), 1)
                    .map(|woken| woken != 0)
                    .unwrap_or(false)
            }
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        unsafe {
            WakeByAddressSingle((*this.as_ptr()).word.as_ptr().cast());
        }
        #[cfg(all(not(miri), target_os = "windows"))]
        return false;

        #[cfg(all(not(miri), target_os = "macos"))]
        unsafe {
            let _ = os_sync_wake_by_address_any(
                (*this.as_ptr()).word.as_ptr().cast(),
                core::mem::size_of::<u32>(),
                OS_SYNC_WAKE_BY_ADDRESS_NONE,
            );
        }
        #[cfg(all(not(miri), target_os = "macos"))]
        return false;
    }

    #[inline]
    /// Wakes all waiters blocked on this futex.
    ///
    /// # Safety
    ///
    /// `this` must point to a valid [`Futex`].
    pub unsafe fn wake_all(this: NonNull<Self>) {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        unsafe {
            wasm::memory_atomic_notify((*this.as_ptr()).word.as_ptr().cast(), i32::MAX as u32);
        }

        #[cfg(miri)]
        {
            let waiters = {
                let mut guard = unsafe { &(*this.as_ptr()).waiters }
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.drain(..).collect::<Vec<_>>()
            };

            for waiter in waiters {
                waiter.unpark();
            }
        }

        #[cfg(all(
            not(miri),
            not(any(target_os = "linux", target_os = "windows", target_os = "macos")),
            not(all(
                feature = "nightly",
                target_family = "wasm",
                target_feature = "atomics",
                any(target_arch = "wasm32", target_arch = "wasm64")
            ))
        ))]
        {}

        #[cfg(all(not(miri), target_os = "linux"))]
        {
            let _ = unsafe {
                linux_futex_wake(core::ptr::addr_of!((*this.as_ptr()).word), i32::MAX as u32)
            };
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        unsafe {
            WakeByAddressAll((*this.as_ptr()).word.as_ptr().cast());
        }

        #[cfg(all(not(miri), target_os = "macos"))]
        unsafe {
            let _ = os_sync_wake_by_address_all(
                (*this.as_ptr()).word.as_ptr().cast(),
                core::mem::size_of::<u32>(),
                OS_SYNC_WAKE_BY_ADDRESS_NONE,
            );
        }
    }
}

impl Deref for Futex {
    type Target = AtomicU32;

    fn deref(&self) -> &Self::Target {
        &self.word
    }
}

impl Default for Futex {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::Futex;
    use core::ptr::NonNull;
    use core::sync::atomic::Ordering;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use core::sync::atomic::{AtomicBool, AtomicUsize};
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::sync::Barrier;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::thread;

    #[test]
    fn default_initializes_to_zero() {
        let futex = Futex::default();
        assert_eq!(futex.load(Ordering::Acquire), 0);
    }

    #[test]
    fn new_initializes_to_value() {
        let futex = Futex::new(7);
        assert_eq!(futex.load(Ordering::Acquire), 7);
    }

    #[test]
    fn wait_returns_immediately_for_mismatched_value() {
        let futex = Futex::new(1);
        unsafe { Futex::wait(NonNull::from(&futex), 0) };
        assert_eq!(futex.load(Ordering::Acquire), 1);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn wake_one_with_same_value_does_not_release_a_waiter() {
        let futex = Futex::new(0);
        let ready = Barrier::new(2);
        let woke = AtomicBool::new(false);
        let mut woke_early = false;

        thread::scope(|scope| {
            scope.spawn(|| {
                ready.wait();
                unsafe { Futex::wait(NonNull::from(&futex), 0) };
                woke.store(true, Ordering::Release);
            });

            ready.wait();
            for _ in 0..1024 {
                unsafe { Futex::wake_one(NonNull::from(&futex)) };
                thread::yield_now();
            }

            woke_early = woke.load(Ordering::Acquire);
            futex.store(1, Ordering::Release);
            unsafe { Futex::wake_one(NonNull::from(&futex)) };
        });

        assert!(!woke_early);
        assert!(woke.load(Ordering::Acquire));
        assert_eq!(futex.load(Ordering::Acquire), 1);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn wake_one_releases_a_waiter() {
        let futex = Futex::new(0);
        let ready = Barrier::new(2);
        let woke = AtomicBool::new(false);

        thread::scope(|scope| {
            scope.spawn(|| {
                ready.wait();
                unsafe { Futex::wait(NonNull::from(&futex), 0) };
                woke.store(true, Ordering::Release);
            });

            ready.wait();
            futex.store(1, Ordering::Release);
            unsafe { Futex::wake_one(NonNull::from(&futex)) };

            for _ in 0..1024 {
                if woke.load(Ordering::Acquire) {
                    break;
                }
                thread::yield_now();
            }
        });

        assert!(woke.load(Ordering::Acquire));
        assert_eq!(futex.load(Ordering::Acquire), 1);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn wake_all_releases_all_waiters() {
        let futex = Futex::new(0);
        let waiter_count = 3;
        let ready = Barrier::new(waiter_count + 1);
        let woke = AtomicUsize::new(0);

        thread::scope(|scope| {
            for _ in 0..waiter_count {
                scope.spawn(|| {
                    ready.wait();
                    unsafe { Futex::wait(NonNull::from(&futex), 0) };
                    woke.fetch_add(1, Ordering::AcqRel);
                });
            }

            ready.wait();
            futex.store(1, Ordering::Release);
            unsafe { Futex::wake_all(NonNull::from(&futex)) };
        });

        assert_eq!(woke.load(Ordering::Acquire), waiter_count);
        assert_eq!(futex.load(Ordering::Acquire), 1);
    }
}
