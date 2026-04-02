#![no_std]
#![cfg_attr(
    all(feature = "nightly", target_family = "wasm"),
    feature(stdarch_wasm_atomic_wait)
)]

#[cfg(any(test, miri))]
extern crate std;

use core::ops::Deref;
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
use rustix::{io::Errno, thread::futex::Flags};
#[cfg(miri)]
use std::sync::Mutex;
#[cfg(miri)]
use std::thread::{self, Thread};
#[cfg(miri)]
use std::vec::Vec;
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
    pub fn wait(&self, expected: u32) {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        loop {
            if self.word.load(Ordering::Acquire) != expected {
                return;
            }

            match unsafe {
                wasm::memory_atomic_wait32(self.word.as_ptr().cast(), expected as i32, -1)
            } {
                0 | 1 | 2 => continue,
                _ => return,
            }
        }

        #[cfg(miri)]
        loop {
            if self.word.load(Ordering::Acquire) != expected {
                return;
            }

            {
                let mut waiters = self
                    .waiters
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if self.word.load(Ordering::Acquire) != expected {
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
            while self.word.load(Ordering::Acquire) == expected {
                core::hint::spin_loop();
            }
        }

        #[cfg(all(not(miri), target_os = "linux"))]
        loop {
            if self.word.load(Ordering::Acquire) != expected {
                return;
            }
            match rustix::thread::futex::wait(&self.word, Flags::PRIVATE, expected, None) {
                Ok(()) | Err(Errno::INTR) | Err(Errno::AGAIN) => continue,
                Err(_) => return,
            }
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        loop {
            if self.word.load(Ordering::Acquire) != expected {
                return;
            }

            let wait_succeeded = unsafe {
                WaitOnAddress(
                    self.word.as_ptr().cast(),
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
            if self.word.load(Ordering::Acquire) != expected {
                return;
            }

            let ret = unsafe {
                os_sync_wait_on_address(
                    self.word.as_ptr().cast(),
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
    pub fn wake_one(&self) {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        unsafe {
            wasm::memory_atomic_notify(self.word.as_ptr().cast(), 1);
        }

        #[cfg(miri)]
        {
            let waiter = {
                let mut guard = self
                    .waiters
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let Some(waiter) = guard.pop() else {
                    return;
                };
                waiter
            };
            waiter.unpark();
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
            let _ = rustix::thread::futex::wake(&self.word, Flags::PRIVATE, 1);
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        unsafe {
            WakeByAddressSingle(self.word.as_ptr().cast());
        }

        #[cfg(all(not(miri), target_os = "macos"))]
        unsafe {
            let _ = os_sync_wake_by_address_any(
                self.word.as_ptr().cast(),
                core::mem::size_of::<u32>(),
                OS_SYNC_WAKE_BY_ADDRESS_NONE,
            );
        }
    }

    #[inline]
    pub fn wake_all(&self) {
        #[cfg(all(
            feature = "nightly",
            not(miri),
            target_family = "wasm",
            target_feature = "atomics",
            any(target_arch = "wasm32", target_arch = "wasm64")
        ))]
        unsafe {
            wasm::memory_atomic_notify(self.word.as_ptr().cast(), i32::MAX as u32);
        }

        #[cfg(miri)]
        {
            let waiters = {
                let mut guard = self
                    .waiters
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
            let _ = rustix::thread::futex::wake(&self.word, Flags::PRIVATE, i32::MAX as u32);
        }

        #[cfg(all(not(miri), target_os = "windows"))]
        unsafe {
            WakeByAddressAll(self.word.as_ptr().cast());
        }

        #[cfg(all(not(miri), target_os = "macos"))]
        unsafe {
            let _ = os_sync_wake_by_address_all(
                self.word.as_ptr().cast(),
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
        futex.wait(0);
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
                futex.wait(0);
                woke.store(true, Ordering::Release);
            });

            ready.wait();
            for _ in 0..1024 {
                futex.wake_one();
                thread::yield_now();
            }

            woke_early = woke.load(Ordering::Acquire);
            futex.store(1, Ordering::Release);
            futex.wake_one();
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
                futex.wait(0);
                woke.store(true, Ordering::Release);
            });

            ready.wait();
            futex.store(1, Ordering::Release);
            futex.wake_one();

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
                    futex.wait(0);
                    woke.fetch_add(1, Ordering::AcqRel);
                });
            }

            ready.wait();
            futex.store(1, Ordering::Release);
            futex.wake_all();
        });

        assert_eq!(woke.load(Ordering::Acquire), waiter_count);
        assert_eq!(futex.load(Ordering::Acquire), 1);
    }
}
