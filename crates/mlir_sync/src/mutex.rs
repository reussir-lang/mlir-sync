use core::hint::spin_loop;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use portable_futex::Futex;

type State = u32;

const UNLOCKED: State = 0;
const LOCKED: State = 1; // locked, no other threads waiting
const CONTENDED: State = 2; // locked, and other threads waiting (contended)

#[repr(transparent)]
pub struct Mutex {
    futex: Futex,
}

impl Mutex {
    #[inline]
    pub const fn new() -> Self {
        Self {
            futex: Futex::new(UNLOCKED),
        }
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        self.futex
            .compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed)
            .is_ok()
    }

    #[inline]
    pub fn lock(&self) {
        if self
            .futex
            .compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed)
            .is_err()
        {
            self.lock_contended();
        }
    }

    #[cold]
    fn lock_contended(&self) {
        let mut state = self.spin();

        if state == UNLOCKED {
            match self
                .futex
                .compare_exchange(UNLOCKED, LOCKED, Acquire, Relaxed)
            {
                Ok(_) => return,
                Err(s) => state = s,
            }
        }

        loop {
            if state != CONTENDED && self.futex.swap(CONTENDED, Acquire) == UNLOCKED {
                return;
            }

            self.futex.wait(CONTENDED);
            state = self.spin();
        }
    }

    fn spin(&self) -> State {
        let mut spin = 100;
        loop {
            let state = self.futex.load(Relaxed);

            if state != LOCKED || spin == 0 {
                return state;
            }

            spin_loop();
            spin -= 1;
        }
    }

    #[inline]
    /// Releases the mutex.
    ///
    /// # Safety
    ///
    /// The caller must currently hold this mutex. Calling `unlock` when the
    /// mutex is not locked by the current execution context violates the mutex
    /// protocol and may cause undefined behavior in code that relies on it.
    pub unsafe fn unlock(&self) {
        if self.futex.swap(UNLOCKED, Release) == CONTENDED {
            self.wake();
        }
    }

    #[cold]
    fn wake(&self) {
        self.futex.wake_one();
    }
}

impl Default for Mutex {
    fn default() -> Self {
        Self::new()
    }
}

#[unsafe(no_mangle)]
/// Continues a contended lock operation from the C ABI surface.
///
/// # Safety
///
/// `mutex` must be a valid, non-null pointer to a live [`Mutex`] previously
/// initialized by this library. The pointed-to mutex must remain valid for the
/// duration of the call.
pub unsafe extern "C" fn mlir_sync_mutex_lock_slow_path(mutex: *mut Mutex) {
    unsafe { (*mutex).lock_contended() }
}

#[unsafe(no_mangle)]
/// Wakes a waiter for a contended mutex from the C ABI surface.
///
/// # Safety
///
/// `mutex` must be a valid, non-null pointer to a live [`Mutex`] previously
/// initialized by this library. The pointed-to mutex must remain valid for the
/// duration of the call.
pub unsafe extern "C" fn mlir_sync_mutex_unlock_slow_path(mutex: *mut Mutex) {
    unsafe { (*mutex).wake() }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    extern crate std;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use super::Mutex;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use core::cell::UnsafeCell;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::collections::BTreeMap;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::thread;

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    struct Shared<T> {
        lock: Mutex,
        value: UnsafeCell<T>,
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    impl<T> Shared<T> {
        fn new(value: T) -> Self {
            Self {
                lock: Mutex::new(),
                value: UnsafeCell::new(value),
            }
        }

        fn with_lock<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
            let result = {
                self.lock.lock();
                f(self.value.get())
            };

            //core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
            unsafe { self.lock.unlock() };
            result
        }
    }

    // The mutex serializes access to the protected value.
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    unsafe impl<T: Send> Sync for Shared<T> {}

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn mutex_guards_integer_addition() {
        let shared = Shared::new(0usize);

        thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    for _ in 0..128 {
                        shared.with_lock(|value| unsafe {
                            *value += 1;
                        });
                    }
                });
            }
        });

        let total = shared.with_lock(|value| unsafe { *value });
        assert_eq!(total, 256);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn mutex_guards_string_concatenation() {
        let shared = Shared::new(std::string::String::new());
        let fragments = [("mlir", 8usize), ("sync", 8usize)];

        thread::scope(|scope| {
            for (fragment, repeat) in fragments {
                let shared = &shared;
                scope.spawn(move || {
                    for _ in 0..repeat {
                        shared.with_lock(|value| unsafe {
                            (*value).push_str(fragment);
                            (*value).push('|');
                        });
                    }
                });
            }
        });

        let mut counts = BTreeMap::new();
        let total_fragments = shared.with_lock(|value| unsafe {
            for fragment in (*value).split_terminator('|') {
                *counts
                    .entry(alloc::string::String::from(fragment))
                    .or_insert(0usize) += 1;
            }
            (*value).split_terminator('|').count()
        });

        assert_eq!(total_fragments, 16);
        assert_eq!(counts.get("mlir"), Some(&8));
        assert_eq!(counts.get("sync"), Some(&8));
    }
}
