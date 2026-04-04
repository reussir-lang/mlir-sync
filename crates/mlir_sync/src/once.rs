use core::ptr::NonNull;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use portable_futex::Futex;

const INCOMPLETE: u32 = 0;
const RUNNING: u32 = 1;
const QUEUED: u32 = 2;
const COMPLETE: u32 = 3;

#[repr(transparent)]
pub struct Once {
    state: Futex,
}

impl Once {
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: Futex::new(INCOMPLETE),
        }
    }

    #[inline]
    pub fn is_completed(&self) -> bool {
        self.state.load(Acquire) == COMPLETE
    }

    #[inline]
    #[track_caller]
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        if self.is_completed() {
            return;
        }

        if !self.slow_path_prologue() {
            return;
        }

        f();
        unsafe { self.slow_path_epilogue() };
    }

    #[cold]
    pub fn slow_path_prologue(&self) -> bool {
        let mut state = self.state.load(Acquire);

        loop {
            match state {
                INCOMPLETE => {
                    if let Err(new) =
                        self.state
                            .compare_exchange_weak(INCOMPLETE, RUNNING, Acquire, Acquire)
                    {
                        state = new;
                        continue;
                    }

                    return true;
                }
                RUNNING | QUEUED => {
                    if state == RUNNING
                        && let Err(new) =
                            self.state
                                .compare_exchange_weak(RUNNING, QUEUED, Relaxed, Acquire)
                        {
                            state = new;
                            continue;
                        }

                    unsafe { Futex::wait(NonNull::from(&self.state), QUEUED) };
                    state = self.state.load(Acquire);
                }
                COMPLETE => return false,
                _ => unreachable!("invalid Once state"),
            }
        }
    }

    #[inline]
    /// Marks a previously granted slow-path execution slot as complete.
    ///
    /// # Safety
    ///
    /// The caller must have previously observed `true` from
    /// [`Once::slow_path_prologue`] for this `Once` and must be the execution
    /// context responsible for finishing initialization.
    pub unsafe fn slow_path_epilogue(&self) {
        if self.state.swap(COMPLETE, Release) == QUEUED {
            unsafe { Futex::wake_all(NonNull::from(&self.state)) };
        }
    }

}

impl Default for Once {
    fn default() -> Self {
        Self::new()
    }
}

#[unsafe(no_mangle)]
#[cold]
/// Arbitrates execution of a Once region after a fast-path completion check.
///
/// Returns `true` to exactly one caller, which then owns executing the region
/// and must later call [`mlir_sync_call_once_slow_path_epilogue`]. All other
/// callers either block until initialization completes or return `false` once
/// another execution context has completed it.
///
/// # Safety
///
/// `once` must be a valid, non-null pointer to a live [`Once`] previously
/// initialized by this library. The pointed-to once must remain valid for the
/// duration of the call.
pub unsafe extern "C" fn mlir_sync_call_once_slow_path_prologue(once: *mut Once) -> bool {
    unsafe { (*once).slow_path_prologue() }
}

#[unsafe(no_mangle)]
#[cold]
/// Publishes completion for a Once region and wakes blocked waiters.
///
/// # Safety
///
/// `once` must be a valid, non-null pointer to a live [`Once`] previously
/// initialized by this library. The caller must previously have received
/// `true` from [`mlir_sync_call_once_slow_path_prologue`] for this same once
/// instance and must be the execution context responsible for finishing the
/// initialization region.
pub unsafe extern "C" fn mlir_sync_call_once_slow_path_epilogue(once: *mut Once) {
    unsafe { (*once).slow_path_epilogue() }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::Once;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::sync::Barrier;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::thread;

    #[test]
    fn default_is_incomplete() {
        let once = Once::default();
        assert!(!once.is_completed());
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn call_once_runs_only_once() {
        let once = Once::new();
        let runs = AtomicUsize::new(0);

        thread::scope(|scope| {
            for _ in 0..8 {
                let once = &once;
                let runs = &runs;
                scope.spawn(move || {
                    once.call_once(|| {
                        runs.fetch_add(1, Ordering::AcqRel);
                    });
                });
            }
        });

        assert_eq!(runs.load(Ordering::Acquire), 1);
        assert!(once.is_completed());
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn waiting_threads_block_until_initialization_finishes() {
        let once = Once::new();
        let value = AtomicUsize::new(0);
        let entered = Barrier::new(2);
        let release = Barrier::new(2);
        let waiter_returned = AtomicBool::new(false);

        thread::scope(|scope| {
            let once = &once;
            let value = &value;
            let entered = &entered;
            let release = &release;
            let waiter_returned = &waiter_returned;
            scope.spawn(move || {
                if !once.is_completed() && once.slow_path_prologue() {
                    entered.wait();
                    release.wait();
                    value.store(17, Ordering::Release);
                    unsafe { once.slow_path_epilogue() };
                }
            });

            entered.wait();

            scope.spawn(move || {
                if !once.is_completed() && once.slow_path_prologue() {
                    value.store(99, Ordering::Release);
                    unsafe { once.slow_path_epilogue() };
                }
                assert_eq!(value.load(Ordering::Acquire), 17);
                waiter_returned.store(true, Ordering::Release);
            });

            for _ in 0..64 {
                assert!(!waiter_returned.load(Ordering::Acquire));
                thread::yield_now();
            }

            release.wait();
        });

        assert!(waiter_returned.load(Ordering::Acquire));
        assert_eq!(value.load(Ordering::Acquire), 17);
        assert!(once.is_completed());
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn slow_path_prologue_returns_false_after_completion() {
        let once = Once::new();

        assert!(once.slow_path_prologue());
        unsafe { once.slow_path_epilogue() };

        assert!(once.is_completed());
        assert!(!once.slow_path_prologue());
    }
}
