use core::hint::spin_loop;
use core::ptr::NonNull;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};

use portable_futex::Futex;

const READ_LOCKED: u32 = 1;
const MASK: u32 = (1 << 30) - 1;
const WRITE_LOCKED: u32 = MASK;
const MAX_READERS: u32 = MASK - 1;
const READERS_WAITING: u32 = 1 << 30;
const WRITERS_WAITING: u32 = 1 << 31;

#[repr(C)]
pub struct RwLock {
    // The state consists of a 30-bit reader counter, a readers-waiting flag,
    // and a writers-waiting flag.
    state: Futex,
    // Writers wait on a separate futex so readers can continue waiting on the
    // lock state word itself.
    writer_notify: Futex,
}

#[inline]
fn is_unlocked(state: u32) -> bool {
    state & MASK == 0
}

#[inline]
fn is_write_locked(state: u32) -> bool {
    state & MASK == WRITE_LOCKED
}

#[inline]
fn has_readers_waiting(state: u32) -> bool {
    state & READERS_WAITING != 0
}

#[inline]
fn has_writers_waiting(state: u32) -> bool {
    state & WRITERS_WAITING != 0
}

#[inline]
fn is_read_lockable(state: u32) -> bool {
    state & MASK < MAX_READERS && !has_readers_waiting(state) && !has_writers_waiting(state)
}

#[inline]
fn has_reached_max_readers(state: u32) -> bool {
    state & MASK == MAX_READERS
}

impl RwLock {
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: Futex::new(0),
            writer_notify: Futex::new(0),
        }
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |state| {
                is_read_lockable(state).then_some(state + READ_LOCKED)
            })
            .is_ok()
    }

    #[inline]
    pub fn read(&self) {
        let state = self.state.load(Relaxed);
        if !is_read_lockable(state)
            || self
                .state
                .compare_exchange_weak(state, state + READ_LOCKED, Acquire, Relaxed)
                .is_err()
        {
            self.read_contended();
        }
    }

    #[inline]
    /// Releases one held read lock.
    ///
    /// # Safety
    ///
    /// The caller must currently hold a read lock acquired from this [`RwLock`].
    pub unsafe fn read_unlock(&self) {
        let state = self.state.fetch_sub(READ_LOCKED, Release) - READ_LOCKED;

        debug_assert!(!has_readers_waiting(state) || has_writers_waiting(state));

        if is_unlocked(state) && has_writers_waiting(state) {
            self.wake_writer_or_readers(state);
        }
    }

    #[cold]
    fn read_contended(&self) {
        let mut state = self.spin_read();

        loop {
            if is_read_lockable(state) {
                match self
                    .state
                    .compare_exchange_weak(state, state + READ_LOCKED, Acquire, Relaxed)
                {
                    Ok(_) => return,
                    Err(updated) => {
                        state = updated;
                        continue;
                    }
                }
            }

            if has_reached_max_readers(state) {
                panic!("too many active read locks on RwLock");
            }

            if !has_readers_waiting(state) {
                match self
                    .state
                    .compare_exchange(state, state | READERS_WAITING, Relaxed, Relaxed)
                {
                    Ok(_) => {}
                    Err(updated) => {
                        state = updated;
                        continue;
                    }
                }
            }

            unsafe { Futex::wait(NonNull::from(&self.state), state | READERS_WAITING) };
            state = self.spin_read();
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |state| {
                is_unlocked(state).then_some(state + WRITE_LOCKED)
            })
            .is_ok()
    }

    #[inline]
    pub fn write(&self) {
        if self
            .state
            .compare_exchange_weak(0, WRITE_LOCKED, Acquire, Relaxed)
            .is_err()
        {
            self.write_contended();
        }
    }

    #[inline]
    /// Releases one held write lock.
    ///
    /// # Safety
    ///
    /// The caller must currently hold the write lock for this [`RwLock`].
    pub unsafe fn write_unlock(&self) {
        let state = self.state.fetch_sub(WRITE_LOCKED, Release) - WRITE_LOCKED;

        debug_assert!(is_unlocked(state));

        if has_writers_waiting(state) || has_readers_waiting(state) {
            self.wake_writer_or_readers(state);
        }
    }

    #[cold]
    fn write_contended(&self) {
        let mut state = self.spin_write();
        let mut other_writers_waiting = 0;

        loop {
            if is_unlocked(state) {
                match self.state.compare_exchange_weak(
                    state,
                    state | WRITE_LOCKED | other_writers_waiting,
                    Acquire,
                    Relaxed,
                ) {
                    Ok(_) => return,
                    Err(updated) => {
                        state = updated;
                        continue;
                    }
                }
            }

            if !has_writers_waiting(state) {
                match self
                    .state
                    .compare_exchange(state, state | WRITERS_WAITING, Relaxed, Relaxed)
                {
                    Ok(_) => {}
                    Err(updated) => {
                        state = updated;
                        continue;
                    }
                }
            }

            other_writers_waiting = WRITERS_WAITING;

            let seq = self.writer_notify.load(Acquire);

            state = self.state.load(Relaxed);
            if is_unlocked(state) || !has_writers_waiting(state) {
                continue;
            }

            unsafe { Futex::wait(NonNull::from(&self.writer_notify), seq) };
            state = self.spin_write();
        }
    }

    #[cold]
    fn wake_writer_or_readers(&self, mut state: u32) {
        debug_assert!(is_unlocked(state));

        if state == WRITERS_WAITING {
            match self.state.compare_exchange(state, 0, Relaxed, Relaxed) {
                Ok(_) => {
                    self.wake_writer();
                    return;
                }
                Err(updated) => state = updated,
            }
        }

        if state == (READERS_WAITING | WRITERS_WAITING) {
            if self
                .state
                .compare_exchange(state, READERS_WAITING, Relaxed, Relaxed)
                .is_err()
            {
                return;
            }

            if self.wake_writer() {
                return;
            }

            state = READERS_WAITING;
        }

        if state == READERS_WAITING
            && self
                .state
                .compare_exchange(state, 0, Relaxed, Relaxed)
                .is_ok()
        {
            unsafe { Futex::wake_all(NonNull::from(&self.state)) };
        }
    }

    fn wake_writer(&self) -> bool {
        self.writer_notify.fetch_add(1, Release);
        unsafe { Futex::wake_one(NonNull::from(&self.writer_notify)) }
    }

    #[inline]
    fn spin_until(&self, f: impl Fn(u32) -> bool) -> u32 {
        let mut spin = 100;
        loop {
            let state = self.state.load(Relaxed);
            if f(state) || spin == 0 {
                return state;
            }
            spin_loop();
            spin -= 1;
        }
    }

    #[inline]
    fn spin_write(&self) -> u32 {
        self.spin_until(|state| is_unlocked(state) || has_writers_waiting(state))
    }

    #[inline]
    fn spin_read(&self) -> u32 {
        self.spin_until(|state| {
            !is_write_locked(state) || has_readers_waiting(state) || has_writers_waiting(state)
        })
    }
}

impl Default for RwLock {
    fn default() -> Self {
        Self::new()
    }
}

#[unsafe(no_mangle)]
#[cold]
/// Continues a contended read lock operation from the C ABI surface.
///
/// # Safety
///
/// `rwlock` must be a valid, non-null pointer to a live [`RwLock`] previously
/// initialized by this library. The pointed-to rwlock must remain valid for
/// the duration of the call.
pub unsafe extern "C" fn mlir_sync_rwlock_read_lock_slow_path(rwlock: *mut RwLock) {
    unsafe { (*rwlock).read_contended() }
}

#[unsafe(no_mangle)]
#[cold]
/// Continues a contended write lock operation from the C ABI surface.
///
/// # Safety
///
/// `rwlock` must be a valid, non-null pointer to a live [`RwLock`] previously
/// initialized by this library. The pointed-to rwlock must remain valid for
/// the duration of the call.
pub unsafe extern "C" fn mlir_sync_rwlock_write_lock_slow_path(rwlock: *mut RwLock) {
    unsafe { (*rwlock).write_contended() }
}

#[unsafe(no_mangle)]
#[cold]
/// Wakes blocked readers or writers after a contended unlock from the C ABI
/// surface.
///
/// # Safety
///
/// `rwlock` must be a valid, non-null pointer to a live [`RwLock`] previously
/// initialized by this library. `state` must be the post-unlock state word
/// that should be passed to [`RwLock::wake_writer_or_readers`].
pub unsafe extern "C" fn mlir_sync_rwlock_unlock_slow_path(rwlock: *mut RwLock, state: u32) {
    unsafe { (*rwlock).wake_writer_or_readers(state) }
}

#[cfg(test)]
mod tests {
    extern crate std;

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use super::RwLock;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use core::cell::UnsafeCell;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::sync::Barrier;
    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    use std::thread;

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    struct Shared<T> {
        lock: RwLock,
        value: UnsafeCell<T>,
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    impl<T> Shared<T> {
        fn new(value: T) -> Self {
            Self {
                lock: RwLock::new(),
                value: UnsafeCell::new(value),
            }
        }

        fn with_read<R>(&self, f: impl FnOnce(&T) -> R) -> R {
            self.lock.read();
            let result = unsafe { f(&*self.value.get()) };
            unsafe { self.lock.read_unlock() };
            result
        }

        fn with_write<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
            self.lock.write();
            let result = unsafe { f(&mut *self.value.get()) };
            unsafe { self.lock.write_unlock() };
            result
        }
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    unsafe impl<T: Send + Sync> Sync for Shared<T> {}

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn readers_can_share_access() {
        let shared = Shared::new(41usize);
        let readers = 4;
        let ready = Barrier::new(readers + 1);
        let release = Barrier::new(readers + 1);
        let active = AtomicUsize::new(0);
        let max_active = AtomicUsize::new(0);

        thread::scope(|scope| {
            for _ in 0..readers {
                let shared = &shared;
                let ready = &ready;
                let release = &release;
                let active = &active;
                let max_active = &max_active;
                scope.spawn(move || {
                    shared.lock.read();
                    let current = active.fetch_add(1, Ordering::AcqRel) + 1;
                    max_active.fetch_max(current, Ordering::AcqRel);
                    ready.wait();
                    release.wait();
                    active.fetch_sub(1, Ordering::AcqRel);
                    unsafe { shared.lock.read_unlock() };
                });
            }

            ready.wait();
            assert_eq!(shared.with_read(|value| *value), 41);
            release.wait();
        });

        assert!(max_active.load(Ordering::Acquire) > 1);
        assert_eq!(active.load(Ordering::Acquire), 0);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn writer_excludes_other_writers_and_readers() {
        let shared = Shared::new(0usize);
        let iterations = 128usize;

        thread::scope(|scope| {
            for _ in 0..2 {
                let shared = &shared;
                scope.spawn(move || {
                    for _ in 0..iterations {
                        shared.with_write(|value| *value += 1);
                    }
                });
            }
        });

        assert_eq!(shared.with_read(|value| *value), iterations * 2);
    }

    #[cfg(any(not(target_family = "wasm"), feature = "nightly"))]
    #[test]
    fn waiting_writer_runs_after_readers_release() {
        let shared = Shared::new(0usize);
        let readers = 3;
        let ready = Barrier::new(readers + 1);
        let release = Barrier::new(readers + 1);
        let writer_done = AtomicBool::new(false);

        thread::scope(|scope| {
            for _ in 0..readers {
                let shared = &shared;
                let ready = &ready;
                let release = &release;
                scope.spawn(move || {
                    shared.lock.read();
                    ready.wait();
                    release.wait();
                    unsafe { shared.lock.read_unlock() };
                });
            }

            ready.wait();

            let shared = &shared;
            let writer_done = &writer_done;
            scope.spawn(move || {
                shared.with_write(|value| *value = 7);
                writer_done.store(true, Ordering::Release);
            });

            for _ in 0..64 {
                assert!(!writer_done.load(Ordering::Acquire));
                thread::yield_now();
            }

            release.wait();
        });

        assert!(writer_done.load(Ordering::Acquire));
        assert_eq!(shared.with_read(|value| *value), 7);
    }
}
