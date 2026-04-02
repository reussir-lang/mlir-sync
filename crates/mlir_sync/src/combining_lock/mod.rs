use core::{
    cell::{Cell, UnsafeCell},
    mem::MaybeUninit,
    ptr::NonNull,
    sync::atomic::Ordering,
};

use crate::combining_lock::node::Node;

mod node;
mod rawlock;

/// The `Lock` struct is a thread-safe, poisonable lock that allows for safe concurrent access to data.
/// Create a new `Lock` with the [`Lock::new`] method.
/// To get access to the data, you can use the [`Lock::run`] method.
///
#[repr(C)]
pub struct Lock<T> {
    raw: rawlock::RawLock,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Sync for Lock<T> {}

impl<T> Lock<T> {
    /// Create a new lock with the given data.
    pub const fn new(data: T) -> Self {
        Self {
            raw: rawlock::RawLock::new(),
            data: UnsafeCell::new(data),
        }
    }

    #[inline(never)]
    fn run_slowly<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R + Send,
        R: Send,
    {
        #[repr(C)]
        struct CombinedNode<'a, T, F, R> {
            node: Node,
            closure: MaybeUninit<F>,
            data: &'a UnsafeCell<T>,
            result: Cell<MaybeUninit<R>>,
        }
        unsafe extern "C" fn execute<T, F, R>(this: *mut Node)
        where
            F: FnOnce(&mut T) -> R,
        {
            let this = this.cast::<CombinedNode<T, F, R>>();
            let this = unsafe { NonNull::new_unchecked(this) };
            let closure = unsafe { this.as_ref().closure.assume_init_read() };
            let data = unsafe { &mut *this.as_ref().data.get() };
            let result = (closure)(data);
            unsafe { this.as_ref().result.set(MaybeUninit::new(result)) };
        }
        let combined_node = CombinedNode {
            node: Node::new(execute::<T, F, R>),
            closure: MaybeUninit::new(f),
            data: &self.data,
            result: Cell::new(MaybeUninit::uninit()),
        };
        let this = NonNull::from(&combined_node).cast();
        Node::attach(this, &self.raw, 100);
        unsafe { combined_node.result.into_inner().assume_init() }
    }

    /// Schedules a closure to run on the lock's data.
    /// The locking strategy splits into two paths:
    /// 1. If the lock is not poisoned and can be acquired immediately, it runs the closure directly.
    ///    On the fast path, the closure is not spilled into the node.
    /// 2. If the lock is poisoned or cannot be acquired immediately, it schedules the closure to run later.
    /// ```rust
    /// use lamlock::Lock;
    /// let lock = Lock::new(0);
    /// lock.run(|data| {
    ///   *data += 1;
    /// }).unwrap();
    /// ```
    #[inline(always)]
    pub fn run<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut T) -> R + Send,
        R: Send,
    {
        if !self.raw.has_tail(Ordering::Relaxed) && self.raw.try_acquire() {
            let result = f(unsafe { &mut *self.data.get() });
            self.raw.release();
            return result;
        }
        self.run_slowly(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;

    #[test]
    fn smoke_test() {
        let lock = Lock::new(0);
        lock.run(|data| {
            *data += 1;
        });
        assert_eq!(lock.run(|x| *x), 1);
    }

    #[test]
    fn multi_thread_test() {
        let cnt = 100;
        let lock = Lock::new(0);
        std::thread::scope(|scope| {
            for i in 0..cnt {
                let lock = &lock;
                scope.spawn(move || {
                    lock.run(|data| {
                        *data += cnt - i;
                    });
                });
            }
        });

        assert_eq!(lock.run(|x| *x), cnt * (cnt + 1) / 2);
    }
}
