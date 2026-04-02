use core::{
    ptr::NonNull,
    sync::atomic::{AtomicPtr, Ordering},
};

use super::rawlock::RawLock;
use portable_futex as futex;

const SPIN_LIMIT: usize = 100;
const WAITING: u32 = 0;
const DONE: u32 = 1;
const HEAD: u32 = 2;
const SLEEPING: u32 = 3;

pub struct Node {
    futex: futex::Futex,
    next: AtomicPtr<Self>,
    closure: unsafe extern "C" fn(*mut Node),
}

impl Node {
    #[inline]
    fn futex_ptr(this: NonNull<Self>) -> NonNull<futex::Futex> {
        unsafe { NonNull::new_unchecked(core::ptr::addr_of_mut!((*this.as_ptr()).futex)) }
    }

    /// Creates a new `Node` with an initial state of `WAITING`.
    /// The `next` pointer is initialized to `null`.
    pub const fn new(closure: unsafe extern "C" fn(*mut Node)) -> Self {
        Self {
            futex: futex::Futex::new(WAITING),
            next: AtomicPtr::new(core::ptr::null_mut()),
            closure,
        }
    }

    /// Go to sleep until the futex is woken up with a message.
    pub fn wait(this: NonNull<Self>) -> u32 {
        match unsafe {
            this.as_ref().futex.compare_exchange(
                WAITING,
                SLEEPING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
        } {
            Ok(_) => {
                unsafe { futex::Futex::wait(Self::futex_ptr(this), SLEEPING) };
                unsafe { this.as_ref().futex.load(Ordering::Acquire) }
            }
            Err(value) => value,
        }
    }

    /// Wakes up the futex with a message.
    fn wake(this: NonNull<Self>, message: u32) {
        if unsafe { this.as_ref().futex.swap(message, Ordering::AcqRel) } == SLEEPING {
            unsafe { futex::Futex::wake_one(Self::futex_ptr(this)) };
        }
    }

    /// Wake up the futex with `DONE` message.
    pub fn wake_as_done(this: NonNull<Self>) {
        Self::wake(this, DONE);
    }
    /// Wake up the futex with `HEAD` message.
    pub fn wake_as_head(this: NonNull<Self>) {
        Self::wake(this, HEAD);
    }

    /// Get the successor node.
    pub fn load_next(&self, ordering: Ordering) -> Option<NonNull<Self>> {
        let ptr = self.next.load(ordering);
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { NonNull::new_unchecked(ptr) })
        }
    }

    #[cfg(all(feature = "nightly", not(miri)))]
    pub unsafe fn prefetch_next(&self, ordering: Ordering) {
        let ptr = self.next.load(ordering);
        core::intrinsics::prefetch_write_data::<Node, 3>(ptr);
    }

    /// Store the next node in the linked list.
    pub fn store_next(&self, next: NonNull<Self>) {
        self.next.store(next.as_ptr(), Ordering::Release);
    }

    /// Attach the node to a raw lock.
    pub fn attach(this: NonNull<Self>, raw: &RawLock, combine_limit: usize) {
        match raw.swap_tail(this) {
            Some(prev) => unsafe {
                prev.as_ref().store_next(this);
                let mut status;
                'waiting: {
                    for _ in 0..SPIN_LIMIT {
                        status = this.as_ref().futex.load(Ordering::Acquire);
                        if status != WAITING {
                            break 'waiting;
                        }
                    }
                    status = Node::wait(this);
                }
                if status == DONE {
                    return;
                }
                debug_assert_eq!(status, HEAD);
            },
            None => {
                raw.acquire();
            }
        }
        let mut cursor = this;
        for _ in 0..combine_limit {
            #[cfg(all(feature = "nightly", not(miri)))]
            unsafe {
                cursor.as_ref().prefetch_next(Ordering::Relaxed);
            }
            unsafe {
                (cursor.as_ref().closure)(cursor.as_ptr());
            }
            match unsafe { cursor.as_ref().load_next(Ordering::Acquire) } {
                Some(next) => {
                    Node::wake_as_done(cursor);
                    cursor = next;
                }
                None => break,
            }
        }

        if raw.try_close(cursor) {
            Node::wake_as_done(cursor);

            raw.release();
            return;
        }

        loop {
            match unsafe { cursor.as_ref().load_next(Ordering::Acquire) } {
                Some(next) => {
                    Node::wake_as_head(next);
                    Node::wake_as_done(cursor);
                    return;
                }
                None => {
                    debug_assert!(raw.has_tail(Ordering::SeqCst));
                    continue;
                }
            }
        }
    }
}

#[unsafe(no_mangle)]
#[cold]
unsafe extern "C" fn mlir_sync_combining_lock_attach_slow_path(
    node: *mut Node,
    raw: *mut RawLock,
    combine_limit: usize,
) {
    let node = unsafe { NonNull::new_unchecked(node) };
    let raw = unsafe { &*raw };
    Node::attach(node, raw, combine_limit);
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    extern "C" fn noop(_: *mut Node) {}

    #[test]
    fn test_node_wait() {
        let node = Node::new(noop);
        std::thread::scope(|s| {
            {
                let node = &node;
                s.spawn(move || {
                    let result = Node::wait(NonNull::from(node));
                    assert_eq!(result, HEAD);
                });
            }
            Node::wake((&node).into(), HEAD);
        })
    }

    #[test]
    fn test_node_next() {
        let node = Node::new(noop);
        std::thread::scope(|s| {
            {
                let node = &node;
                s.spawn(move || {
                    let local_node = Node::new(noop);
                    node.store_next(NonNull::from(&local_node));
                    assert_eq!(Node::wait(NonNull::from(&local_node)), DONE);
                });
            }
            loop {
                match node.load_next(Ordering::Acquire) {
                    Some(next) => {
                        Node::wake(next, DONE);
                        break;
                    }
                    None => core::hint::spin_loop(),
                }
            }
        })
    }
}
