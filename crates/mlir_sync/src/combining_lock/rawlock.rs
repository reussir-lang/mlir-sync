use core::{
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicPtr, Ordering},
};

use super::node::Node;

#[repr(C)]
pub struct RawLock {
    tail: AtomicPtr<Node>,
    status: AtomicBool,
}

impl RawLock {
    pub const fn new() -> Self {
        Self {
            tail: AtomicPtr::new(core::ptr::null_mut()),
            status: AtomicBool::new(false),
        }
    }

    pub fn has_tail(&self, ordering: Ordering) -> bool {
        !self.tail.load(ordering).is_null()
    }

    pub fn swap_tail(&self, new_tail: NonNull<Node>) -> Option<NonNull<Node>> {
        let old_tail = self.tail.swap(new_tail.as_ptr(), Ordering::AcqRel);
        NonNull::new(old_tail)
    }

    pub fn try_close(&self, expected: NonNull<Node>) -> bool {
        self.tail
            .compare_exchange(
                expected.as_ptr(),
                core::ptr::null_mut(),
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }
    pub fn try_acquire(&self) -> bool {
        self.status
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }
    pub fn acquire(&self) {
        loop {
            match self
                .status
                .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(_) => {
                    while self.status.load(Ordering::Relaxed) {
                        core::hint::spin_loop();
                    }
                }
            }
        }
    }
    pub fn release(&self) {
        self.status.store(false, Ordering::Release);
    }
}
