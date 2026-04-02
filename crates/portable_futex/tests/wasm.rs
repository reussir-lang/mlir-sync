#![cfg(target_arch = "wasm32")]

use core::sync::atomic::Ordering;

use portable_futex::Futex;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wait_mismatch_returns_and_wake_updates_value() {
    let futex = Futex::new(1);

    futex.wait(0);
    futex.wake(7);

    assert_eq!(futex.load(Ordering::Acquire), 7);
}
