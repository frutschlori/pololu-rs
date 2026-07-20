//! Storage for the SD-loaded gain-MLP parametrization (GAINMLP.JSN).
//!
//! Mirrors the trajectory registration in `setpoint`: the network is parsed
//! once during SD init, moved into a `StaticCell` and registered here. The
//! outer trajectory loop evaluates it every tick; the resulting scale factors
//! for the inner motor gains are published through atomics so the wheel-speed
//! inner loop can pick them up without locking.

use core::cell::RefCell;
use embassy_futures::block_on;
use embassy_sync::blocking_mutex::raw::ThreadModeRawMutex;
use embassy_sync::mutex::Mutex;
use gain_mlp::GainMlp;
use portable_atomic::{AtomicU32, Ordering};
use static_cell::StaticCell;

pub static GAIN_MLP_REF: Mutex<ThreadModeRawMutex, RefCell<Option<&'static GainMlp>>> =
    Mutex::new(RefCell::new(None));
static GAIN_MLP_CELL: StaticCell<GainMlp> = StaticCell::new();

pub fn store_gain_mlp(mlp: GainMlp) -> &'static GainMlp {
    GAIN_MLP_CELL.init(mlp)
}

pub fn register_gain_mlp(mlp: &'static GainMlp) {
    block_on(async {
        let g = GAIN_MLP_REF.lock().await;
        *g.borrow_mut() = Some(mlp);
    })
}

pub async fn gain_mlp() -> Option<&'static GainMlp> {
    let g = GAIN_MLP_REF.lock().await;
    let r = *g.borrow();
    r
}

// ============== Inner-gain scale factors (outer -> inner loop) ==============

const ONE_F32_BITS: u32 = 0x3F80_0000;

static INNER_KP_SCALE: AtomicU32 = AtomicU32::new(ONE_F32_BITS);
static INNER_KI_SCALE: AtomicU32 = AtomicU32::new(ONE_F32_BITS);

pub fn set_inner_gain_scales(kp_scale: f32, ki_scale: f32) {
    INNER_KP_SCALE.store(kp_scale.to_bits(), Ordering::Relaxed);
    INNER_KI_SCALE.store(ki_scale.to_bits(), Ordering::Relaxed);
}

pub fn reset_inner_gain_scales() {
    set_inner_gain_scales(1.0, 1.0);
}

pub fn inner_gain_scales() -> (f32, f32) {
    (
        f32::from_bits(INNER_KP_SCALE.load(Ordering::Relaxed)),
        f32::from_bits(INNER_KI_SCALE.load(Ordering::Relaxed)),
    )
}
