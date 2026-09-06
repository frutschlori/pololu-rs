use crate::orchestrator_signal::STOP_ODOM_SIG;
use crate::read_robot_config_from_sd::RobotConfig;
use crate::robotstate;
use defmt::info;
use embassy_futures::select::{Either, select};
use embassy_time::{Duration, Instant, Ticker};
use portable_atomic::{AtomicI32, Ordering};

use crate::encoder::wheel_speed_from_counts_now;

/// Odometry task: the wheel-measured body twist `(v, w)` the EKF predicts on.
///
/// This is deliberately a task of its own rather than a few lines in
/// `inner_controller`, even though both derive a wheel speed from the same
/// counters: the inner loop needs the *low-passed* speeds (`fc = 3 Hz`, i.e.
/// `tau = 53 ms`) for its PI feedback, while the EKF must predict on the raw
/// ones. Publishing `write_odom` from the inner loop's `omega_*_lp` puts that
/// 53 ms lag into the pose estimate the outer tracking law closes around, and
/// it makes the firmware disagree with the JAX simulator's `estimator.py`,
/// which predicts on the raw wheel speeds. Keep the two sources separate.
///
/// Responds to `STOP_ODOM_SIG` to cleanly exit when switching modes.
#[embassy_executor::task]
pub async fn odometry_task(
    left_counter: &'static AtomicI32,
    right_counter: &'static AtomicI32,
    cfg: Option<RobotConfig>,
    period_ms: u64,
) {
    let robot_cfg = cfg.unwrap_or_default();

    let dt_nominal: f32 = period_ms as f32 / 1000.0;
    // Ticker::every catches up after stalls (e.g. a long SD write) by yielding
    // the missed ticks back to back. Dividing one encoder count by such a tiny
    // interval is a large spurious twist, and unlike the inner loop's PI term
    // that spike would go straight into `ekf.predict`. Skip those ticks and
    // keep the previous counts for the next real sample - same guard as
    // `inner_controller`.
    let min_sample_dt: f32 = dt_nominal * 0.5;
    let mut ticker = Ticker::every(Duration::from_millis(period_ms));
    let mut prev_l: i32 = left_counter.load(Ordering::Relaxed);
    let mut prev_r: i32 = right_counter.load(Ordering::Relaxed);
    let mut last_sample = Instant::now();

    info!(
        "Odometry task started (wheel_r={}, wheel_base={}, cpr={})",
        robot_cfg.wheel_radius, robot_cfg.wheel_base, robot_cfg.encoder_cpr
    );

    loop {
        match select(ticker.next(), STOP_ODOM_SIG.wait()).await {
            Either::First(_) => { /* normal tick */ }
            Either::Second(_) => {
                defmt::info!("odometry_task stopped by STOP_ODOM_SIG");
                return;
            }
        }

        let sample_now = Instant::now();
        let dt = {
            let elapsed_s = (sample_now - last_sample).as_micros() as f32 / 1_000_000.0;
            if elapsed_s > 0.0 { elapsed_s } else { dt_nominal }
        };
        if dt < min_sample_dt {
            embassy_futures::yield_now().await;
            continue;
        }

        // Raw angular velocity of each wheel [rad/s] - no low-pass here.
        let ((omega_l, omega_r), (ln, rn)) = wheel_speed_from_counts_now(
            left_counter,
            right_counter,
            robot_cfg.encoder_cpr,
            prev_l,
            prev_r,
            dt,
        ).await;
        let measurement_stamp = Instant::now();
        prev_l = ln;
        prev_r = rn;
        last_sample = measurement_stamp;

        // Differential-drive kinematics as measured (not from CMD_unicycle)
        let v = (robot_cfg.wheel_radius * (omega_r + omega_l)) / 2.0;
        let w = (robot_cfg.wheel_radius * (omega_r - omega_l)) / robot_cfg.wheel_base;

        robotstate::write_odom(robotstate::OdomPose {
            v,
            w,
            stamp: measurement_stamp,
        }).await;
    }
}
