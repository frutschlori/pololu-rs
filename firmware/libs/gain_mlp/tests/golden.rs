//! Golden test against the JAX implementation in wmr-simulator.
//!
//! Fixtures in `tests/data/` are generated on the PC with:
//!
//!   uv run python scripts/export_gain_mlp.py <path>/GAINMLP.JSN \
//!       --problem problems/<problem>.yaml [--tuned models/tuned_gains.yaml] \
//!       --golden <path>/GOLDEN.JSN
//!
//! (run from the wmr-simulator repo; see tests/data/README.md for the exact
//! command used for the checked-in fixtures). Run this test on the host:
//!
//!   cargo test --target x86_64-unknown-linux-gnu

use gain_mlp::{GainMlp, RefSetpoint, NUM_GAINS};
use serde::Deserialize;

#[derive(Deserialize)]
struct GoldenFile {
    cases: Vec<GoldenCase>,
}

#[derive(Deserialize)]
struct GoldenCase {
    #[serde(rename = "ref")]
    reference: [f32; 5],
    pose: [f32; 3],
    twist: [f32; 2],
    base_gains: [f32; NUM_GAINS],
    expected_factors: [f32; NUM_GAINS],
    expected_gains: [f32; NUM_GAINS],
}

fn assert_close(actual: f32, expected: f32, atol: f32, rtol: f32, what: &str) {
    let tol = atol + rtol * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: {actual} vs expected {expected} (tol {tol})"
    );
}

#[test]
fn matches_jax_golden_cases() {
    let network_bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/GAINMLP.JSN")).unwrap();
    let mlp = GainMlp::from_json(&network_bytes).expect("network parse");

    let golden_bytes =
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/GOLDEN.JSN")).unwrap();
    let golden: GoldenFile = serde_json::from_slice(&golden_bytes).unwrap();
    assert!(!golden.cases.is_empty());

    let mut any_nontrivial = false;
    for (index, case) in golden.cases.iter().enumerate() {
        let setpoint = RefSetpoint {
            x: case.reference[0],
            y: case.reference[1],
            theta: case.reference[2],
            v: case.reference[3],
            omega: case.reference[4],
        };
        let factors = mlp.factors(&setpoint, &case.pose, &case.twist);
        let gains = mlp.apply(&case.base_gains, &setpoint, &case.pose, &case.twist);
        for i in 0..NUM_GAINS {
            assert_close(
                factors[i],
                case.expected_factors[i],
                1e-5,
                1e-4,
                &format!("case {index} factor {i}"),
            );
            assert_close(
                gains[i],
                case.expected_gains[i],
                1e-5,
                1e-4,
                &format!("case {index} gain {i}"),
            );
            if (case.expected_factors[i] - 1.0).abs() > 1e-3 {
                any_nontrivial = true;
            }
        }
    }
    assert!(any_nontrivial, "golden fixture only contains identity factors");
}
