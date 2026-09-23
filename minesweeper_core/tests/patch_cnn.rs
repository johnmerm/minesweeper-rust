//! Checks the hand-written network against the one that was trained.
//!
//! The architecture exists twice in executable form — PyTorch in `neural/model.py`
//! and Rust in `probability/patch_cnn.rs` — and they must agree to the last few
//! decimal places, because the Rust one is serving weights the PyTorch one
//! produced. Nothing about a disagreement looks like an error: the network still
//! runs and still returns numbers between 0 and 1.
//!
//! `neural/export_weights.py` therefore writes a `.vectors` file next to the
//! weights, holding patches and the outputs PyTorch gave for them, so this can be
//! checked without PyTorch present. Skipped when no model has been exported:
//!
//! ```bash
//! MINESWEEPER_WEIGHTS=neural/onnx/model.bin cargo test -p minesweeper_core
//! ```

use std::path::PathBuf;

use minesweeper_core::probability::{PatchCnn, WeightsError};

const PATCH_LEN: usize = 8 * 9 * 9;

fn weights_path() -> Option<PathBuf> {
    let path = PathBuf::from(
        std::env::var("MINESWEEPER_WEIGHTS").unwrap_or_else(|_| "../neural/onnx/model.bin".into()),
    );
    if path.exists() {
        Some(path)
    } else {
        eprintln!("skipping: no weights at {} — run neural/export_weights.py", path.display());
        None
    }
}

#[test]
fn matches_the_trained_model() {
    let Some(weights) = weights_path() else { return };
    let vectors = weights.with_extension("vectors");
    if !vectors.exists() {
        eprintln!("skipping: no {} beside the weights", vectors.display());
        return;
    }

    let network = PatchCnn::from_bytes(&std::fs::read(&weights).unwrap()).expect("weights");
    let raw = std::fs::read(&vectors).unwrap();
    assert_eq!(&raw[..8], b"MSPCNNV1", "{} is not a test-vector file", vectors.display());

    let count = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let floats: Vec<f32> = raw[12..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(floats.len(), count * (PATCH_LEN + 1), "test-vector file is the wrong length");

    let mut worst = 0.0f32;
    for i in 0..count {
        let record = &floats[i * (PATCH_LEN + 1)..][..PATCH_LEN + 1];
        let (patch, expected) = (&record[..PATCH_LEN], record[PATCH_LEN]);
        worst = worst.max((network.predict(patch) - expected).abs());
    }

    assert!(
        worst < 1e-4,
        "the Rust network and the trained one disagree by {worst:.2e} — \
         the architectures have drifted apart"
    );
}

/// A file that is not a weights file, or is the wrong size for this
/// architecture, must be refused rather than read as garbage.
#[test]
fn nonsense_weights_are_refused() {
    assert_eq!(PatchCnn::from_bytes(b"").unwrap_err(), WeightsError::BadMagic);
    assert_eq!(PatchCnn::from_bytes(b"not a model at all").unwrap_err(), WeightsError::BadMagic);

    let mut truncated = b"MSPCNN1\0".to_vec();
    truncated.extend(std::iter::repeat(0u8).take(4 * 100));
    assert!(matches!(
        PatchCnn::from_bytes(&truncated).unwrap_err(),
        WeightsError::WrongLength { .. }
    ));
}

/// Learning from a position must move the prediction towards the exact answer.
/// This is the whole premise of correcting the network during play, so it is
/// worth pinning down rather than assuming the gradient's sign.
#[test]
fn learning_moves_towards_the_target() {
    let Some(weights) = weights_path() else { return };
    let mut network = PatchCnn::from_bytes(&std::fs::read(&weights).unwrap()).expect("weights");

    // A patch that looks like a real one: a few numbers, mostly unopened.
    let mut patch = vec![0.0f32; PATCH_LEN];
    for pos in 0..81 {
        if pos % 5 == 0 {
            patch[81 + pos] = 1.0; // visible
            patch[pos] = 2.0 / 8.0; // showing a 2
        } else {
            patch[2 * 81 + pos] = 1.0; // hidden
        }
        patch[7 * 81 + pos] = 1.0;
    }
    patch[5 * 81 + 4 * 9 + 4] = 1.0;
    for pos in 0..81 {
        patch[6 * 81 + pos] = 0.15;
    }

    for target in [0.0f32, 1.0] {
        let mut network = PatchCnn::from_bytes(&std::fs::read(&weights).unwrap()).unwrap();
        let before = network.predict(&patch);
        for _ in 0..40 {
            network.learn(&patch, target, 0.5);
        }
        let after = network.predict(&patch);
        assert!(
            (after - target).abs() < (before - target).abs(),
            "learning towards {target} moved {before:.4} to {after:.4}, which is not closer"
        );
    }

    // And it must remain a probability throughout.
    for _ in 0..200 {
        network.learn(&patch, 1.0, 5.0);
    }
    let p = network.predict(&patch);
    assert!(p.is_finite() && (0.0..=1.0).contains(&p), "prediction left [0, 1]: {p}");
}
