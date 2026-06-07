//! `with_calibration` / `calibration` — sharing one TQ+ calibration across
//! independently-built indexes (the primitive the segmented Postgres store
//! uses to keep per-segment scores comparable).

use turbovec::{IdMapIndex, TurboQuantIndex};

/// Deterministic SplitMix64-ish vectors in ~[-1, 1).
fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n * dim];
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    for x in v.iter_mut() {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        *x = (z as f32 / u64::MAX as f32) * 2.0 - 1.0;
    }
    v
}

#[test]
fn calibration_round_trips_through_getter() {
    let dim = 64;
    let shift: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
    let scale: Vec<f32> = (0..dim).map(|i| 1.0 + i as f32 * 0.001).collect();
    let idx = TurboQuantIndex::with_calibration(dim, 4, shift.clone(), scale.clone()).unwrap();
    let (gs, gc) = idx.calibration().expect("calibration should be set");
    assert_eq!(gs, &shift[..]);
    assert_eq!(gc, &scale[..]);
}

#[test]
fn fresh_index_has_no_calibration_until_first_add() {
    let mut idx = TurboQuantIndex::new(64, 4).unwrap();
    assert!(idx.calibration().is_none());
    idx.add(&gen(8, 64, 1));
    // Present after an add (identity for a small batch, but materialized).
    assert!(idx.calibration().is_some());
}

#[test]
#[should_panic(expected = "shift length")]
fn with_calibration_panics_on_length_mismatch() {
    let _ = TurboQuantIndex::with_calibration(64, 4, vec![0.0; 32], vec![1.0; 64]);
}

#[test]
fn shared_calibration_gives_identical_results_regardless_of_batching() {
    // Two indexes built with the SAME explicit calibration must encode every
    // vector identically — independent of how the adds are batched — because
    // the calibration is fixed, never refit. This is the property the
    // segmented store relies on for cross-segment score comparability.
    let dim = 64;
    let n = 200;
    let data = gen(n, dim, 7);
    let ids: Vec<u64> = (0..n as u64).collect();

    // Derive a calibration from one fitted probe index.
    let mut probe = TurboQuantIndex::new(dim, 4).unwrap();
    probe.add(&data);
    let (sh, sc) = probe.calibration().unwrap();
    let (sh, sc) = (sh.to_vec(), sc.to_vec());

    let mut a = IdMapIndex::with_calibration(dim, 4, sh.clone(), sc.clone()).unwrap();
    a.add_with_ids(&data, &ids).unwrap();

    // `b` ingests the same data split across two batches; the shared
    // calibration must NOT be refit on the first batch.
    let mut b = IdMapIndex::with_calibration(dim, 4, sh, sc).unwrap();
    b.add_with_ids(&data[..(n / 2) * dim], &ids[..n / 2]).unwrap();
    b.add_with_ids(&data[(n / 2) * dim..], &ids[n / 2..]).unwrap();

    let q = gen(5, dim, 99);
    let (sa, ia) = a.search(&q, 10);
    let (sb, ib) = b.search(&q, 10);
    assert_eq!(ia, ib, "shared-calibration indexes disagreed on ids");
    for (x, y) in sa.iter().zip(sb.iter()) {
        assert!((x - y).abs() < 1e-5, "shared-calibration scores differ: {x} vs {y}");
    }
}
