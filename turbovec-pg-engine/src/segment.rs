//! Segment-based ("LSM-style") build layer over `turbovec::IdMapIndex`.
//!
//! # Why this exists
//!
//! turbovec's index is built for *load-once, query-many*. Every mutation
//! invalidates the SIMD-blocked code layout (`TurboQuantIndex::add` resets
//! the `blocked` `OnceLock`), which is then rebuilt over the **entire**
//! corpus on the next search. That is fine for the library's "encode a
//! corpus, then serve it" use case, but fatal for a Postgres index, which
//! must accept inserts and deletes one tuple at a time.
//!
//! [`SegmentedIndex`] splits the corpus into:
//!
//! * a set of **sealed** segments — each an immutable-for-insert
//!   `IdMapIndex` whose SIMD layout is built once and reused, and
//! * one **hot** segment — a small `IdMapIndex` that absorbs new inserts.
//!
//! When the hot segment reaches `max_hot` vectors it is sealed (pushed onto
//! the sealed list) and a fresh hot segment takes over. So the cost of the
//! "rebuild the blocked layout" step is bounded by `max_hot`, not by the
//! total corpus size — the classic log-structured-merge trade.
//!
//! A search fans out across every segment, asks each for its local top-`k`,
//! and merges. Because each true global top-`k` element is, in particular,
//! within the top-`k` of *its own* segment, the merge is exact.
//!
//! Deletes are O(1): the engine routes a delete to the one segment that owns
//! the id and calls `IdMapIndex::remove` (a swap-remove under the hood). The
//! only rebuild cost is that segment's blocked layout on its next search —
//! again bounded by segment size, not corpus size.
//!
//! # What this is *not* (yet)
//!
//! This is a proof of concept for the *build/lifecycle* layer. It does not
//! implement Postgres MVCC visibility, WAL crash-safety, or shared-memory
//! storage — see the `turbovec-pg` crate README for how those map onto a
//! production design. One concrete known gap is called out at
//! [`SegmentedIndex::add`]: each segment fits its own TQ+ calibration.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use turbovec::{AddError, ConstructError, IdMapIndex};

/// Which segment currently owns a given external id.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Owner {
    /// The writable hot segment.
    Hot,
    /// Sealed segment at this index into `SegmentedIndex::sealed`.
    Sealed(usize),
}

/// On-disk manifest written alongside the per-segment `.tvim` files.
#[derive(Serialize, Deserialize)]
struct Manifest {
    dim: usize,
    bit_width: usize,
    max_hot: usize,
    n_sealed: usize,
    /// `owner` flattened to parallel arrays. `owner_seg[i] == -1` means the
    /// id lives in the hot segment; otherwise it is a sealed-segment index.
    owner_ids: Vec<u64>,
    owner_seg: Vec<i64>,
}

/// A segmented, incrementally-mutable index over turbovec.
pub struct SegmentedIndex {
    dim: usize,
    bit_width: usize,
    /// Soft cap: the hot segment is sealed once its length reaches this. A
    /// single batched `add` may overshoot it; that is fine.
    max_hot: usize,
    sealed: Vec<IdMapIndex>,
    hot: IdMapIndex,
    /// external id -> owning segment. Doubles as the live-id set, so
    /// `len()` is just `owner.len()`.
    owner: HashMap<u64, Owner>,
    /// The ids currently in the hot segment, so sealing can re-point them
    /// from `Owner::Hot` to `Owner::Sealed(_)` without scanning `owner`.
    hot_ids: HashSet<u64>,
}

impl SegmentedIndex {
    /// Create an empty index. `dim` must be a positive multiple of 8 and
    /// `bit_width` in `{2, 3, 4}` (the turbovec core constraints).
    pub fn new(dim: usize, bit_width: usize, max_hot: usize) -> Result<Self, ConstructError> {
        let hot = IdMapIndex::new(dim, bit_width)?;
        Ok(Self {
            dim,
            bit_width,
            max_hot: max_hot.max(1),
            sealed: Vec::new(),
            hot,
            owner: HashMap::new(),
            hot_ids: HashSet::new(),
        })
    }

    /// Add `n = vectors.len() / dim` vectors with the given external ids.
    ///
    /// Ids must be globally unique across *all* segments (not just the hot
    /// one) and unique within the batch; the whole batch is rejected if any
    /// id is already present, so a partial insert is impossible.
    ///
    /// # Known PoC limitation — per-segment calibration
    ///
    /// Each segment is an independent `IdMapIndex`, so each fits its own TQ+
    /// per-coordinate calibration on its first batch (or falls back to
    /// identity below turbovec's ~1000-sample floor). Search is
    /// self-consistent *within* a segment, so ranking is correct, but two
    /// large segments can encode the same coordinate against slightly
    /// different calibrations. A production design would fit one global
    /// calibration once and share it across segments — which needs
    /// turbovec-core to expose its `tqplus_shift` / `tqplus_scale` vectors
    /// (today they are `pub(crate)`).
    pub fn add(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.dim;
        if dim == 0 || vectors.len() % dim != 0 {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        if ids.len() != n {
            return Err(AddError::IdsCountMismatch {
                expected: n,
                got: ids.len(),
            });
        }
        // Global + within-batch duplicate check up front so the add is
        // all-or-nothing.
        let mut seen = HashSet::with_capacity(n);
        for &id in ids {
            if self.owner.contains_key(&id) || !seen.insert(id) {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }

        self.hot.add_with_ids_2d(vectors, dim, ids)?;
        for &id in ids {
            self.owner.insert(id, Owner::Hot);
            self.hot_ids.insert(id);
        }

        if self.hot.len() >= self.max_hot {
            self.seal();
        }
        Ok(())
    }

    /// Seal the hot segment: build its SIMD layout, move it onto the sealed
    /// list, and start a fresh hot segment. No-op if the hot segment is
    /// empty.
    pub fn seal(&mut self) {
        if self.hot.is_empty() {
            return;
        }
        let new_idx = self.sealed.len();
        // Warm the blocked layout now so the segment's first query after
        // sealing doesn't pay the one-time build cost.
        self.hot.prepare();
        let fresh = IdMapIndex::new(self.dim, self.bit_width)
            .expect("dim/bit_width validated at construction");
        let sealed_hot = std::mem::replace(&mut self.hot, fresh);
        self.sealed.push(sealed_hot);
        for id in self.hot_ids.drain() {
            self.owner.insert(id, Owner::Sealed(new_idx));
        }
    }

    /// Remove the vector with external id `id`. Returns `true` if it was
    /// present. O(1) plus a bounded (segment-sized) layout rebuild on that
    /// segment's next search.
    pub fn remove(&mut self, id: u64) -> bool {
        match self.owner.remove(&id) {
            None => false,
            Some(Owner::Hot) => {
                self.hot_ids.remove(&id);
                self.hot.remove(id)
            }
            Some(Owner::Sealed(seg)) => self.sealed[seg].remove(id),
        }
    }

    /// Top-`k` nearest ids for a single `dim`-length query, as
    /// `(id, score)` pairs sorted by descending score.
    ///
    /// # Panics
    /// Panics if `query.len() != dim`.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(
            query.len(),
            self.dim,
            "query length {} does not match index dim {}",
            query.len(),
            self.dim,
        );
        if k == 0 {
            return Vec::new();
        }

        // Fan out: ask every non-empty segment for its local top-k.
        let mut merged: Vec<(u64, f32)> = Vec::new();
        let mut collect = |seg: &IdMapIndex| {
            if seg.is_empty() {
                return;
            }
            let (scores, ids) = seg.search(query, k);
            merged.extend(ids.into_iter().zip(scores));
        };
        for seg in &self.sealed {
            collect(seg);
        }
        collect(&self.hot);

        // Merge: globally sort the per-segment winners and keep the best k.
        merged.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }

    /// Number of live vectors across all segments.
    pub fn len(&self) -> usize {
        self.owner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    /// Number of sealed segments (excludes the hot segment).
    pub fn n_sealed(&self) -> usize {
        self.sealed.len()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn bit_width(&self) -> usize {
        self.bit_width
    }

    /// True if `id` is currently present.
    pub fn contains(&self, id: u64) -> bool {
        self.owner.contains_key(&id)
    }

    /// Persist every segment as a `.tvim` file plus a JSON manifest under
    /// `dir`. Round-trips through [`SegmentedIndex::load`].
    pub fn save(&self, dir: impl AsRef<Path>) -> std::io::Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        for (i, seg) in self.sealed.iter().enumerate() {
            seg.write(dir.join(format!("seg_{i}.tvim")))?;
        }
        self.hot.write(dir.join("hot.tvim"))?;

        let mut owner_ids = Vec::with_capacity(self.owner.len());
        let mut owner_seg = Vec::with_capacity(self.owner.len());
        for (&id, &own) in &self.owner {
            owner_ids.push(id);
            owner_seg.push(match own {
                Owner::Hot => -1i64,
                Owner::Sealed(s) => s as i64,
            });
        }
        let manifest = Manifest {
            dim: self.dim,
            bit_width: self.bit_width,
            max_hot: self.max_hot,
            n_sealed: self.sealed.len(),
            owner_ids,
            owner_seg,
        };
        let f = std::fs::File::create(dir.join("manifest.json"))?;
        serde_json::to_writer(f, &manifest)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(())
    }

    /// Reload an index previously written by [`SegmentedIndex::save`].
    pub fn load(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        let f = std::fs::File::open(dir.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_reader(f)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut sealed = Vec::with_capacity(manifest.n_sealed);
        for i in 0..manifest.n_sealed {
            sealed.push(IdMapIndex::load(dir.join(format!("seg_{i}.tvim")))?);
        }
        let hot = IdMapIndex::load(dir.join("hot.tvim"))?;

        let mut owner = HashMap::with_capacity(manifest.owner_ids.len());
        let mut hot_ids = HashSet::new();
        for (&id, &seg) in manifest.owner_ids.iter().zip(manifest.owner_seg.iter()) {
            if seg < 0 {
                owner.insert(id, Owner::Hot);
                hot_ids.insert(id);
            } else {
                owner.insert(id, Owner::Sealed(seg as usize));
            }
        }

        Ok(Self {
            dim: manifest.dim,
            bit_width: manifest.bit_width,
            max_hot: manifest.max_hot,
            sealed,
            hot,
            owner,
            hot_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbovec::IdMapIndex;

    /// Deterministic pseudo-random vectors (SplitMix64-ish), small in
    /// magnitude. Returns a flat (n*dim) buffer.
    fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; n * dim];
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        for x in v.iter_mut() {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            // Map to ~[-1, 1).
            *x = (z as f32 / u64::MAX as f32) * 2.0 - 1.0;
        }
        v
    }

    fn unique_tmp_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("tvpg_{tag}_{nanos}_{:?}", std::thread::current().id()))
    }

    #[test]
    fn sealing_counts_and_len_track_inserts() {
        let dim = 64;
        let mut idx = SegmentedIndex::new(dim, 4, 32).unwrap();
        let n = 200;
        let vecs = gen(n, dim, 1);
        let ids: Vec<u64> = (0..n as u64).collect();
        // Add in batches of 10 so sealing fires mid-stream.
        for chunk_start in (0..n).step_by(10) {
            let end = (chunk_start + 10).min(n);
            let vslice = &vecs[chunk_start * dim..end * dim];
            idx.add(vslice, &ids[chunk_start..end]).unwrap();
        }
        assert_eq!(idx.len(), n);
        // Batches of 10 against a soft cap of 32 seal at ~40 vectors each,
        // so 200 vectors produce 5 sealed segments.
        assert!(idx.n_sealed() >= 5, "got {} sealed", idx.n_sealed());
    }

    #[test]
    fn rejects_duplicate_ids_globally_and_in_batch() {
        let dim = 64;
        let mut idx = SegmentedIndex::new(dim, 4, 8).unwrap();
        let vecs = gen(20, dim, 2);
        let ids: Vec<u64> = (0..20).collect();
        idx.add(&vecs, &ids).unwrap();
        // Re-adding an id that was sealed into an earlier segment must fail.
        let one = gen(1, dim, 99);
        assert!(matches!(
            idx.add(&one, &[3]),
            Err(AddError::IdAlreadyPresent(3))
        ));
        // Duplicate within a single batch must fail too.
        let two = gen(2, dim, 7);
        assert!(matches!(
            idx.add(&two, &[100, 100]),
            Err(AddError::IdAlreadyPresent(100))
        ));
        // Failed adds left the index untouched.
        assert_eq!(idx.len(), 20);
    }

    #[test]
    fn matches_monolithic_idmap_topk() {
        // The decisive correctness test: a segmented index and a single
        // monolithic IdMapIndex over identical data must agree on results.
        // Both use identity TQ+ calibration here (well under turbovec's
        // ~1000-sample floor), so per-vector codes — and therefore scores —
        // are bit-identical; only the fan-out/merge logic differs.
        let dim = 64;
        let n = 300;
        let vecs = gen(n, dim, 42);
        let ids: Vec<u64> = (0..n as u64).collect();

        let mut seg = SegmentedIndex::new(dim, 4, 37).unwrap();
        // Insert in irregular batches to exercise sealing.
        for (b, chunk_start) in (0..n).step_by(23).enumerate() {
            let end = (chunk_start + 23).min(n);
            let _ = b;
            seg.add(&vecs[chunk_start * dim..end * dim], &ids[chunk_start..end])
                .unwrap();
        }
        assert!(seg.n_sealed() >= 1);

        let mut mono = IdMapIndex::new(dim, 4).unwrap();
        mono.add_with_ids(&vecs, &ids).unwrap();

        let queries = gen(15, dim, 1234);
        let k = 10;
        for q in 0..15 {
            let query = &queries[q * dim..(q + 1) * dim];
            let seg_res = seg.search(query, k);
            let (mscores, mids) = mono.search(query, k);

            // Top-1 must match exactly.
            assert_eq!(
                seg_res[0].0, mids[0],
                "query {q}: top-1 id mismatch (seg={}, mono={})",
                seg_res[0].0, mids[0],
            );
            // Top-1 score must match closely (same codes → same score).
            assert!(
                (seg_res[0].1 - mscores[0]).abs() < 1e-4,
                "query {q}: top-1 score mismatch {} vs {}",
                seg_res[0].1,
                mscores[0],
            );
            // Top-k id *sets* must be equal.
            let seg_set: HashSet<u64> = seg_res.iter().map(|(id, _)| *id).collect();
            let mono_set: HashSet<u64> = mids.iter().copied().collect();
            assert_eq!(seg_set, mono_set, "query {q}: top-{k} id set mismatch");
        }
    }

    #[test]
    fn self_queries_match_monolithic_across_segments() {
        // Query the index with each of its own stored vectors. The winner
        // for vector i lives in whichever segment absorbed it, so this
        // exercises the fan-out + merge for every segment. Comparing the
        // top-1 against a monolithic IdMapIndex (rather than asserting
        // "self") keeps the test independent of quantizer recall; random
        // vectors give distinct f32 scores, so there are no tie-break
        // ambiguities between the two code paths.
        let dim = 64;
        let n = 256;
        let vecs = gen(n, dim, 9);
        let ids: Vec<u64> = (0..n as u64).collect();

        let mut seg = SegmentedIndex::new(dim, 4, 40).unwrap();
        for chunk_start in (0..n).step_by(17) {
            let end = (chunk_start + 17).min(n);
            seg.add(&vecs[chunk_start * dim..end * dim], &ids[chunk_start..end])
                .unwrap();
        }
        assert!(seg.n_sealed() >= 3, "got {} sealed", seg.n_sealed());

        let mut mono = IdMapIndex::new(dim, 4).unwrap();
        mono.add_with_ids(&vecs, &ids).unwrap();

        for i in 0..n {
            let q = &vecs[i * dim..(i + 1) * dim];
            let s = seg.search(q, 5);
            let (_mscores, mids) = mono.search(q, 5);
            assert_eq!(s[0].0, mids[0], "self-query {i}: top-1 differs");
        }
    }

    #[test]
    fn delete_routes_to_owning_segment() {
        // 130 vectors in batches of 20 against a soft cap of 32: batches
        // seal at ~40 vectors, leaving the final 10 in the hot segment. So
        // this deletes one id from a sealed segment AND one from the hot
        // segment, exercising both routes through `remove`.
        let dim = 64;
        let n = 130;
        let vecs = gen(n, dim, 5);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut idx = SegmentedIndex::new(dim, 4, 32).unwrap();
        for chunk_start in (0..n).step_by(20) {
            let end = (chunk_start + 20).min(n);
            idx.add(&vecs[chunk_start * dim..end * dim], &ids[chunk_start..end])
                .unwrap();
        }
        assert!(idx.n_sealed() >= 2, "got {} sealed", idx.n_sealed());

        let sealed_id = 5u64; // an early id → in a sealed segment
        let recent_id = 125u64; // a late id → still in the hot segment
        assert!(idx.remove(sealed_id));
        assert!(idx.remove(recent_id));
        assert!(!idx.remove(sealed_id)); // a second delete is a no-op
        assert_eq!(idx.len(), n - 2);
        assert!(!idx.contains(sealed_id) && !idx.contains(recent_id));

        // Neither deleted vector resurfaces, even when queried with itself.
        let q_sealed = &vecs[sealed_id as usize * dim..(sealed_id as usize + 1) * dim];
        assert!(idx
            .search(q_sealed, 10)
            .iter()
            .all(|(id, _)| *id != sealed_id));
        let q_recent = &vecs[recent_id as usize * dim..(recent_id as usize + 1) * dim];
        assert!(idx
            .search(q_recent, 10)
            .iter()
            .all(|(id, _)| *id != recent_id));

        // A survivor in the same sealed segment whose layout was just
        // invalidated by the delete still self-matches at top-1.
        let q10 = &vecs[10 * dim..11 * dim];
        assert_eq!(idx.search(q10, 1)[0].0, 10);
    }

    #[test]
    fn save_load_round_trip_preserves_results() {
        let dim = 64;
        let n = 150;
        let vecs = gen(n, dim, 5);
        let ids: Vec<u64> = (1000..1000 + n as u64).collect();
        let mut idx = SegmentedIndex::new(dim, 4, 40).unwrap();
        idx.add(&vecs, &ids).unwrap();
        idx.remove(1005); // exercise a tombstoned/removed slot through save

        let dir = unique_tmp_dir("saveload");
        idx.save(&dir).unwrap();
        let reloaded = SegmentedIndex::load(&dir).unwrap();

        assert_eq!(reloaded.len(), idx.len());
        assert_eq!(reloaded.n_sealed(), idx.n_sealed());

        let queries = gen(8, dim, 77);
        for q in 0..8 {
            let query = &queries[q * dim..(q + 1) * dim];
            let a = idx.search(query, 7);
            let b = reloaded.search(query, 7);
            let a_ids: Vec<u64> = a.iter().map(|(id, _)| *id).collect();
            let b_ids: Vec<u64> = b.iter().map(|(id, _)| *id).collect();
            assert_eq!(a_ids, b_ids, "query {q}: result order changed after reload");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_index_search_is_empty() {
        let idx = SegmentedIndex::new(64, 4, 16).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search(&vec![0.1f32; 64], 5).is_empty());
    }
}
