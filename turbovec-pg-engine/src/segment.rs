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
//! # Durability
//!
//! [`SegmentedIndex::sync`] persists the index incrementally and
//! crash-safely: sealed segments are **write-once** files, the changing hot
//! segment is snapshotted per generation, and a single atomic rename of a
//! `CURRENT` pointer commits each generation. A crash leaves the previous
//! committed generation fully intact (an interrupted `sync` writes new files
//! but never flips `CURRENT`). [`SegmentedIndex::open`] re-opens the last
//! committed state, so any process can attach to the same on-disk store —
//! the (PoC-level) "shared across backends" story: a shared, durable source
//! of truth that readers re-open to see a writer's committed generations.
//!
//! # What this is *not* (yet)
//!
//! A proof of concept for the *build / lifecycle / durability* layer. It does
//! not implement Postgres MVCC visibility, integration with Postgres' own
//! WAL, or zero-copy shared memory across backends — see the `turbovec-pg`
//! crate README for how those map onto a production design. Two known gaps:
//! per-segment TQ+ calibration (see [`SegmentedIndex::add`]) and unbounded
//! tombstone growth until sealed-segment compaction (a future step).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

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

/// Bumped if the on-disk manifest shape changes; `open` refuses other values.
const MANIFEST_FORMAT: u32 = 2;

/// One committed generation's manifest. Written as `MANIFEST-<gen>`; the
/// `CURRENT` file names the active one.
#[derive(Serialize, Deserialize)]
struct Manifest {
    format: u32,
    dim: usize,
    bit_width: usize,
    max_hot: usize,
    n_sealed: usize,
    /// File name of each sealed segment (write-once, ordinal-stable).
    seg_files: Vec<String>,
    /// File name of the hot-segment snapshot for this generation.
    hot_file: String,
    /// Live `owner` flattened to parallel arrays. `owner_seg[i] == -1` means
    /// the id lives in the hot segment; otherwise it is a sealed-segment index.
    owner_ids: Vec<u64>,
    owner_seg: Vec<i64>,
    /// Ids deleted from a sealed segment, with the owning sealed index. The
    /// write-once segment files still physically contain these ids, so `open`
    /// re-applies the deletes to match the live set.
    tombstone_ids: Vec<u64>,
    tombstone_seg: Vec<usize>,
}

// ─── Crash-safe file primitives ──────────────────────────────────────────────

/// Write bytes to `dir/name` atomically: write `name.tmp`, fsync it, rename
/// over `name`. A crash leaves either the old `name` or nothing — never a
/// torn file.
fn write_bytes_atomic(dir: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
    let tmp = dir.join(format!("{name}.tmp"));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(name))
}

/// Write an `IdMapIndex` to `dir/name` atomically (tmp + fsync + rename).
fn write_index_atomic(idx: &IdMapIndex, dir: &Path, name: &str) -> io::Result<()> {
    let tmp = dir.join(format!("{name}.tmp"));
    idx.write(&tmp)?;
    File::open(&tmp)?.sync_all()?; // IdMapIndex::write flushes but doesn't fsync
    std::fs::rename(&tmp, dir.join(name))
}

/// Best-effort directory fsync so a rename is durably recorded. Ignored on
/// platforms that don't allow opening a directory for sync.
fn fsync_dir(dir: &Path) {
    if let Ok(f) = File::open(dir) {
        let _ = f.sync_all();
    }
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

    // ── Durable-store bookkeeping (used by `sync` / `open`) ──
    /// Directory backing this index, once attached via `save`/`open`.
    dir: Option<PathBuf>,
    /// Last committed generation number (0 = never synced).
    generation: u64,
    /// On-disk file name for each sealed segment, parallel to `sealed`.
    /// `None` = sealed in memory but not yet written (write-once on next sync).
    seg_files: Vec<Option<String>>,
    /// Ids deleted from a sealed segment, with the owning sealed index. Lets
    /// `sync` keep sealed files write-once (deletes are recorded here instead
    /// of rewriting the file) and lets `open` re-apply them.
    tombstones: HashSet<(u64, usize)>,
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
            dir: None,
            generation: 0,
            seg_files: Vec::new(),
            tombstones: HashSet::new(),
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
        self.seg_files.push(None); // not yet written; persisted on next sync
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
                // Hot is rewritten in full on every sync, so a hot delete is
                // durable without a tombstone.
                self.hot_ids.remove(&id);
                self.hot.remove(id)
            }
            Some(Owner::Sealed(seg)) => {
                let removed = self.sealed[seg].remove(id);
                if removed {
                    // The sealed file is write-once, so record the delete to
                    // re-apply on open instead of rewriting the segment.
                    self.tombstones.insert((id, seg));
                }
                removed
            }
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

    /// Top-`k` nearest ids restricted to the `allowed` external ids — the
    /// hybrid-retrieval path (a SQL `WHERE` / tenant / ACL stage produces the
    /// candidate set, turbovec ranks within it).
    ///
    /// The allowlist is partitioned by owning segment and pushed into each
    /// segment's `IdMapIndex::search_with_allowlist`, where turbovec's kernel
    /// filters at 32-vector block granularity. A segment that owns none of the
    /// allowed ids is skipped entirely, so the cost scales with how selective
    /// the filter is, not with corpus size.
    ///
    /// Ids in `allowed` that are not currently present (deleted, or never
    /// inserted) are ignored rather than rejected — unlike
    /// `IdMapIndex::search_with_allowlist`, which panics on an unknown id.
    ///
    /// # Panics
    /// Panics if `query.len() != dim`.
    pub fn search_with_allowlist(
        &self,
        query: &[f32],
        k: usize,
        allowed: &[u64],
    ) -> Vec<(u64, f32)> {
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

        // Partition allowed ids by owning segment, de-duplicating as we go.
        // Unknown / deleted ids fall through the `None` arm and are dropped.
        let mut per_sealed: Vec<Vec<u64>> = vec![Vec::new(); self.sealed.len()];
        let mut per_hot: Vec<u64> = Vec::new();
        let mut seen = HashSet::with_capacity(allowed.len());
        for &id in allowed {
            if !seen.insert(id) {
                continue;
            }
            match self.owner.get(&id) {
                Some(Owner::Sealed(s)) => per_sealed[*s].push(id),
                Some(Owner::Hot) => per_hot.push(id),
                None => {}
            }
        }

        let mut merged: Vec<(u64, f32)> = Vec::new();
        for (s, ids) in per_sealed.iter().enumerate() {
            if ids.is_empty() {
                continue; // no allowed ids here — skip the whole segment
            }
            let (scores, rids) = self.sealed[s].search_with_allowlist(query, k, Some(ids));
            merged.extend(rids.into_iter().zip(scores));
        }
        if !per_hot.is_empty() {
            let (scores, rids) = self.hot.search_with_allowlist(query, k, Some(&per_hot));
            merged.extend(rids.into_iter().zip(scores));
        }

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

    /// Last committed durability generation (0 = never synced). Increments on
    /// each successful [`sync`](Self::sync).
    pub fn generation(&self) -> u64 {
        self.generation
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

    /// Incrementally and crash-safely persist the index to its attached
    /// directory (set by a prior [`save`](Self::save) or [`open`](Self::open)).
    ///
    /// Each call:
    /// 1. writes any not-yet-persisted **sealed** segments — once each, never
    ///    rewritten (deletes are carried as tombstones, not file rewrites);
    /// 2. snapshots the hot segment to `hot-<gen>.tvim`;
    /// 3. writes `MANIFEST-<gen>`; and
    /// 4. **atomically renames** `CURRENT` to name the new manifest — the
    ///    single commit point. A crash before that rename leaves the previous
    ///    committed generation fully intact.
    ///
    /// So an append-mostly workload only ever writes the new segment(s) + a
    /// small hot snapshot + the manifest per sync, not the whole corpus.
    pub fn sync(&mut self) -> io::Result<()> {
        let dir = self.dir.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "sync() on a detached index; call save(dir) or open(dir) first",
            )
        })?;
        std::fs::create_dir_all(&dir)?;
        let gen = self.generation + 1;

        // 1. Persist any sealed segments that aren't on disk yet (write-once).
        for i in 0..self.sealed.len() {
            if self.seg_files[i].is_none() {
                let name = format!("seg-{i:08}.tvim");
                write_index_atomic(&self.sealed[i], &dir, &name)?;
                self.seg_files[i] = Some(name);
            }
        }

        // 2. Snapshot the (mutable) hot segment for this generation.
        let hot_file = format!("hot-{gen:08}.tvim");
        write_index_atomic(&self.hot, &dir, &hot_file)?;

        // 3. Write the manifest for this generation.
        let manifest = self.build_manifest(&hot_file);
        let manifest_name = format!("MANIFEST-{gen:08}");
        let json = serde_json::to_vec(&manifest)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        write_bytes_atomic(&dir, &manifest_name, &json)?;

        // 4. Commit: atomically flip CURRENT to the new manifest.
        write_bytes_atomic(&dir, "CURRENT", manifest_name.as_bytes())?;
        fsync_dir(&dir);

        // 5. GC the superseded generation's hot snapshot + manifest (never the
        //    committed generation, never write-once sealed files).
        gc_old_generations(&dir, gen);

        self.generation = gen;
        Ok(())
    }

    fn build_manifest(&self, hot_file: &str) -> Manifest {
        let mut owner_ids = Vec::with_capacity(self.owner.len());
        let mut owner_seg = Vec::with_capacity(self.owner.len());
        for (&id, &own) in &self.owner {
            owner_ids.push(id);
            owner_seg.push(match own {
                Owner::Hot => -1i64,
                Owner::Sealed(s) => s as i64,
            });
        }
        let mut tombstone_ids = Vec::with_capacity(self.tombstones.len());
        let mut tombstone_seg = Vec::with_capacity(self.tombstones.len());
        for &(id, seg) in &self.tombstones {
            tombstone_ids.push(id);
            tombstone_seg.push(seg);
        }
        Manifest {
            format: MANIFEST_FORMAT,
            dim: self.dim,
            bit_width: self.bit_width,
            max_hot: self.max_hot,
            n_sealed: self.sealed.len(),
            seg_files: self
                .seg_files
                .iter()
                .map(|f| f.clone().expect("all sealed segments persisted before manifest"))
                .collect(),
            hot_file: hot_file.to_string(),
            owner_ids,
            owner_seg,
            tombstone_ids,
            tombstone_seg,
        }
    }

    /// Attach the index to `dir` and persist it. If already attached to the
    /// same directory this is an incremental [`sync`](Self::sync); attaching to
    /// a different directory re-roots and writes a full copy there.
    pub fn save(&mut self, dir: impl AsRef<Path>) -> io::Result<()> {
        let dir = dir.as_ref().to_path_buf();
        match &self.dir {
            Some(cur) if *cur == dir => {}
            _ => {
                self.dir = Some(dir);
                self.generation = 0;
                for f in self.seg_files.iter_mut() {
                    *f = None; // force a full write into the new location
                }
            }
        }
        self.sync()
    }

    /// Open the last committed generation of a durable store under `dir`. The
    /// returned index stays attached, so subsequent [`sync`](Self::sync) calls
    /// continue incrementally.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest_name = std::fs::read_to_string(dir.join("CURRENT"))?;
        let manifest_name = manifest_name.trim().to_string();
        let bytes = std::fs::read(dir.join(&manifest_name))?;
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if manifest.format != MANIFEST_FORMAT {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported manifest format {}", manifest.format),
            ));
        }
        if manifest.seg_files.len() != manifest.n_sealed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest seg_files length does not match n_sealed",
            ));
        }

        let mut sealed = Vec::with_capacity(manifest.n_sealed);
        for name in &manifest.seg_files {
            sealed.push(IdMapIndex::load(dir.join(name))?);
        }
        let hot = IdMapIndex::load(dir.join(&manifest.hot_file))?;

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

        let mut tombstones = HashSet::with_capacity(manifest.tombstone_ids.len());
        for (&id, &seg) in manifest.tombstone_ids.iter().zip(manifest.tombstone_seg.iter()) {
            tombstones.insert((id, seg));
        }

        let generation = manifest_name
            .strip_prefix("MANIFEST-")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        let mut me = Self {
            dim: manifest.dim,
            bit_width: manifest.bit_width,
            max_hot: manifest.max_hot,
            sealed,
            hot,
            owner,
            hot_ids,
            dir: Some(dir),
            generation,
            seg_files: manifest.seg_files.into_iter().map(Some).collect(),
            tombstones,
        };

        // Write-once segment files still physically contain deleted ids — prune
        // them so search matches the live set. `remove` is a no-op (returns
        // false) for ids already absent, so applying every tombstone is safe.
        let to_remove: Vec<(u64, usize)> = me.tombstones.iter().copied().collect();
        for (id, seg) in to_remove {
            if seg < me.sealed.len() {
                me.sealed[seg].remove(id);
            }
        }

        Ok(me)
    }

    /// Alias for [`open`](Self::open), kept for call-site compatibility.
    pub fn load(dir: impl AsRef<Path>) -> io::Result<Self> {
        Self::open(dir)
    }
}

/// Delete superseded `hot-<gen>` snapshots and `MANIFEST-<gen>` files whose
/// generation is older than `keep_gen`. Never touches `seg-*` (write-once),
/// `CURRENT`, or the kept generation. Best-effort: GC failures are ignored.
fn gc_old_generations(dir: &Path, keep_gen: u64) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let stale = name
            .strip_prefix("hot-")
            .and_then(|s| s.strip_suffix(".tvim"))
            .or_else(|| name.strip_prefix("MANIFEST-"))
            .and_then(|s| s.parse::<u64>().ok())
            .is_some_and(|g| g < keep_gen);
        if stale {
            let _ = std::fs::remove_file(entry.path());
        }
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

    #[test]
    fn allowlist_matches_monolithic() {
        // Filtered search across segments must equal a monolithic
        // IdMapIndex's filtered search. Same identity-calibration argument as
        // the unfiltered equivalence test, so results are bit-identical.
        let dim = 64;
        let n = 300;
        let vecs = gen(n, dim, 314);
        let ids: Vec<u64> = (0..n as u64).collect();

        let mut seg = SegmentedIndex::new(dim, 4, 50).unwrap();
        for chunk_start in (0..n).step_by(40) {
            let end = (chunk_start + 40).min(n);
            seg.add(&vecs[chunk_start * dim..end * dim], &ids[chunk_start..end])
                .unwrap();
        }
        assert!(seg.n_sealed() >= 2);

        let mut mono = IdMapIndex::new(dim, 4).unwrap();
        mono.add_with_ids(&vecs, &ids).unwrap();

        // Allowlist: every 3rd id (spans every segment), comfortably > k.
        let allow: Vec<u64> = ids.iter().copied().step_by(3).collect();
        let allow_set: HashSet<u64> = allow.iter().copied().collect();
        let k = 10;
        let queries = gen(12, dim, 271);
        for q in 0..12 {
            let query = &queries[q * dim..(q + 1) * dim];
            let s = seg.search_with_allowlist(query, k, &allow);
            let (mscores, mids) = mono.search_with_allowlist(query, k, Some(&allow));

            assert_eq!(s.len(), k, "query {q}: expected k filtered results");
            assert_eq!(s[0].0, mids[0], "query {q}: filtered top-1 differs");
            assert!((s[0].1 - mscores[0]).abs() < 1e-4);
            let s_set: HashSet<u64> = s.iter().map(|(id, _)| *id).collect();
            let m_set: HashSet<u64> = mids.iter().copied().collect();
            assert_eq!(s_set, m_set, "query {q}: filtered top-k set differs");
            // Every returned id must be in the allowlist.
            assert!(s.iter().all(|(id, _)| allow_set.contains(id)));
        }
    }

    #[test]
    fn allowlist_restricts_and_ignores_unknown_ids() {
        let dim = 64;
        let n = 120;
        let vecs = gen(n, dim, 88);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut seg = SegmentedIndex::new(dim, 4, 32).unwrap();
        for chunk_start in (0..n).step_by(25) {
            let end = (chunk_start + 25).min(n);
            seg.add(&vecs[chunk_start * dim..end * dim], &ids[chunk_start..end])
                .unwrap();
        }
        // Delete id 7, then include it (and an id that never existed) in the
        // allowlist alongside live ids spread across segments.
        seg.remove(7);
        let allow = vec![3u64, 7 /* deleted */, 50, 999_999 /* never existed */, 88];

        let res = seg.search_with_allowlist(&vecs[3 * dim..4 * dim], 10, &allow);
        let got: HashSet<u64> = res.iter().map(|(id, _)| *id).collect();

        // Only live, known allowlist ids may appear; unknown/deleted are dropped.
        let live_allowed: HashSet<u64> = HashSet::from([3u64, 50, 88]);
        assert!(got.is_subset(&live_allowed), "got ids outside allowlist: {got:?}");
        assert!(!got.contains(&7) && !got.contains(&999_999));
        // Querying with vector 3 (allowed) surfaces it.
        assert!(got.contains(&3));
    }

    #[test]
    fn durable_sync_open_round_trip() {
        let dim = 64;
        let n = 150;
        let vecs = gen(n, dim, 5);
        let ids: Vec<u64> = (1000..1000 + n as u64).collect();
        let mut idx = SegmentedIndex::new(dim, 4, 40).unwrap();
        idx.add(&vecs, &ids).unwrap();

        let dir = unique_tmp_dir("durable_rt");
        idx.save(&dir).unwrap();
        assert_eq!(idx.generation(), 1);

        let reopened = SegmentedIndex::open(&dir).unwrap();
        assert_eq!(reopened.len(), idx.len());
        assert_eq!(reopened.n_sealed(), idx.n_sealed());
        assert_eq!(reopened.generation(), 1);

        let queries = gen(8, dim, 77);
        for q in 0..8 {
            let query = &queries[q * dim..(q + 1) * dim];
            let a: Vec<u64> = idx.search(query, 7).iter().map(|(id, _)| *id).collect();
            let b: Vec<u64> = reopened.search(query, 7).iter().map(|(id, _)| *id).collect();
            assert_eq!(a, b, "query {q}: result order changed after reopen");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn durable_sync_is_incremental_and_write_once() {
        let dim = 64;
        let mut idx = SegmentedIndex::new(dim, 4, 32).unwrap();
        let vecs = gen(80, dim, 31);
        let ids: Vec<u64> = (0..80).collect();
        for cs in (0..80).step_by(20) {
            idx.add(&vecs[cs * dim..(cs + 20) * dim], &ids[cs..cs + 20]).unwrap();
        }
        assert_eq!(idx.n_sealed(), 2);

        let dir = unique_tmp_dir("durable_inc");
        idx.save(&dir).unwrap();
        assert_eq!(idx.generation(), 1);
        let seg0 = dir.join("seg-00000000.tvim");
        let seg0_bytes = std::fs::read(&seg0).unwrap();

        // Add a third sealed segment, then sync incrementally to the same dir.
        let more = gen(40, dim, 32);
        let more_ids: Vec<u64> = (80..120).collect();
        for cs in (0..40).step_by(20) {
            idx.add(&more[cs * dim..(cs + 20) * dim], &more_ids[cs..cs + 20]).unwrap();
        }
        assert_eq!(idx.n_sealed(), 3);
        idx.save(&dir).unwrap();
        assert_eq!(idx.generation(), 2);

        // Write-once: the first sealed segment's file is byte-identical.
        assert_eq!(std::fs::read(&seg0).unwrap(), seg0_bytes, "seg-0 was rewritten");
        // The new sealed segment was written.
        assert!(dir.join("seg-00000002.tvim").exists());
        // The superseded generation's hot snapshot + manifest were GC'd.
        assert!(!dir.join("hot-00000001.tvim").exists());
        assert!(!dir.join("MANIFEST-00000001").exists());
        assert!(dir.join("MANIFEST-00000002").exists());

        let reopened = SegmentedIndex::open(&dir).unwrap();
        assert_eq!(reopened.len(), 120);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn durable_delete_survives_reopen() {
        let dim = 64;
        let n = 90;
        let vecs = gen(n, dim, 21);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut idx = SegmentedIndex::new(dim, 4, 32).unwrap();
        for cs in (0..n).step_by(20) {
            let e = (cs + 20).min(n);
            idx.add(&vecs[cs * dim..e * dim], &ids[cs..e]).unwrap();
        }
        assert!(idx.n_sealed() >= 1);
        assert!(idx.remove(5)); // id 5 lives in a sealed segment

        let dir = unique_tmp_dir("durable_del");
        idx.save(&dir).unwrap();

        let reopened = SegmentedIndex::open(&dir).unwrap();
        assert_eq!(reopened.len(), n - 1);
        assert!(!reopened.contains(5));
        // The tombstone was re-applied: id 5 never resurfaces from the
        // write-once segment file, even queried with its own vector.
        let q5 = &vecs[5 * dim..6 * dim];
        assert!(reopened.search(q5, 10).iter().all(|(id, _)| *id != 5));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn durable_atomic_commit_ignores_uncommitted_generation() {
        let dim = 64;
        let n = 80;
        let vecs = gen(n, dim, 11);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut idx = SegmentedIndex::new(dim, 4, 32).unwrap();
        for cs in (0..n).step_by(20) {
            idx.add(&vecs[cs * dim..(cs + 20) * dim], &ids[cs..cs + 20]).unwrap();
        }
        let dir = unique_tmp_dir("durable_atomic");
        idx.save(&dir).unwrap(); // commits generation 1

        // Simulate a crash partway through the NEXT sync: new-generation files
        // exist on disk, but CURRENT was never flipped to point at them.
        std::fs::write(dir.join("MANIFEST-00000002"), b"{ not valid json").unwrap();
        std::fs::write(dir.join("CURRENT.tmp"), b"MANIFEST-00000002").unwrap();
        std::fs::write(dir.join("hot-00000002.tvim.tmp"), b"garbage").unwrap();

        // open() must still load the last COMMITTED generation (1), ignoring
        // the uncommitted generation-2 artifacts.
        let reopened = SegmentedIndex::open(&dir).unwrap();
        assert_eq!(reopened.generation(), 1);
        assert_eq!(reopened.len(), idx.len());
        let q = &vecs[7 * dim..8 * dim];
        let a: Vec<u64> = idx.search(q, 5).iter().map(|(i, _)| *i).collect();
        let b: Vec<u64> = reopened.search(q, 5).iter().map(|(i, _)| *i).collect();
        assert_eq!(a, b);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn durable_reopen_continues_incrementally() {
        // Simulates two backends sharing one on-disk store: handle A creates
        // and commits; handle B opens, appends, commits; handle C sees both.
        let dim = 64;
        let dir = unique_tmp_dir("durable_share");

        {
            let mut a = SegmentedIndex::new(dim, 4, 32).unwrap();
            let v = gen(50, dim, 41);
            let ids: Vec<u64> = (0..50).collect();
            a.add(&v, &ids).unwrap();
            a.save(&dir).unwrap();
        }

        let v2 = gen(30, dim, 42);
        let ids2: Vec<u64> = (50..80).collect();
        {
            let mut b = SegmentedIndex::open(&dir).unwrap();
            assert_eq!(b.len(), 50);
            b.add(&v2, &ids2).unwrap();
            b.sync().unwrap();
        }

        let c = SegmentedIndex::open(&dir).unwrap();
        assert_eq!(c.len(), 80);
        // A vector committed by the second handle is retrievable.
        let q = &v2[0..dim];
        assert!(c.search(q, 80).iter().any(|(id, _)| *id == 50));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
