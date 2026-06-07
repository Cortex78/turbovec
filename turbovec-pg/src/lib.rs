//! Postgres extension exposing a turbovec-backed vector index.
//!
//! This is the **FFI shim** for the Path C proof of concept: a thin layer of
//! `#[pg_extern]` functions over [`turbovec_pg_engine::SegmentedIndex`]. All
//! of the interesting logic — the segment-based build layer and the ctid↔u64
//! mapping — lives in `turbovec-pg-engine` (pure Rust, unit-tested). This
//! file only:
//!
//! * registers SQL-callable functions with Postgres via `pgrx`,
//! * bridges Postgres `tid` (ctid) values to the engine's `u64` ids, and
//! * holds the live indexes in a per-backend registry.
//!
//! # What this PoC demonstrates
//!
//! That turbovec's quantizer + SIMD kernel can be driven from inside
//! Postgres, addressing each indexed vector by the heap tuple's `ctid` so a
//! search result joins straight back to the table — the same heap-pointer
//! model a native index access method uses.
//!
//! # What this PoC deliberately does NOT do (yet)
//!
//! * **It is not a Postgres index access method.** Functions are explicit
//!   (`turbovec_search(...)`), so the planner is not involved. A production
//!   version would register an `amhandler` / operator class so `ORDER BY
//!   embedding <#> query LIMIT k` uses it automatically.
//! * **State is per-backend.** The registry is a process-local static, so an
//!   index built in one connection is invisible to others and is lost when
//!   the backend exits (use `turbovec_save` / `turbovec_load`). A real
//!   implementation would live in shared memory / DSA or be page-backed
//!   through the buffer manager.
//! * **No MVCC / WAL.** No tuple-visibility checks and no crash-safety. See
//!   the crate README for how these map onto a production design.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use pgrx::prelude::*;

use turbovec_pg_engine::{tid, SegmentedIndex};

::pgrx::pg_module_magic!();

/// Per-backend registry of named indexes.
///
/// Process-local on purpose for the PoC (see the module docs). `Mutex`
/// serializes the single-writer engine; reads also take the lock because the
/// engine's lazy SIMD-layout build mutates internal caches on first search.
fn registry() -> &'static Mutex<HashMap<String, SegmentedIndex>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, SegmentedIndex>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

// ─── ctid ⇄ u64 bridge ───────────────────────────────────────────────────────
//
// Postgres hands us a `tid` as `pg_sys::ItemPointerData` (32-bit block + 16-bit
// offset). We pack it into the engine's `u64` id space with the pure, tested
// `turbovec_pg_engine::tid` helpers, so the same heap tuple always maps to the
// same vector id and search results carry a real ctid back to SQL.

fn ctid_to_u64(ctid: pg_sys::ItemPointerData) -> u64 {
    let block = pgrx::itemptr::item_pointer_get_block_number(&ctid);
    let offset = pgrx::itemptr::item_pointer_get_offset_number(&ctid);
    tid::pack(block, offset)
}

fn u64_to_ctid(v: u64) -> pg_sys::ItemPointerData {
    let (block, offset) = tid::unpack(v);
    // Zero-initialize then set every field; `item_pointer_set_all` writes the
    // block-id hi/lo words and the offset, so no field is left uninitialized.
    let mut ctid = unsafe { std::mem::zeroed::<pg_sys::ItemPointerData>() };
    pgrx::itemptr::item_pointer_set_all(&mut ctid, block, offset);
    ctid
}

fn usize_arg(label: &str, v: i32) -> usize {
    usize::try_from(v).unwrap_or_else(|_| error!("{label} must be a positive integer, got {v}"))
}

// ─── SQL surface ─────────────────────────────────────────────────────────────

/// Create an empty index in this backend.
///
/// `dim` must be a positive multiple of 8; `bit_width` ∈ {2,3,4}. `max_hot`
/// is the soft cap at which the writable "hot" segment is sealed.
#[pg_extern]
fn turbovec_create(
    name: &str,
    dim: i32,
    bit_width: default!(i32, 4),
    max_hot: default!(i32, 50000),
) -> bool {
    let idx = SegmentedIndex::new(
        usize_arg("dim", dim),
        usize_arg("bit_width", bit_width),
        usize_arg("max_hot", max_hot),
    )
    .unwrap_or_else(|e| error!("turbovec_create('{name}'): {e}"));

    let mut reg = registry().lock().unwrap();
    if reg.contains_key(name) {
        error!("turbovec index '{name}' already exists in this backend");
    }
    reg.insert(name.to_string(), idx);
    true
}

/// Add one heap row's embedding, addressed by its `ctid`.
#[pg_extern]
fn turbovec_add(name: &str, ctid: pg_sys::ItemPointerData, embedding: Vec<f32>) -> bool {
    let id = ctid_to_u64(ctid);
    let mut reg = registry().lock().unwrap();
    let idx = reg
        .get_mut(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    idx.add(&embedding, &[id])
        .unwrap_or_else(|e| error!("turbovec_add('{name}'): {e}"));
    true
}

/// Remove the vector for a heap row by its `ctid`. Returns `true` if present.
#[pg_extern]
fn turbovec_delete(name: &str, ctid: pg_sys::ItemPointerData) -> bool {
    let id = ctid_to_u64(ctid);
    let mut reg = registry().lock().unwrap();
    let idx = reg
        .get_mut(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    idx.remove(id)
}

/// Top-`k` nearest rows for `query`, as `(ctid, score)`. Join back to the
/// source table on `ctid` to recover the rows.
#[pg_extern]
fn turbovec_search(
    name: &str,
    query: Vec<f32>,
    k: i32,
) -> TableIterator<'static, (name!(ctid, pg_sys::ItemPointerData), name!(score, f32))> {
    let k = usize_arg("k", k);
    let reg = registry().lock().unwrap();
    let idx = reg
        .get(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    let rows: Vec<(pg_sys::ItemPointerData, f32)> = idx
        .search(&query, k)
        .into_iter()
        .map(|(id, score)| (u64_to_ctid(id), score))
        .collect();
    TableIterator::new(rows.into_iter())
}

/// Top-`k` nearest rows for `query`, restricted to the heap tuples in
/// `allowlist` — the hybrid-RAG path. A SQL `WHERE` / tenant / ACL predicate
/// produces the candidate ctids; turbovec ranks only within them and skips
/// whole segments that own none of the allowed tuples (so a selective filter
/// avoids most of the SIMD work instead of over-fetching and discarding).
///
/// ctids in the allowlist that are no longer indexed (deleted, or never
/// added) are ignored.
//
// NOTE: `Vec<pg_sys::ItemPointerData>` maps to SQL `tid[]`. If a given pgrx
// release prefers `pgrx::Array<'_, pg_sys::ItemPointerData>` for array params,
// swap the parameter type and iterate the Array instead — the body is identical.
#[pg_extern]
fn turbovec_search_filtered(
    name: &str,
    query: Vec<f32>,
    k: i32,
    allowlist: Vec<pg_sys::ItemPointerData>,
) -> TableIterator<'static, (name!(ctid, pg_sys::ItemPointerData), name!(score, f32))> {
    let k = usize_arg("k", k);
    let allowed: Vec<u64> = allowlist.into_iter().map(ctid_to_u64).collect();
    let reg = registry().lock().unwrap();
    let idx = reg
        .get(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    let rows: Vec<(pg_sys::ItemPointerData, f32)> = idx
        .search_with_allowlist(&query, k, &allowed)
        .into_iter()
        .map(|(id, score)| (u64_to_ctid(id), score))
        .collect();
    TableIterator::new(rows.into_iter())
}

/// Number of live vectors in the index.
#[pg_extern]
fn turbovec_size(name: &str) -> i64 {
    let reg = registry().lock().unwrap();
    let idx = reg
        .get(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    idx.len() as i64
}

/// Persist the index (per-segment `.tvim` files + manifest) under `dir`.
#[pg_extern]
fn turbovec_save(name: &str, dir: &str) -> bool {
    let reg = registry().lock().unwrap();
    let idx = reg
        .get(name)
        .unwrap_or_else(|| error!("no turbovec index '{name}' in this backend"));
    idx.save(dir)
        .unwrap_or_else(|e| error!("turbovec_save('{name}', '{dir}'): {e}"));
    true
}

/// Load an index previously written by [`turbovec_save`] into this backend
/// under `name` (replacing any existing index of that name).
#[pg_extern]
fn turbovec_load(name: &str, dir: &str) -> bool {
    let idx =
        SegmentedIndex::load(dir).unwrap_or_else(|e| error!("turbovec_load('{dir}'): {e}"));
    registry().lock().unwrap().insert(name.to_string(), idx);
    true
}

// ─── In-database tests (cargo pgrx test) ─────────────────────────────────────

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn create_add_search_join_by_ctid() {
        Spi::run("SELECT turbovec_create('t', 8, 4, 16)").unwrap();
        Spi::run("SELECT turbovec_add('t', '(0,1)'::tid, ARRAY[1,0,0,0,0,0,0,0]::real[])")
            .unwrap();
        Spi::run("SELECT turbovec_add('t', '(0,2)'::tid, ARRAY[0,1,0,0,0,0,0,0]::real[])")
            .unwrap();
        assert_eq!(Spi::get_one::<i64>("SELECT turbovec_size('t')").unwrap(), Some(2));

        // Querying with the first vector should return its ctid as the top hit.
        let top = Spi::get_one::<String>(
            "SELECT ctid::text FROM turbovec_search('t', ARRAY[1,0,0,0,0,0,0,0]::real[], 1)",
        )
        .unwrap();
        assert_eq!(top, Some("(0,1)".to_string()));

        // Delete it, and it must no longer be returned.
        assert_eq!(
            Spi::get_one::<bool>("SELECT turbovec_delete('t', '(0,1)'::tid)").unwrap(),
            Some(true),
        );
        let after = Spi::get_one::<String>(
            "SELECT ctid::text FROM turbovec_search('t', ARRAY[1,0,0,0,0,0,0,0]::real[], 1)",
        )
        .unwrap();
        assert_eq!(after, Some("(0,2)".to_string()));
    }

    #[pg_test]
    fn filtered_search_restricts_to_allowlist() {
        Spi::run("SELECT turbovec_create('f', 8, 4, 16)").unwrap();
        Spi::run("SELECT turbovec_add('f', '(0,1)'::tid, ARRAY[1,0,0,0,0,0,0,0]::real[])")
            .unwrap();
        Spi::run("SELECT turbovec_add('f', '(0,2)'::tid, ARRAY[0,1,0,0,0,0,0,0]::real[])")
            .unwrap();
        Spi::run("SELECT turbovec_add('f', '(0,3)'::tid, ARRAY[0,0,1,0,0,0,0,0]::real[])")
            .unwrap();

        // The true nearest to the query is (0,1), but the allowlist excludes
        // it — so the top hit must come from {(0,2),(0,3)} instead.
        let top = Spi::get_one::<String>(
            "SELECT ctid::text FROM turbovec_search_filtered('f', \
             ARRAY[1,0,0,0,0,0,0,0]::real[], 5, ARRAY['(0,2)','(0,3)']::tid[])",
        )
        .unwrap();
        assert!(top.is_some(), "filtered search returned no rows");
        assert_ne!(
            top,
            Some("(0,1)".to_string()),
            "allowlist failed to exclude (0,1)",
        );
    }
}

/// pgrx test harness scaffolding (required by `cargo pgrx test`).
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {}

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
