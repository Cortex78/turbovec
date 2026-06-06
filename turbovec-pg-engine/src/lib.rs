//! Engine for a turbovec-backed Postgres index (proof of concept).
//!
//! This crate is pure Rust — it depends only on the `turbovec` core crate,
//! never on `pgrx` or Postgres — so the two hard pieces of a deep Postgres
//! integration can be developed and unit-tested in isolation:
//!
//! * [`segment`] — a segment-based ("LSM-style") build layer that turns
//!   turbovec's *rebuild-the-whole-SIMD-layout-on-every-mutation* index
//!   into something that accepts incremental inserts and deletes without
//!   re-encoding the entire corpus on each write.
//! * [`tid`] — packing a Postgres `ItemPointer` (ctid: 32-bit block +
//!   16-bit offset) into the `u64` id that `turbovec::IdMapIndex` uses, so
//!   an ANN hit maps straight back to a heap row.
//!
//! The thin `#[pg_extern]` FFI shim that exposes these over SQL lives in
//! the separate `turbovec-pg` crate.

pub mod segment;
pub mod tid;

pub use segment::SegmentedIndex;
