//! Postgres ctid <-> `u64` packing.
//!
//! A PostgreSQL heap tuple is addressed by an `ItemPointer` (its `ctid`):
//! a 32-bit block number plus a 16-bit item offset within that block.
//! `turbovec::IdMapIndex` addresses every vector by a stable `u64`.
//!
//! Packing `(block, offset)` into the low 48 bits of a `u64` lets a turbovec
//! id *be* a heap tuple pointer. An ANN result then maps straight back to a
//! table row with `WHERE ctid = '(block,offset)'` — no surrogate id column,
//! no second lookup table. This is the same trick a real Postgres index AM
//! uses to point index entries at heap tuples.
//!
//! ```text
//! u64 layout:  [ 16 unused bits | 32-bit block number | 16-bit offset ]
//!               63            48 47                   16 15           0
//! ```

/// Pack a `(block, offset)` heap pointer into a `u64`.
#[inline]
pub fn pack(block: u32, offset: u16) -> u64 {
    ((block as u64) << 16) | (offset as u64)
}

/// Inverse of [`pack`]: recover `(block, offset)` from a packed `u64`.
///
/// Only the low 48 bits are interpreted; the top 16 bits are ignored, so
/// this is a true inverse of [`pack`] for every `(u32, u16)` input.
#[inline]
pub fn unpack(v: u64) -> (u32, u16) {
    let block = ((v >> 16) & 0xFFFF_FFFF) as u32;
    let offset = (v & 0xFFFF) as u16;
    (block, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_boundary_values() {
        // (block, offset) pairs spanning the full range of each field,
        // including the values Postgres actually uses (offsets start at 1).
        let cases = [
            (0u32, 0u16),
            (0, 1),
            (1, 1),
            (42, 7),
            (u32::MAX, u16::MAX),
            (u32::MAX, 0),
            (0, u16::MAX),
            (123_456, 200),
            (1 << 31, 1),
        ];
        for (block, offset) in cases {
            let packed = pack(block, offset);
            assert_eq!(
                unpack(packed),
                (block, offset),
                "round-trip failed for block={block}, offset={offset}",
            );
        }
    }

    #[test]
    fn distinct_pointers_pack_to_distinct_ids() {
        // Adjacent offsets and adjacent blocks must never collide — a
        // collision would alias two heap tuples onto one vector id.
        assert_ne!(pack(0, 1), pack(0, 2));
        assert_ne!(pack(0, 1), pack(1, 0));
        assert_ne!(pack(1, 0), pack(0, 1 << 0));
        // The block field sits above the 16-bit offset field, so bumping
        // the block by one moves the id by exactly 2^16.
        assert_eq!(pack(5, 0) + (1 << 16), pack(6, 0));
    }

    #[test]
    fn high_bits_are_ignored_on_unpack() {
        // Set the unused top 16 bits and confirm unpack still recovers the
        // original pointer — keeps unpack a faithful inverse even if a
        // caller ever tags the high bits.
        let packed = pack(777, 9) | (0xABCDu64 << 48);
        assert_eq!(unpack(packed), (777, 9));
    }
}
