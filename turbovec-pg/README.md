# turbovec-pg — a turbovec-backed Postgres vector index (Path C PoC)

A **proof of concept** for running turbovec's quantizer + SIMD search *inside*
PostgreSQL, as a pgvector-style vector index for embeddings / RAG. It is the
"Path C" option from the integration review: a native Postgres extension that
reuses the `turbovec` core crate, rather than calling turbovec from an
application sidecar.

> **Status: experimental.** This demonstrates feasibility and the two hard
> pieces (ctid mapping + an incremental segment build). It is **not** a
> drop-in pgvector replacement yet — see [Limitations](#limitations).

## Two crates

| Crate | In workspace? | Tested in CI/sandbox? | Role |
|---|---|---|---|
| [`turbovec-pg-engine`](../turbovec-pg-engine) | ✅ yes | ✅ `cargo test` (10 tests pass) | Pure Rust. The logic that does **not** need Postgres: the segment-based build engine + the ctid↔u64 mapping. |
| `turbovec-pg` (this crate) | ❌ excluded | ⚙️ built with `cargo pgrx` | Thin `#[pg_extern]` FFI shim. Registers SQL functions, bridges Postgres `tid`↔`u64`, holds live indexes. |

Splitting this way means the genuinely novel logic is unit-tested with plain
`cargo test` (no Postgres required), and the pgrx layer stays a small, obvious
glue file.

## The three pillars of the PoC

1. **FFI shim** — `src/lib.rs`. `#[pg_extern]` functions:
   `turbovec_create`, `turbovec_add`, `turbovec_delete`, `turbovec_search`,
   `turbovec_search_filtered` (hybrid allowlist), `turbovec_size`,
   `turbovec_save` / `turbovec_sync` / `turbovec_load` (durable store),
   `turbovec_compact` (reclaim deleted space).
2. **ctid ⇄ u64 mapping** — [`turbovec-pg-engine/src/tid.rs`](../turbovec-pg-engine/src/tid.rs).
   A Postgres heap pointer (32-bit block + 16-bit offset) packs into the low 48
   bits of the `u64` that `turbovec::IdMapIndex` already uses for stable ids. So
   **the ctid is the vector id** — a search result joins straight back to the
   table with `JOIN docs d ON d.ctid = h.ctid`, exactly how a real index AM
   points at heap tuples.
3. **Segment-based build** — [`turbovec-pg-engine/src/segment.rs`](../turbovec-pg-engine/src/segment.rs).
   This is the heart of it (next section).

## Why the segment layer matters

turbovec's index is *load-once, query-many*: every `add`/`remove` invalidates
the SIMD-blocked code layout, which is then rebuilt over the **whole corpus** on
the next search (`TurboQuantIndex` resets its `blocked` `OnceLock`). That is
fatal for a database that inserts one tuple at a time.

`SegmentedIndex` applies the log-structured-merge idea:

```
            ┌─────────── sealed (immutable-for-insert) ───────────┐   ┌── hot ──┐
 inserts ─► │  seg 0       seg 1       seg 2       …               │   │  open   │ ◄─ new rows land here
            │  (layout     (layout     (layout                     │   │ (small) │
            │   built once) built once) built once)                │   └─────────┘
            └─────────────────────────────────────────────────────┘        │
                                                                   seal when ≥ max_hot
 search(q,k):  ask every segment for its local top-k  ──►  merge  ──►  global top-k
```

* **Inserts** touch only the small hot segment; the layout rebuild on seal is
  bounded by `max_hot`, not by total corpus size.
* **Searches** fan out across segments and merge. The merge is *exact*: a
  global top-k element is, by definition, in the top-k of its own segment.
  (Verified against a monolithic `IdMapIndex` in the engine tests.)
* **Deletes** are O(1): routed to the one segment that owns the id via
  `IdMapIndex::remove`; only that segment's layout rebuilds on its next search.

## How the PoC maps to the review's blockers

From the integration review, the five things a deep Postgres integration must
confront, and where this PoC stands:

| # | Blocker | PoC status |
|---|---|---|
| 1 | Rebuild-the-whole-layout on every write | **Addressed** — segment build bounds rebuilds to `max_hot` / one segment. |
| 5 | No C ABI (Rust + PyO3 only) | **Addressed** — pgrx provides the SQL-callable FFI surface. |
| — | Stable id ↔ heap tuple | **Addressed** — ctid↔u64 mapping (`IdMapIndex` was already well-suited). |
| 4 | No persistence / crash-safety | **Addressed at file level** — incremental, write-once segments committed by an atomic `CURRENT` rename (a crash leaves the prior generation intact); any process re-opens the committed state. Integration with Postgres' own WAL still pending. |
| 2 | Brute-force flat scan (O(N)) | **Partially mitigated** — a selective `search_filtered` allowlist skips whole segments; unfiltered search is still a full SIMD scan (no IVF/HNSW coarse structure). |
| 3 | Per-backend rotation matrix / `rayon` pool | **Unchanged** — inherited from core. |
| — | MVCC visibility, planner integration | **Out of scope** — see below. |

## Build & run

Prerequisites: a Rust toolchain and Postgres build deps (`bison`, `flex`,
`libclang`, `readline`, `zlib`, …). Then:

```bash
cargo install cargo-pgrx --version 0.12.9
cargo pgrx init                       # downloads & builds pinned Postgres(es)

cd turbovec-pg
cargo pgrx run pg17                   # compile, install into a scratch PG, open psql
#   ... or install into an existing cluster:
# cargo pgrx install --release --pg-config /usr/lib/postgresql/17/bin/pg_config
```

In the `psql` that `cargo pgrx run` opens, walk through [`sql/demo.sql`](sql/demo.sql):

```sql
CREATE EXTENSION turbovec_pg;
SELECT turbovec_create('docs_idx', 1536, 4);
SELECT turbovec_add('docs_idx', ctid, embedding) FROM docs;
SELECT d.id, d.title, h.score
FROM turbovec_search('docs_idx', (SELECT embedding FROM docs WHERE id=1), 10) h
JOIN docs d ON d.ctid = h.ctid
ORDER BY h.score DESC;
```

In-database tests:

```bash
cargo pgrx test pg17                  # runs the #[pg_test] suite in src/lib.rs
```

> The pgrx build is **not** run in this repo's sandbox (no `cargo-pgrx`/Postgres
> there). The engine it wraps **is** built and tested: `cargo test -p
> turbovec-pg-engine`.

## Limitations

* **Per-backend live state.** The live index is a process-local static, visible
  only to the connection that built or opened it. `turbovec_save` /
  `turbovec_sync` make it durable and `turbovec_load` lets another backend
  re-open the committed state, but there is no shared *live* view yet — that
  needs shared memory / buffer-manager pages.
* **Not an index access method.** Search is an explicit function call, so the
  planner can't choose it for `ORDER BY embedding <#> q LIMIT k`, and it can't
  combine with a `WHERE` via a bitmap scan.
* **No MVCC; not tied into Postgres' WAL.** No tuple-visibility recheck against
  the snapshot. Persistence *is* crash-safe at the file level (atomic `CURRENT`
  commit), but it is not part of Postgres' transaction/WAL, so a crash rolls
  back to the last `turbovec_sync`, not to the last committed transaction.
* **Manual compaction.** Deletes from sealed segments are tombstoned (sealed
  files are write-once); `turbovec_compact` reclaims the space by rewriting the
  affected segments. There is no automatic compaction trigger or background
  merging of small segments yet.
* **ctid stability.** The PoC treats ctid as a stable handle — true for
  insert-/append-mostly corpora (typical RAG), but `UPDATE` and `VACUUM FULL`
  move tuples. A real AM hooks VACUUM; until then, rebuild after bulk updates.

## Roadmap (PoC → production)

- [x] **Kernel-level filtering (hybrid RAG).** `turbovec_search_filtered(name,
  query, k, allowlist tid[])` partitions the allowlist by segment and pushes
  each subset into turbovec's block-granular kernel filter, skipping segments
  that own no allowed tuples. (Engine: `SegmentedIndex::search_with_allowlist`,
  unit-tested against a monolithic `IdMapIndex`.)
- [~] **Durable, shareable storage.** *Done:* crash-safe, incremental on-disk
  segments with atomic `CURRENT`-pointer commits (`SegmentedIndex::sync`/`open`,
  exposed as `turbovec_save`/`turbovec_sync`/`turbovec_load`); any process
  re-opens the committed state. *Pending:* zero-copy shared memory / buffer-
  manager pages and Postgres-WAL integration so live writes are visible
  cross-backend without a re-open.
- [x] **Global calibration + compaction.** One TQ+ calibration is fit by the
  first sealed segment and shared by every later segment (via `turbovec-core`'s
  new `with_calibration` / `calibration`), so per-segment scores are directly
  comparable; `turbovec_compact` reclaims tombstoned space by rewriting affected
  segments. *Pending:* automatic compaction triggers and background merging of
  small segments.
- [ ] **Real index AM / operator class** (`<#>` inner product, `<=>` cosine) so
  the planner drives it — the step that makes it a true pgvector alternative.
