-- turbovec-pg demo: a turbovec-backed vector index driven from SQL.
--
-- Build & install first (see ../README.md):
--   cargo pgrx install --release          -- into a local Postgres, or
--   cargo pgrx run pg17                    -- spins up a scratch Postgres + psql
--
-- Then, in psql:

CREATE EXTENSION turbovec_pg;

-- ── A document table with embeddings ─────────────────────────────────────────
-- `embedding` is a plain real[] of length `dim` (1536 here, OpenAI-style).
CREATE TABLE docs (
    id        bigserial PRIMARY KEY,
    title     text,
    body      text,
    embedding real[]
);

-- INSERT INTO docs (title, body, embedding) VALUES ...   -- your data + vectors

-- ── 1. Create an in-backend turbovec index ───────────────────────────────────
--   turbovec_create(name, dim, bit_width DEFAULT 4, max_hot DEFAULT 50000)
SELECT turbovec_create('docs_idx', 1536, 4);

-- ── 2. Populate it from the heap, addressing each row by its ctid ─────────────
-- The ctid IS the vector id (packed block+offset), so no surrogate id column
-- is needed inside the index.
SELECT turbovec_add('docs_idx', ctid, embedding)
FROM docs
WHERE embedding IS NOT NULL;

SELECT turbovec_size('docs_idx');   -- how many vectors are indexed

-- ── 3. Search: turbovec returns (ctid, score); join back on ctid ──────────────
WITH q AS (
    SELECT embedding AS v FROM docs WHERE id = 1
)
SELECT d.id, d.title, h.score
FROM q, turbovec_search('docs_idx', (SELECT v FROM q), 10) AS h
JOIN docs d ON d.ctid = h.ctid
ORDER BY h.score DESC;

-- ── Hybrid / RAG-style filtering (kernel-level allowlist) ─────────────────────
-- A SQL predicate (tenant / ACL / time window / full-text) produces the
-- candidate ctids; turbovec ranks ONLY within them and skips whole segments
-- that own none of the allowed tuples — no over-fetch, no post-filter.
SELECT d.id, d.title, h.score
FROM turbovec_search_filtered(
        'docs_idx',
        (SELECT embedding FROM docs WHERE id = 1),
        10,
        ARRAY(SELECT ctid FROM docs WHERE title ILIKE '%postgres%')   -- candidate set
     ) h
JOIN docs d ON d.ctid = h.ctid
ORDER BY h.score DESC;

-- ── Deletes & persistence ─────────────────────────────────────────────────────
SELECT turbovec_delete('docs_idx', ctid) FROM docs WHERE id = 1;

-- Persist to disk (per-segment .tvim files + manifest.json) and reload. This
-- is how an index survives a backend restart in the PoC, since state is
-- per-backend in memory.
SELECT turbovec_save('docs_idx', '/var/lib/postgresql/turbovec/docs_idx');
SELECT turbovec_load('docs_idx', '/var/lib/postgresql/turbovec/docs_idx');

-- ⚠ ctid stability: this PoC treats ctid as a stable handle. That holds for
-- insert-/append-mostly corpora (typical for RAG), but UPDATE and
-- VACUUM FULL move tuples to new ctids. A production index AM hooks VACUUM to
-- keep the mapping current; until then, rebuild the index after bulk updates.
