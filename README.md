# iitm-graph-metal-kernels

Metal GPU implementations of three graph algorithms from Neo4j's Graph Data Science (GDS) library, benchmarked against a live Neo4j GDS instance as the CPU baseline, on large SNAP graphs. See the parent [`../TASK.md`](../TASK.md) and [`../RESOURCES.md`](../RESOURCES.md) for full context, algorithm choices, and dataset candidates.

## Build order

1. **PageRank** — done. CSR-based, double-buffered, no atomics.
2. **BFS** — done. Level-synchronous frontier expansion, atomic compare-and-swap claims on out-neighbors.
3. **Triangle Counting** — done. Forward-oriented adjacency (`Graph::forward_adjacency`) + one thread per forward edge (not per vertex, to avoid a hub vertex's work serializing onto a single thread), choosing a linear merge or a binary search per edge depending on how skewed the two neighbor lists are.

## Layout

- `src/graph.rs` — SNAP edge-list loader, builds forward + reverse CSR.
- `src/cpu/` — single-threaded reference implementations for PageRank and BFS, used only as a local correctness cross-check, not as the reported baseline. There is no CPU Triangle Counting implementation here: Neo4j GDS's `gds.triangleCount.stats` is the tested baseline for that algorithm (see Results).
- `src/gpu/` — Metal GPU implementations, using the same `objc2-metal` bindings as the sibling [metal-llm-kernels](https://github.com/PrabhuMane93/metal-llm-kernels) project.
- `src/shaders/*.metal` — MSL kernel source, compiled from embedded strings at runtime (`newLibraryWithSource_options_error`), no `.metallib` build step needed.
- `data/` — downloaded SNAP graph files (gitignored; fetch via `scripts/`).
- `results/` — per-algorithm benchmark CSVs (gitignored by default; commit real results deliberately).

Raw SNAP datasets for all ten benchmark graphs are downloaded to [`../dataset_preparation`](../dataset_preparation) (a sibling folder, outside this repo). That folder also holds the script that converts an edge list into the node/relationship CSVs used for `neo4j-admin database import`. Those raw dataset files are large and are not committed to this repo.

## Running

```
cargo run --release -- path/to/graph-edgelist.txt
```

Loads the graph, runs PageRank and BFS on CPU then on GPU (Triangle Counting is GPU-only; see above), prints timing + speedup + a correctness check for each (max abs diff for PageRank, mismatched levels for BFS), and appends one row per algorithm to `results/<algorithm>.csv`.

## Correctness

Every GPU kernel is checked against an independent implementation of the same algorithm: PageRank and BFS against the CPU reference in this repo, Triangle Counting against Neo4j GDS's `gds.triangleCount.stats`. Across all ten benchmark graphs, every GPU result matches exactly — zero error on every run.

## Results

CPU timing below is Neo4j GDS's own `computeMillis` (`gds.pageRank.stats`, `gds.bfs.stats`, `gds.triangleCount.stats`), not the local CPU reference. Full per-graph tables and charts are in the project report.

| Algorithm | Speedup range (CPU/GPU) | Notes |
|---|---|---|
| PageRank | 0.69x – 2.81x | GPU wins on 9 of 10 graphs; loses once (wiki-topcats) |
| BFS | 0.23x – 5.31x | Mixed: GPU loses on 5 graphs (dispatch overhead per level dominates on small/sparse ones), wins on the other 5 |
| Triangle Counting | 2.98x – 46.50x | GPU wins on every graph; biggest wins on the most degree-skewed graphs |

All ten graphs have at least one million vertices, drawn from SNAP: com-youtube, wiki-Talk, wiki-talk-temporal, as-skitter, cit-Patents, com-lj, soc-pokec, wiki-topcats, sx-stackoverflow, com-orkut.
