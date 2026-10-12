# iitm-graph-metal-kernels

Metal GPU implementations of three graph algorithms from Neo4j's Graph Data Science (GDS) library, benchmarked against a live Neo4j GDS instance as the CPU baseline, on large SNAP graphs. See the parent [`../TASK.md`](../TASK.md) and [`../RESOURCES.md`](../RESOURCES.md) for full context, algorithm choices, and dataset candidates.

Full report: [GPU-Accelerated Graph Analytics on Apple Silicon — Methodology and Benchmarks Against Neo4j GDS](https://app.notion.com/p/GPU-Accelerated-Graph-Analytics-on-Apple-Silicon-Methodology-and-Benchmarks-Against-Neo4j-GDS-3f74fe7d13398165a3d9ee265d8857e0?source=copy_link)

## Build order

1. **PageRank** — CSR-based, double-buffered, no atomics.
2. **BFS** — Level-synchronous frontier expansion, atomic compare-and-swap claims on out-neighbors.
3. **Triangle Counting** — Forward-oriented adjacency (`Graph::forward_adjacency`) + one thread per forward edge (not per vertex, to avoid a hub vertex's work serializing onto a single thread), choosing a linear merge or a binary search per edge depending on how skewed the two neighbor lists are.

## Layout

- `src/graph.rs` — SNAP edge-list loader, builds forward + reverse CSR.
- `src/gpu/` — Metal GPU implementations, using the same `objc2-metal` bindings as the sibling [metal-llm-kernels](https://github.com/PrabhuMane93/metal-llm-kernels) project.
- `src/shaders/*.metal` — MSL kernel source, compiled from embedded strings at runtime (`newLibraryWithSource_options_error`), no `.metallib` build step needed.
- `data/` — downloaded SNAP graph files (gitignored; download and convert via convert_to_neo4j.py).
- `results/` — per-algorithm benchmark CSVs (gitignored by default; commit real results deliberately).

## Dataset setup

All ten SNAP edge-list files used for benchmarking are hosted here (large, not committed to this repo): [Google Drive folder](https://drive.google.com/drive/folders/1ZS7AL_u8DSiYeqpGBHP-4JSeORwJqoU-?usp=sharing).

1. Download the edge-list files you want from that folder into `data/` (e.g. `data/com-youtube.ungraph.txt`).
2. Run the Rust kernels directly against a raw edge list — no conversion needed:
   ```
   cargo run --release -- data/com-youtube.ungraph.txt
   ```
3. To reproduce the Neo4j GDS CPU baseline instead, convert every `.txt` file in `data/` to import-ready CSVs in one pass, then bulk-import each graph:
   ```
   python3 data/convert_to_neo4j.py
   neo4j-admin database import full --nodes=data/com-youtube.ungraph-csv/nodes_header.csv,data/com-youtube.ungraph-csv/nodes.csv --relationships=data/com-youtube.ungraph-csv/rels_header.csv,data/com-youtube.ungraph-csv/rels.csv -- neo4j
   ```
   Running `data/convert_to_neo4j.py` with no arguments scans `data/` and writes each graph's CSVs to its own `data/<graph>-csv/` folder.

## Testbench

```
python3 testbench/run_benchmarks.py
```

Builds the release binary, runs it against every `.txt` file in `data/`, and appends one row per graph to `results/pagerank_gpu.csv`, `results/bfs_gpu.csv`, and `results/triangle_count_gpu.csv`.

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
