import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
BINARY = REPO_ROOT / "target" / "release" / "iitm_graph_metal_kernels"

LOADED_RE = re.compile(r"loaded (.+): (\d+) vertices, (\d+) edges")
PAGERANK_RE = re.compile(r"GPU pagerank: ([\d.]+) ms")
BFS_REACHED_RE = re.compile(r"reached (\d+) / (\d+) vertices")
BFS_RE = re.compile(r"GPU bfs: ([\d.]+) ms")
TRIANGLE_RE = re.compile(r"GPU triangle_count: ([\d.]+) ms \((\d+) triangles\)")


def build_binary() -> None:
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT, check=True)


def run_graph(path: Path) -> dict:
    out = subprocess.run(
        [str(BINARY), str(path)], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout

    loaded = LOADED_RE.search(out)
    pagerank_ms = PAGERANK_RE.search(out)
    reached = BFS_REACHED_RE.search(out)
    bfs_ms = BFS_RE.search(out)
    triangle = TRIANGLE_RE.search(out)

    return {
        "graph": path.name,
        "num_vertices": loaded.group(2),
        "num_edges": loaded.group(3),
        "pagerank_gpu_ms": pagerank_ms.group(1),
        "bfs_gpu_ms": bfs_ms.group(1),
        "bfs_reached": reached.group(1),
        "triangle_count_gpu_ms": triangle.group(1),
        "triangle_count": triangle.group(2),
    }


def write_row(csv_path: Path, fieldnames: list, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    build_binary()

    graphs = sorted(DATA_DIR.glob("*.txt"))
    if not graphs:
        print(f"no .txt graphs found in {DATA_DIR}")
        sys.exit(1)

    for path in graphs:
        print(f"running {path.name}...")
        result = run_graph(path)

        write_row(
            RESULTS_DIR / "pagerank_gpu.csv",
            ["graph", "num_vertices", "num_edges", "pagerank_gpu_ms"],
            {k: result[k] for k in ("graph", "num_vertices", "num_edges", "pagerank_gpu_ms")},
        )
        write_row(
            RESULTS_DIR / "bfs_gpu.csv",
            ["graph", "num_vertices", "num_edges", "bfs_gpu_ms", "bfs_reached"],
            {k: result[k] for k in ("graph", "num_vertices", "num_edges", "bfs_gpu_ms", "bfs_reached")},
        )
        write_row(
            RESULTS_DIR / "triangle_count_gpu.csv",
            ["graph", "num_vertices", "num_edges", "triangle_count_gpu_ms", "triangle_count"],
            {
                k: result[k]
                for k in ("graph", "num_vertices", "num_edges", "triangle_count_gpu_ms", "triangle_count")
            },
        )
        print(f"  pagerank {result['pagerank_gpu_ms']} ms, bfs {result['bfs_gpu_ms']} ms, "
              f"triangle_count {result['triangle_count_gpu_ms']} ms")


if __name__ == "__main__":
    main()
