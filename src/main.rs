mod gpu;
mod graph;

use gpu::pagerank::PageRankConfig;
use graph::Graph;
use std::time::Instant;

/// Usage: iitm_graph_metal_kernels <path-to-snap-edgelist>
///
/// Loads the graph and runs PageRank, BFS, and Triangle Counting on the
/// Metal GPU, printing wall-clock time and the computed result for each.
/// Neo4j GDS is the tested CPU baseline (see the project report), not a
/// local reference implementation.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let graph_path = args.get(1).unwrap_or_else(|| {
        eprintln!("usage: iitm_graph_metal_kernels <path-to-snap-edgelist>");
        std::process::exit(1);
    });

    let graph = Graph::load_snap_edgelist(graph_path)
        .unwrap_or_else(|e| panic!("failed to load graph at {graph_path}: {e}"));
    println!(
        "loaded {graph_path}: {} vertices, {} edges",
        graph.num_vertices, graph.num_edges
    );

    run_pagerank(&graph);
    println!();
    run_bfs(&graph);
    println!();
    run_triangle_count(&graph);
}

fn run_pagerank(graph: &Graph) {
    let config = PageRankConfig::default();

    let gpu_start = Instant::now();
    let gpu_rank = gpu::pagerank::page_rank(graph, &config);
    let gpu_elapsed = gpu_start.elapsed();

    let rank_sum: f32 = gpu_rank.iter().sum();
    println!(
        "GPU pagerank: {:.3} ms (rank sum {:.6})",
        gpu_elapsed.as_secs_f64() * 1000.0,
        rank_sum
    );
}

/// Picks the highest out-degree vertex as the BFS source.
fn pick_bfs_source(graph: &Graph) -> u32 {
    graph
        .out_degree
        .iter()
        .enumerate()
        .max_by_key(|&(_, &deg)| deg)
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

fn run_bfs(graph: &Graph) {
    let source = pick_bfs_source(graph);
    println!(
        "BFS source: internal vertex {source}, raw SNAP id {} (out-degree {})",
        graph.raw_ids[source as usize], graph.out_degree[source as usize]
    );

    let gpu_start = Instant::now();
    let gpu_levels = gpu::bfs::bfs(graph, source);
    let gpu_elapsed = gpu_start.elapsed();

    let reached = gpu_levels.iter().filter(|&&level| level != -1).count();

    println!(
        "reached {reached} / {} vertices",
        graph.num_vertices as usize
    );
    println!("GPU bfs: {:.3} ms", gpu_elapsed.as_secs_f64() * 1000.0);
}

fn run_triangle_count(graph: &Graph) {
    let gpu_start = Instant::now();
    let gpu_count = gpu::triangle_count::triangle_count(graph);
    let gpu_elapsed = gpu_start.elapsed();

    println!(
        "GPU triangle_count: {:.3} ms ({gpu_count} triangles)",
        gpu_elapsed.as_secs_f64() * 1000.0
    );
}
