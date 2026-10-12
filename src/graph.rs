use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Compressed sparse row representation of a directed graph's adjacency.
/// `row_offsets[v]..row_offsets[v+1]` indexes into `col_indices` for v's neighbors.
pub struct Csr {
    pub row_offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
}

pub struct Graph {
    pub num_vertices: u32,
    pub num_edges: u32,
    /// Out-neighbors, used by BFS / WCC.
    pub out: Csr,
    /// In-neighbors, used by PageRank (each vertex sums contributions from predecessors).
    pub inn: Csr,
    pub out_degree: Vec<u32>,
    pub raw_ids: Vec<u64>,
}

impl Graph {
    /// Loads a SNAP-format edge list: lines starting with '#' are comments,
    /// the rest are whitespace-separated "src dst" pairs. SNAP vertex ids are
    /// sparse (not 0..n), so they're remapped to a dense 0..num_vertices range.
    pub fn load_snap_edgelist(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut id_map: HashMap<u64, u32> = HashMap::new();
        let mut edges: Vec<(u32, u32)> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(a), Some(b)) = (parts.next(), parts.next()) else {
                continue;
            };
            let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) else {
                continue;
            };
            let next_id = id_map.len() as u32;
            let src = *id_map.entry(a).or_insert(next_id);
            let next_id = id_map.len() as u32;
            let dst = *id_map.entry(b).or_insert(next_id);
            edges.push((src, dst));
        }

        let num_vertices = id_map.len() as u32;
        let num_edges = edges.len() as u32;

        let out = build_csr(num_vertices, edges.iter().map(|&(s, d)| (s, d)));
        let inn = build_csr(num_vertices, edges.iter().map(|&(s, d)| (d, s)));
        let out_degree = (0..num_vertices as usize)
            .map(|v| out.row_offsets[v + 1] - out.row_offsets[v])
            .collect();

        let mut raw_ids = vec![0u64; num_vertices as usize];
        for (&raw, &dense) in &id_map {
            raw_ids[dense as usize] = raw;
        }

        Ok(Graph {
            num_vertices,
            num_edges,
            out,
            inn,
            out_degree,
            raw_ids,
        })
    }

    /// Sorted undirected neighbors with larger ids only, so each triangle is counted once.
    pub fn forward_adjacency(&self) -> Csr {
        let n = self.num_vertices as usize;
        let mut lists: Vec<Vec<u32>> = Vec::with_capacity(n);

        for v in 0..n {
            let mut neighbors: Vec<u32> = Vec::new();
            let out_start = self.out.row_offsets[v] as usize;
            let out_end = self.out.row_offsets[v + 1] as usize;
            neighbors.extend_from_slice(&self.out.col_indices[out_start..out_end]);
            let in_start = self.inn.row_offsets[v] as usize;
            let in_end = self.inn.row_offsets[v + 1] as usize;
            neighbors.extend_from_slice(&self.inn.col_indices[in_start..in_end]);
            neighbors.sort_unstable();
            neighbors.dedup();
            neighbors.retain(|&u| u as usize > v);
            lists.push(neighbors);
        }

        let mut row_offsets = vec![0u32; n + 1];
        for v in 0..n {
            row_offsets[v + 1] = row_offsets[v] + lists[v].len() as u32;
        }
        let mut col_indices = Vec::with_capacity(row_offsets[n] as usize);
        for list in lists {
            col_indices.extend(list);
        }

        Csr {
            row_offsets,
            col_indices,
        }
    }
}

fn build_csr(num_vertices: u32, edges: impl Iterator<Item = (u32, u32)> + Clone) -> Csr {
    let mut offsets = vec![0u32; num_vertices as usize + 1];
    for (src, _) in edges.clone() {
        offsets[src as usize + 1] += 1;
    }
    for i in 0..num_vertices as usize {
        offsets[i + 1] += offsets[i];
    }

    let mut cursor = offsets.clone();
    let mut col_indices = vec![0u32; offsets[num_vertices as usize] as usize];
    for (src, dst) in edges {
        let pos = cursor[src as usize];
        col_indices[pos as usize] = dst;
        cursor[src as usize] += 1;
    }

    Csr {
        row_offsets: offsets,
        col_indices,
    }
}

#[cfg(test)]
mod tests;
