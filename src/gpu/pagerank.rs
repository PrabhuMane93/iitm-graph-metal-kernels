use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::graph::Graph;

const SHADER_SRC: &str = include_str!("../shaders/pagerank.metal");
const THREADS_PER_GROUP: usize = 256;

pub struct PageRankConfig {
    pub damping_factor: f32,
    pub max_iterations: u32,
    pub tolerance: f32,
}

impl Default for PageRankConfig {
    fn default() -> Self {
        Self {
            damping_factor: 0.85,
            max_iterations: 20,
            tolerance: 1e-4,
        }
    }
}

/// Runs PageRank on the Metal GPU, over the in-neighbor CSR with standard
/// dangling-node redistribution.
pub fn page_rank(graph: &Graph, config: &PageRankConfig) -> Vec<f32> {
    let n = graph.num_vertices as usize;
    if n == 0 {
        return Vec::new();
    }

    // ── 1. SETUP ────────────────────────────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source = NSString::from_str(SHADER_SRC);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("pagerank.metal failed to compile");
    let function = library
        .newFunctionWithName(&NSString::from_str("pagerank_iteration"))
        .expect("pagerank_iteration kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. BUFFERS ──────────────────────────────────────────────────
    let row_offsets_buf = device
        .newBufferWithLength_options(
            (graph.inn.row_offsets.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("row_offsets_buf alloc failed");
    let col_indices_buf = device
        .newBufferWithLength_options(
            (graph.inn.col_indices.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("col_indices_buf alloc failed");
    let out_degree_buf = device
        .newBufferWithLength_options(
            (graph.out_degree.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("out_degree_buf alloc failed");
    let mut rank_in_buf = device
        .newBufferWithLength_options(
            (n * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("rank_in_buf alloc failed");
    let mut rank_out_buf = device
        .newBufferWithLength_options(
            (n * size_of::<f32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("rank_out_buf alloc failed");

    unsafe {
        let ptr = row_offsets_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, graph.inn.row_offsets.len())
            .copy_from_slice(&graph.inn.row_offsets);

        let ptr = col_indices_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, graph.inn.col_indices.len())
            .copy_from_slice(&graph.inn.col_indices);

        let ptr = out_degree_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, graph.out_degree.len())
            .copy_from_slice(&graph.out_degree);

        let ptr = rank_in_buf.contents().as_ptr() as *mut f32;
        std::slice::from_raw_parts_mut(ptr, n).fill(1.0f32 / n as f32);
    }

    // ── 3. DISPATCH, iterated to convergence (or max_iterations) ─────
    let base = (1.0 - config.damping_factor) / n as f32;
    let num_vertices = n as u32;
    let damping_ptr: NonNull<c_void> = NonNull::from(&config.damping_factor).cast();
    let base_ptr: NonNull<c_void> = NonNull::from(&base).cast();
    let num_vertices_ptr: NonNull<c_void> = NonNull::from(&num_vertices).cast();

    let threadgroups = MTLSize {
        width: n.div_ceil(THREADS_PER_GROUP),
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: THREADS_PER_GROUP,
        height: 1,
        depth: 1,
    };

    for _ in 0..config.max_iterations {
        // Dangling mass is O(n), cheap enough to compute CPU-side via unified memory.
        let dangling_sum: f32 = unsafe {
            let ptr = rank_in_buf.contents().as_ptr() as *const f32;
            let rank_in_view = std::slice::from_raw_parts(ptr, n);
            (0..n)
                .filter(|&v| graph.out_degree[v] == 0)
                .map(|v| rank_in_view[v])
                .sum()
        };
        let dangling_contrib = config.damping_factor * dangling_sum / n as f32;
        let dangling_contrib_ptr: NonNull<c_void> = NonNull::from(&dangling_contrib).cast();

        let cmd_buffer = queue.commandBuffer().expect("no command buffer");
        let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&row_offsets_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&col_indices_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out_degree_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&rank_in_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&rank_out_buf), 0, 4);
            encoder.setBytes_length_atIndex(damping_ptr, size_of::<f32>() as NSUInteger, 5);
            encoder.setBytes_length_atIndex(base_ptr, size_of::<f32>() as NSUInteger, 6);
            encoder.setBytes_length_atIndex(
                dangling_contrib_ptr,
                size_of::<f32>() as NSUInteger,
                7,
            );
            encoder.setBytes_length_atIndex(num_vertices_ptr, size_of::<u32>() as NSUInteger, 8);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

        std::mem::swap(&mut rank_in_buf, &mut rank_out_buf);
    }

    // ── 4. READ BACK ────────────────────────────────────────────────
    unsafe {
        let ptr = rank_in_buf.contents().as_ptr() as *const f32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}
