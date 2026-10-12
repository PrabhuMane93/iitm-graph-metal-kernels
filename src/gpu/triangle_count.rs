use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::graph::Graph;

const SHADER_SRC: &str = include_str!("../shaders/triangle_count.metal");
const THREADS_PER_GROUP: usize = 256;

/// Triangle counting on the Metal GPU, one thread per forward edge.
pub fn triangle_count(graph: &Graph) -> u64 {
    let n = graph.num_vertices as usize;
    if n == 0 {
        return 0;
    }

    let forward = graph.forward_adjacency();
    let num_edges = forward.col_indices.len();
    if num_edges == 0 {
        return 0;
    }

    // edge_src[e] = owning vertex of forward-adjacency slot e.
    let mut edge_src = vec![0u32; num_edges];
    for v in 0..n {
        let start = forward.row_offsets[v] as usize;
        let end = forward.row_offsets[v + 1] as usize;
        edge_src[start..end].fill(v as u32);
    }

    // ── 1. SETUP ────────────────────────────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source_src = NSString::from_str(SHADER_SRC);
    let library = device
        .newLibraryWithSource_options_error(&source_src, None)
        .expect("triangle_count.metal failed to compile");
    let function = library
        .newFunctionWithName(&NSString::from_str("triangle_count"))
        .expect("triangle_count kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. BUFFERS ──────────────────────────────────────────────────
    let row_offsets_buf = device
        .newBufferWithLength_options(
            (forward.row_offsets.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("row_offsets_buf alloc failed");
    let col_indices_buf = device
        .newBufferWithLength_options(
            (forward.col_indices.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("col_indices_buf alloc failed");
    let edge_src_buf = device
        .newBufferWithLength_options(
            (edge_src.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("edge_src_buf alloc failed");
    let total_buf = device
        .newBufferWithLength_options(
            size_of::<u32>() as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("total_buf alloc failed");

    unsafe {
        let ptr = row_offsets_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, forward.row_offsets.len())
            .copy_from_slice(&forward.row_offsets);

        let ptr = col_indices_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, forward.col_indices.len())
            .copy_from_slice(&forward.col_indices);

        let ptr = edge_src_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, edge_src.len()).copy_from_slice(&edge_src);

        let ptr = total_buf.contents().as_ptr() as *mut u32;
        *ptr = 0;
    }

    // ── 3. DISPATCH, one thread per forward edge ───────────────────────
    let num_edges_u32 = num_edges as u32;
    let num_edges_ptr: NonNull<c_void> = NonNull::from(&num_edges_u32).cast();

    let threadgroups = MTLSize {
        width: num_edges.div_ceil(THREADS_PER_GROUP),
        height: 1,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: THREADS_PER_GROUP,
        height: 1,
        depth: 1,
    };

    let cmd_buffer = queue.commandBuffer().expect("no command buffer");
    let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
    encoder.setComputePipelineState(&pipeline);
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&row_offsets_buf), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&col_indices_buf), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&edge_src_buf), 0, 2);
        encoder.setBytes_length_atIndex(num_edges_ptr, size_of::<u32>() as NSUInteger, 3);
        encoder.setBuffer_offset_atIndex(Some(&total_buf), 0, 4);
    }
    encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
    encoder.endEncoding();
    cmd_buffer.commit();
    cmd_buffer.waitUntilCompleted();

    // ── 4. READ BACK ────────────────────────────────────────────────
    unsafe { *(total_buf.contents().as_ptr() as *const u32) as u64 }
}
