use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::NonNull;

use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::graph::Graph;

const SHADER_SRC: &str = include_str!("../shaders/bfs.metal");
const THREADS_PER_GROUP: usize = 256;

/// Runs level-synchronous BFS on the Metal GPU from `source`, over
/// out-edges.
pub fn bfs(graph: &Graph, source: u32) -> Vec<i32> {
    let n = graph.num_vertices as usize;
    if n == 0 {
        return Vec::new();
    }

    // ── 1. SETUP ────────────────────────────────────────────────────
    let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
    let source_src = NSString::from_str(SHADER_SRC);
    let library = device
        .newLibraryWithSource_options_error(&source_src, None)
        .expect("bfs.metal failed to compile");
    let function = library
        .newFunctionWithName(&NSString::from_str("bfs_iteration"))
        .expect("bfs_iteration kernel not found in library");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline creation failed");
    let queue = device.newCommandQueue().expect("no command queue");

    // ── 2. BUFFERS ──────────────────────────────────────────────────
    let row_offsets_buf = device
        .newBufferWithLength_options(
            (graph.out.row_offsets.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("row_offsets_buf alloc failed");
    let col_indices_buf = device
        .newBufferWithLength_options(
            (graph.out.col_indices.len() * size_of::<u32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("col_indices_buf alloc failed");
    let mut frontier_buf = device
        .newBufferWithLength_options(n as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("frontier_buf alloc failed");
    let mut next_frontier_buf = device
        .newBufferWithLength_options(n as NSUInteger, MTLResourceOptions::StorageModeShared)
        .expect("next_frontier_buf alloc failed");
    let level_buf = device
        .newBufferWithLength_options(
            (n * size_of::<i32>()) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("level_buf alloc failed");
    let claimed_count_buf = device
        .newBufferWithLength_options(
            size_of::<u32>() as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("claimed_count_buf alloc failed");

    unsafe {
        let ptr = row_offsets_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, graph.out.row_offsets.len())
            .copy_from_slice(&graph.out.row_offsets);

        let ptr = col_indices_buf.contents().as_ptr() as *mut u32;
        std::slice::from_raw_parts_mut(ptr, graph.out.col_indices.len())
            .copy_from_slice(&graph.out.col_indices);

        let ptr = level_buf.contents().as_ptr() as *mut i32;
        std::slice::from_raw_parts_mut(ptr, n).fill(-1);
        *ptr.add(source as usize) = 0;

        let ptr = frontier_buf.contents().as_ptr() as *mut u8;
        std::slice::from_raw_parts_mut(ptr, n).fill(0);
        *ptr.add(source as usize) = 1;
    }

    // ── 3. DISPATCH, one level per iteration until the frontier empties ──
    let num_vertices = n as u32;
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

    let mut next_level: i32 = 1;
    loop {
        unsafe {
            let ptr = next_frontier_buf.contents().as_ptr() as *mut u8;
            std::slice::from_raw_parts_mut(ptr, n).fill(0);
            let ptr = claimed_count_buf.contents().as_ptr() as *mut u32;
            *ptr = 0;
        }

        let next_level_ptr: NonNull<c_void> = NonNull::from(&next_level).cast();

        let cmd_buffer = queue.commandBuffer().expect("no command buffer");
        let encoder = cmd_buffer.computeCommandEncoder().expect("no encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&row_offsets_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&col_indices_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&frontier_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&next_frontier_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&level_buf), 0, 4);
            encoder.setBytes_length_atIndex(next_level_ptr, size_of::<i32>() as NSUInteger, 5);
            encoder.setBytes_length_atIndex(num_vertices_ptr, size_of::<u32>() as NSUInteger, 6);
            encoder.setBuffer_offset_atIndex(Some(&claimed_count_buf), 0, 7);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        encoder.endEncoding();
        cmd_buffer.commit();
        cmd_buffer.waitUntilCompleted();

        let claimed = unsafe { *(claimed_count_buf.contents().as_ptr() as *const u32) };
        if claimed == 0 {
            break;
        }

        std::mem::swap(&mut frontier_buf, &mut next_frontier_buf);
        next_level += 1;
    }

    // ── 4. READ BACK ────────────────────────────────────────────────
    unsafe {
        let ptr = level_buf.contents().as_ptr() as *const i32;
        std::slice::from_raw_parts(ptr, n).to_vec()
    }
}
