#include <metal_stdlib>
using namespace metal;

// One thread per vertex per level; claims unvisited out-neighbors via
// atomic CAS on level, bumping claimed_count so the host detects an empty frontier.
kernel void bfs_iteration(
    device const uint*   out_row_offsets [[buffer(0)]],
    device const uint*   out_col_indices [[buffer(1)]],
    device const uchar*  frontier        [[buffer(2)]],
    device uchar*        next_frontier   [[buffer(3)]],
    device atomic_int*   level           [[buffer(4)]],
    constant int&         next_level      [[buffer(5)]],
    constant uint&        num_vertices    [[buffer(6)]],
    device atomic_uint*  claimed_count   [[buffer(7)]],
    uint v [[thread_position_in_grid]])
{
    if (v >= num_vertices || frontier[v] == 0) {
        return;
    }

    uint start = out_row_offsets[v];
    uint end = out_row_offsets[v + 1];

    for (uint i = start; i < end; i++) {
        uint u = out_col_indices[i];
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                &level[u], &expected, next_level,
                memory_order_relaxed, memory_order_relaxed)) {
            next_frontier[u] = 1;
            atomic_fetch_add_explicit(claimed_count, 1u, memory_order_relaxed);
        }
    }
}
