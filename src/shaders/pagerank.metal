#include <metal_stdlib>
using namespace metal;

// One thread per vertex, double-buffered ranks, no atomics needed.
kernel void pagerank_iteration(
    device const uint*  in_row_offsets   [[buffer(0)]],
    device const uint*  in_col_indices   [[buffer(1)]],
    device const uint*  out_degree       [[buffer(2)]],
    device const float* rank_in          [[buffer(3)]],
    device float*       rank_out         [[buffer(4)]],
    constant float&     damping          [[buffer(5)]],
    constant float&     base             [[buffer(6)]],
    constant float&     dangling_contrib [[buffer(7)]],
    constant uint&      num_vertices     [[buffer(8)]],
    uint v [[thread_position_in_grid]])
{
    if (v >= num_vertices) {
        return;
    }

    uint start = in_row_offsets[v];
    uint end = in_row_offsets[v + 1];

    float sum = 0.0;
    for (uint i = start; i < end; i++) {
        uint u = in_col_indices[i];
        uint deg = out_degree[u];
        if (deg > 0) {
            sum += rank_in[u] / float(deg);
        }
    }

    rank_out[v] = base + dangling_contrib + damping * sum;
}
