#include <metal_stdlib>
using namespace metal;

// Binary-searches `target` in the sorted range col_indices[lo..hi).
inline bool binary_search(device const uint* col_indices, uint lo, uint hi, uint target) {
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        uint val = col_indices[mid];
        if (val == target) {
            return true;
        } else if (val < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

// One thread per forward edge; binary search when list sizes are skewed,
// linear merge otherwise.
kernel void triangle_count(
    device const uint*  row_offsets    [[buffer(0)]],
    device const uint*  col_indices    [[buffer(1)]],
    device const uint*  edge_src       [[buffer(2)]],
    constant uint&      num_edges      [[buffer(3)]],
    device atomic_uint* triangle_total [[buffer(4)]],
    uint e [[thread_position_in_grid]])
{
    if (e >= num_edges) {
        return;
    }

    uint v = edge_src[e];
    uint u = col_indices[e];

    uint v_start = row_offsets[v];
    uint v_end = row_offsets[v + 1];
    uint u_start = row_offsets[u];
    uint u_end = row_offsets[u + 1];
    uint v_len = v_end - v_start;
    uint u_len = u_end - u_start;

    const uint SKEW_THRESHOLD = 32;
    uint count = 0;

    if (u_len > v_len * SKEW_THRESHOLD) {
        // v small, u large: binary search v into u.
        for (uint i = v_start; i < v_end; i++) {
            if (binary_search(col_indices, u_start, u_end, col_indices[i])) {
                count++;
            }
        }
    } else if (v_len > u_len * SKEW_THRESHOLD) {
        // u small, v large: binary search u into v.
        for (uint j = u_start; j < u_end; j++) {
            if (binary_search(col_indices, v_start, v_end, col_indices[j])) {
                count++;
            }
        }
    } else {
        // Comparable sizes: linear merge is cheapest.
        uint i = v_start;
        uint j = u_start;
        while (i < v_end && j < u_end) {
            uint a = col_indices[i];
            uint b = col_indices[j];
            if (a < b) {
                i++;
            } else if (a > b) {
                j++;
            } else {
                count++;
                i++;
                j++;
            }
        }
    }

    if (count > 0) {
        atomic_fetch_add_explicit(triangle_total, count, memory_order_relaxed);
    }
}
