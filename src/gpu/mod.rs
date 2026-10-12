pub mod bfs;
pub mod pagerank;
pub mod triangle_count;

// Force CoreGraphics link
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}
