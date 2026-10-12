import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RELATIONSHIP_TYPE = "CONNECTED_TO"


def convert(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_header = output_dir / "nodes_header.csv"
    nodes_path = output_dir / "nodes.csv"
    rels_header = output_dir / "rels_header.csv"
    rels_path = output_dir / "rels.csv"

    nodes_header.write_text("id:ID\n")
    rels_header.write_text(":START_ID,:END_ID,:TYPE\n")

    seen_nodes = set()
    edge_count = 0

    with input_path.open() as src, rels_path.open("w") as rels_out:
        for line in src:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            from_id, to_id = fields[0], fields[1]
            seen_nodes.add(from_id)
            seen_nodes.add(to_id)
            rels_out.write(f"{from_id},{to_id},{RELATIONSHIP_TYPE}\n")
            edge_count += 1

    with nodes_path.open("w") as nodes_out:
        for node_id in sorted(seen_nodes, key=int):
            nodes_out.write(f"{node_id}\n")

    print(f"{input_path.name}: {len(seen_nodes)} nodes, {edge_count} edges -> {output_dir}")


if __name__ == "__main__":
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR
    output_base = Path(sys.argv[2]) if len(sys.argv) > 2 else SCRIPT_DIR

    for txt_file in sorted(input_dir.glob("*.txt")):
        convert(txt_file, output_base / f"{txt_file.stem}-csv")
