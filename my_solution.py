import sys
import argparse
import time
import tracemalloc
import tempfile
import hashlib
from pathlib import Path
from typing import List, Iterator, Tuple, Dict, Set, Iterable, TextIO
from collections import defaultdict


def bucket_index(claim_id: str, status_code: str, num_buckets: int) -> int:
    """
    Compute a deterministic bucket index for a (claim_id, status_code) pair.

    Uses an MD5 hash of the composite key to ensure stable distribution
    across runs. The result is an integer in the range [0, num_buckets - 1].

    Args:
        claim_id (str): Claim identifier.
        status_code (str): Status code associated with the claim.
        num_buckets (int): Total number of buckets.

    Returns:
        int: Bucket index for this key.
    """
    key = f"{claim_id}|{status_code}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_buckets


def bucket_input_file(file_path: str, num_buckets: int = 64) -> List[Path]:
    """
    Stream the input file and partition records into bucket files based on
    (claim_id, status_code).

    Each record is written to one of `num_buckets` files using a hash-based
    bucket index. Bucket files are opened lazily, only when first needed, to
    reduce the number of simultaneously open file handles.

    Args:
        file_path (str): Path to the input routing file.
        num_buckets (int): Number of output bucket files.

    Returns:
        List[Path]: Paths to the generated bucket files.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="routing_buckets_"))
    bucket_paths = [temp_dir / f"bucket_{i}.txt" for i in range(num_buckets)]
    bucket_files: dict[int, TextIO] = {}

    try:
        for source, dest, claim_id, status_code in stream_routes(file_path):
            idx = bucket_index(claim_id, status_code, num_buckets)

            if idx not in bucket_files:
                bucket_files[idx] = bucket_paths[idx].open("w", encoding="utf-8")

            bucket_files[idx].write(f"{source}|{dest}|{claim_id}|{status_code}\n")
    finally:
        for f in bucket_files.values():
            f.close()

    return bucket_paths


def longest_cycle_length(graph: Dict[str, Set[str]]) -> int:
    """
    Compute the length of the longest simple directed cycle in a graph.

    A cycle is defined as a path that starts and ends at the same node,
    with no repeated nodes except the start/end node.

    Args:
        graph (Dict[str, Set[str]]):
            Adjacency list representing the directed graph.
            Keys are source nodes, values are sets of destination nodes.

    Returns:
        int: Length of the longest cycle found. Returns 0 if no cycle exists.
    """
    max_len: int = 0

    def dfs(start: str, current: str, visited: Set[str], depth: int) -> None:
        """
        Depth-first search to explore all simple paths starting from `start`.

        Args:
            start (str): Starting node for cycle detection.
            current (str): Current node in traversal.
            visited (Set[str]): Nodes visited in the current path.
            depth (int): Current path length (number of edges traversed so far).
        """
        nonlocal max_len

        for neighbor in graph.get(current, set()):
            # If we return to the start node, a cycle is found
            if neighbor == start:
                max_len = max(max_len, depth + 1)

            # Continue DFS if neighbor not already in current path
            elif neighbor not in visited:
                visited.add(neighbor)
                dfs(start, neighbor, visited, depth + 1)
                visited.remove(neighbor)

    # Start DFS from each node
    for node in graph:
        dfs(start=node, current=node, visited={node}, depth=0)

    return max_len


def build_graph(edges: Iterable[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """
    Build a directed adjacency list from (source, destination) edges.

    Ensures all nodes (including destination-only nodes) appear as keys.
    Deduplicates edges by using sets.

    Args:
        edges: Iterable of (source, destination) tuples.

    Returns:
        Dict[str, Set[str]]: Adjacency list mapping each node to its neighbors.
    """
    graph: Dict[str, Set[str]] = defaultdict(set)

    for source, destination in edges:
        graph[source].add(destination)
        if destination not in graph:
            graph[destination] = set()

    return graph


def process_bucket(bucket_path: Path) -> Tuple[str | None, str | None, int]:
    """
    Process a single bucket file and find the longest cycle within it.

    Records are grouped by (claim_id, status_code). For each group, a directed
    graph is built and analyzed for its longest simple cycle. The best result
    in the bucket is returned.

    Args:
        bucket_path (Path): Path to a bucket file.

    Returns:
        Tuple[str | None, str | None, int]:
            (claim_id, status_code, cycle_length) for the longest cycle found
            in the bucket. If no cycle is found, returns (None, None, 0).
    """
    grouped_edges: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    with bucket_path.open("r", encoding="utf-8") as f:
        for line in f:
            source, destination, claim_id, status_code = line.rstrip("\n").split("|")
            grouped_edges[(claim_id, status_code)].append((source, destination))

    best_claim_id: str | None = None
    best_status_code: str | None = None
    best_cycle_length: int = 0

    for (claim_id, status_code), edges in grouped_edges.items():
        graph = build_graph(edges)
        cycle_length = longest_cycle_length(graph)

        if cycle_length > best_cycle_length:
            best_claim_id = claim_id
            best_status_code = status_code
            best_cycle_length = cycle_length

    return best_claim_id, best_status_code, best_cycle_length


def stream_routes(file_path: str) -> Iterator[Tuple[str, str, str, str]]:
    """
    Stream routing records from a newline-delimited file.

    Each line is expected to follow the format:
    <source_system>|<destination_system>|<claim_id>|<status_code>

    The function yields one parsed record at a time, enabling efficient
    processing of large files without loading the entire dataset into memory.

    Args:
        file_path (str): Path to the input routing file.

    Yields:
        Iterator[Tuple[str, str, str, str]]:
            Tuples of (source_system, destination_system, claim_id, status_code).

    Raises:
        ValueError: If a line does not contain exactly four pipe-delimited fields.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("|")
            if len(parts) != 4:
                raise ValueError(f"Invalid record at line {line_number}")

            source_system, destination_system, claim_id, status_code = parts
            yield source_system, destination_system, claim_id, status_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the longest routing cycle in a claims routing file"
    )

    parser.add_argument(
        "input_file",
        help="Path to the input routing file (newline-delimited records)"
    )

    parser.add_argument(
        "--buckets",
        type=int,
        default=64,
        help="Number of hash buckets to partition data (default: 64)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (prints timing and memory usage to stderr)"
    )

    # trigger help if no args
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_time: float | None = None

    if args.debug:
        tracemalloc.start()
        start_time = time.perf_counter()

    bucket_paths = bucket_input_file(args.input_file)

    best_claim_id = None
    best_status_code = None
    best_cycle_length = 0

    for bucket_path in bucket_paths:
        if not bucket_path.exists():
            continue

        claim_id, status_code, cycle_length = process_bucket(bucket_path)

        if cycle_length > best_cycle_length:
            best_claim_id = claim_id
            best_status_code = status_code
            best_cycle_length = cycle_length

    if best_claim_id is None:
        print(",,0")
    else:
        print(f"{best_claim_id},{best_status_code},{best_cycle_length}")

    if args.debug and start_time is not None:
        elapsed_seconds = time.perf_counter() - start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 * 1024)

        print(f"[debug] elapsed_seconds={elapsed_seconds:.3f}", file=sys.stderr)
        print(f"[debug] peak_memory_mb={peak_mb:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()