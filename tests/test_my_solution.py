import tempfile
from pathlib import Path

import pytest

from my_solution import (bucket_index, 
                         bucket_input_file, 
                         stream_routes, 
                         build_graph, 
                         longest_cycle_length,
                         process_bucket
                        )


# bucket_index should be deterministic for the same key
def test_bucket_index_is_deterministic():
    idx1 = bucket_index("123", "200", 256)
    idx2 = bucket_index("123", "200", 256)
    assert idx1 == idx2


# bucket_index should always return a valid bucket number
def test_bucket_index_is_within_range():
    idx = bucket_index("123", "200", 256)
    assert 0 <= idx < 256


# stream_routes should parse valid records correctly
def test_stream_routes_parses_valid_file():
    content = "\n".join(
        [
            "Epic|Availity|1|100",
            "Availity|Optum|1|100",
        ]
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    records = list(stream_routes(tmp_path))

    assert records == [
        ("Epic", "Availity", "1", "100"),
        ("Availity", "Optum", "1", "100"),
    ]


# stream_routes should skip blank lines
def test_stream_routes_skips_blank_lines():
    content = "\n".join(
        [
            "Epic|Availity|1|100",
            "",
            "Availity|Optum|1|100",
        ]
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    records = list(stream_routes(tmp_path))

    assert records == [
        ("Epic", "Availity", "1", "100"),
        ("Availity", "Optum", "1", "100"),
    ]


# stream_routes should raise on malformed input
def test_stream_routes_raises_on_invalid_record():
    content = "Epic|Availity|1\n"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    with pytest.raises(ValueError):
        list(stream_routes(tmp_path))


# bucket_input_file should send matching claim/status rows to the same bucket
def test_bucket_input_file_groups_same_claim_and_status_together():
    content = "\n".join(
        [
            "Epic|Availity|1|100",
            "Availity|Optum|1|100",
            "Cerner|Epic|2|200",
        ]
    )

    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    bucket_paths = bucket_input_file(tmp_path, num_buckets=8)

    bucket_contents = {}
    for path in bucket_paths:
        if not Path(path).exists():
            continue

        text = Path(path).read_text(encoding="utf-8")
        if text:
            bucket_contents[path] = text.strip().splitlines()

    matching_rows = [
        "Epic|Availity|1|100",
        "Availity|Optum|1|100",
    ]

    found_bucket = None
    for lines in bucket_contents.values():
        if all(row in lines for row in matching_rows):
            found_bucket = lines
            break

    assert found_bucket is not None


# longest_cycle_length should return 0 when no cycle exists
def test_longest_cycle_length_returns_zero_for_acyclic_graph():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
    ]
    graph = build_graph(edges)

    assert longest_cycle_length(graph) == 0


# longest_cycle_length should detect a simple 3-node cycle
def test_longest_cycle_length_detects_three_node_cycle():
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
    ]
    graph = build_graph(edges)

    assert longest_cycle_length(graph) == 3


# longest_cycle_length should return the largest cycle in a graph
def test_longest_cycle_length_returns_largest_cycle():
    edges = [
        ("A", "B"),
        ("B", "A"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "C"),
    ]
    graph = build_graph(edges)

    assert longest_cycle_length(graph) == 4


# longest_cycle_length should count a self-loop as a cycle of length 1
def test_longest_cycle_length_counts_self_loop():
    edges = [
        ("A", "A"),
    ]
    graph = build_graph(edges)

    assert longest_cycle_length(graph) == 1

# process_bucket should return the best cycle in a bucket file
def test_process_bucket_returns_longest_cycle(tmp_path: Path):
    bucket_file = tmp_path / "bucket_0.txt"
    bucket_file.write_text(
        "\n".join(
            [
                "A|B|1|100",
                "B|C|1|100",
                "C|A|1|100",
                "X|Y|2|200",
                "Y|Z|2|200",
                "Z|W|2|200",
            ]
        ),
        encoding="utf-8",
    )

    claim_id, status_code, cycle_length = process_bucket(bucket_file)

    assert claim_id == "1"
    assert status_code == "100"
    assert cycle_length == 3

# end-to-end: sample input should return the longest cycle across all groups
def test_small_input_sample_returns_expected_longest_cycle(tmp_path):
    input_file = tmp_path / "small_input_v1.txt"
    input_file.write_text(
        "\n".join(
            [
                "System950049|System950045|190011|190110",
                "System950040|System950038|190011|190110",
                "System950047|System950048|190011|190110",
                "System950046|System950047|190011|190110",
                "System950044|System950041|190011|190110",
                "System950041|System950042|190011|190110",
                "System950039|System950040|190011|190110",
                "System950048|System950049|190011|190110",
                "System950038|System950039|190011|190110",
                "System950045|System950046|190011|190110",
                "System950042|System950043|190011|190110",
                "System950043|System950044|190011|190110",
                "System950093|System950094|190017|190116",
                "System950092|System950093|190017|190116",
                "System950091|System950092|190017|190116",
                "System950088|System950089|190017|190116",
                "System950086|System950087|190017|190116",
                "System950087|System950088|190017|190116",
                "System950094|System950085|190017|190116",
                "System950089|System950090|190017|190116",
                "System950085|System950086|190017|190116",
                "System950090|System950091|190017|190116",
            ]
        ),
        encoding="utf-8",
    )

    bucket_paths = bucket_input_file(str(input_file), num_buckets=8)

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

    assert best_claim_id == "190017"
    assert best_status_code == "190116"
    assert best_cycle_length == 10