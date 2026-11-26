"""Explain modeled fusion edges on a validated graph, without optimizing it."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from .estimator import _fusion_blocking_reasons, _operand_use_counts
from .models import Graph
from .parser import load_graph


def analyze_fusion(graph: Graph) -> dict[str, Any]:
    """Explain actual edges in producer order, then consumer order.

    All operations in the original graph count, including dead operations.
    The report models logical traffic only and does not change the graph.
    """
    use_counts = _operand_use_counts(graph)
    outputs = set(graph.outputs)
    consumers_by_value: dict[str, list[int]] = defaultdict(list)
    for index, consumer in enumerate(graph.operations):
        for operand in dict.fromkeys(consumer.inputs):
            consumers_by_value[operand].append(index)

    edges: list[dict[str, Any]] = []
    total_bytes_avoided = 0
    for index, producer in enumerate(graph.operations):
        intermediate = producer.output
        for consumer_index in consumers_by_value.get(intermediate, []):
            consumer = graph.operations[consumer_index]
            reasons = _fusion_blocking_reasons(
                producer,
                consumer,
                use_count=use_counts[intermediate],
                graph_output=intermediate in outputs,
                adjacent=consumer_index == index + 1,
            )
            bytes_avoided = 0 if reasons else 2 * graph.values[intermediate].nbytes
            total_bytes_avoided += bytes_avoided
            edges.append(
                {
                    "producer": producer.name,
                    "consumer": consumer.name,
                    "intermediate": intermediate,
                    "use_count": use_counts[intermediate],
                    "blocking_reasons": reasons,
                    "eligible": not reasons,
                    "bytes_avoided": bytes_avoided,
                }
            )
    return {
        "graph": graph.name,
        "edges": edges,
        "total_bytes_avoided": total_bytes_avoided,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain original-graph fusion hints.")
    parser.add_argument("graph", metavar="GRAPH.json", help="path to a tensor graph JSON file")
    args = parser.parse_args(argv)
    try:
        graph = load_graph(args.graph)
        report_json = json.dumps(analyze_fusion(graph), indent=2, sort_keys=True)
    except (ValueError, RecursionError) as exc:
        # Includes GraphValidationError and Python's JSON integer/nesting limits.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
