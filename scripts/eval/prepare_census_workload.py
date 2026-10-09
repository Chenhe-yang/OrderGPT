"""Prepare the shared CENSUS query workload in canonical column coordinates."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aligned_tabular.data import AlignedTabularDataset
from aligned_tabular.query_evaluation import generate_varskip_queries, exact_query_cardinalities
from scripts.eval.common import write_json
from scripts.eval.eval_census_progressive_sampling import canonical_sha256, workload_dataset_fingerprint, full_records_current_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/census/records"))
    parser.add_argument("--output", type=Path, default=Path("data/census/workload.json"))
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; reuse that shared workload or choose a new output")
    dataset = AlignedTabularDataset(args.data_dir, variable_permutation_seed=-1)
    if dataset.num_variables < 11:
        parser.error("workload needs at least 11 columns for 5–11 predicates")
    records = full_records_current_ids(dataset)
    queries = generate_varskip_queries(records, dataset.cardinalities.numpy(), rng_seed=args.seed, num_queries=args.num_queries)
    counts = exact_query_cardinalities(records, queries)
    for query, count in zip(queries, counts):
        query["actual_cardinality"] = int(count)
    payload = {"protocol": "canonical_varskip_workload_v1", "query_seed": args.seed,
               "num_queries": len(queries), "dataset_fingerprint": workload_dataset_fingerprint(dataset), "queries": queries}
    payload["workload_sha256"] = canonical_sha256(payload)
    write_json(args.output, payload)
    print(f"Saved {len(queries)} queries with exact full-table counts to {args.output}")


if __name__ == "__main__":
    main()
