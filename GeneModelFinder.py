import argparse
import glob
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache

DEFAULT_SKETCHES_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/reference.msh"))

REQUIRED_FILE_PATTERNS = [
    "*exon_probs.pbl",
    "*igenic_probs.pbl",
    "*intron_probs.pbl",
    "*metapars.cfg",
    "*parameters.cfg",
    "*weightmatrix.txt",
    ]

@dataclass(frozen=True)
class Sketch:
    hashes: int
    length: int
    sketch_id: str
    comment: str

@dataclass(frozen=True)
class MashHit:
    ref_id: str
    query_id: str
    mash_dist: float
    p_value: float
    matching_hashes: int
    total_hashes: int

@cache
def _require_mash() -> None:
    if shutil.which("mash") is None:
        raise EnvironmentError(
            "mash was not found in PATH. Please install mash and ensure it's in the system PATH."
        )

def _check_required_files(model_path: str, required_patterns: list[str]) -> bool:
    if not os.path.isdir(model_path):
        return False
    for local_pattern in required_patterns:
        full_pattern = os.path.join(model_path, local_pattern)
        if not glob.glob(full_pattern):
            return False
    return True

def check_models(models_path: str, sketches_path: str, show_all_missing: bool = False) -> None:
    sketches = run_mash_info(sketches_path)
    logging.info("Found %d reference sketches.", len(sketches))

    # Check that there's a gene model for each reference sketch
    missing_ids = []
    for sketch in sketches:
        model_path = os.path.join(models_path, sketch.sketch_id)
        if not _check_required_files(model_path, REQUIRED_FILE_PATTERNS):
            missing_ids.append(sketch.sketch_id)

    if missing_ids:
        message = f"Parameter files missing for {len(missing_ids)} models in {models_path}.\n"
        if len(missing_ids) <= 10 or show_all_missing:
            message += f"Model IDs with missing files: {'\n'.join(missing_ids)}"
        else:
            message += (
                f"Model IDs with missing files (showing first 10): "
                f"{'\n'.join(missing_ids[:10])}"
            )
        raise FileNotFoundError(message)

def run_mash_info(sketches_path: str) -> list[Sketch]:
    _require_mash()
    command = ["mash", "info", "-t", sketches_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Mash info returned non-zero exit status {e.returncode}.\n"
            f"Captured stderr: {e.stderr}"
        ) from e
    return parse_mash_info(result.stdout)

def parse_mash_info(result: str) -> list[Sketch]:
    if not result:
        return []
    sketches = []
    try:
        for row in result.rstrip().split("\n")[1:]:
            hashes, length, sketch_id, comment = row.split("\t")
            sketch = Sketch(int(hashes), int(length), sketch_id, comment)
            sketches.append(sketch)
    except ValueError as e:
        raise ValueError("Mash info output not formatted as expected (tabular format)") from e
    return sketches

def run_mash_dist(
    query: str, reference: str, opts: list[str] | None = None
) -> list[MashHit]:
    _require_mash()
    command = ["mash", "dist"]
    if opts is not None:
        command.extend(opts)
    command.extend([reference, query])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Mash dist returned non-zero exit status {e.returncode}.\n"
            f"Captured stderr: {e.stderr}"
        ) from e

    return parse_mash_dist(result.stdout)

def parse_mash_dist(result: str) -> list[MashHit]:
    if not result:
        return []
    hits = []
    try:
        for row in result.rstrip().split("\n"):
            ref_id, query_id, mash_dist, p_value, hash_ratio = row.split("\t")
            matching_hashes, total_hashes = hash_ratio.split("/")
            hit = MashHit(
                ref_id,
                query_id,
                float(mash_dist),
                float(p_value),
                int(matching_hashes),
                int(total_hashes),
            )
            hits.append(hit)
    except ValueError as e:
        raise ValueError(
            "Mash dist output not formatted as expected (standard output format)"
        ) from e
    return hits

def write_hits(best_hits: list[MashHit], simple_output: bool) -> None:
    if not best_hits:
        logging.info("No hits found.")
        return
    logging.info("Found %d hit(s).", len(best_hits))
    if simple_output:
        for hit in best_hits:
            print(hit.ref_id)
    else:
        header = ["Hit", "Mash_distance", "Matching hashes", "p value"]
        print("\t".join(header))
        for hit in best_hits:
            hash_ratio = f"{hit.matching_hashes}/{hit.total_hashes}"
            hit_info = [hit.ref_id, str(hit.mash_dist), hash_ratio, str(hit.p_value)]
            print("\t".join(hit_info))

def main(
    query: str, n: int, max_dist: float, simple_output: bool,
    sketches_path: str, check_model_path: str,
) -> None:
    if check_model_path:
        check_models(check_model_path, sketches_path)
    opts = ["-d", str(max_dist)]
    hits = run_mash_dist(query, sketches_path, opts)
    sorted_hits = sorted(hits, key=lambda hit: hit.mash_dist)
    write_hits(sorted_hits[:n], simple_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GeneModelFinder")
    parser.add_argument("query", default=None, type=str,
                        help="Path to a sequence file in .fasta (.fna) format")
    parser.add_argument("-n", default=1, type=int,
                        help="The number of hits to output (default: %(default)s)")
    parser.add_argument("-d", "--max_dist", default=0.3, type=float,
                        help="The maximum mash distance to report (default: %(default)s)")
    parser.add_argument("-s", "--simple-output", action="store_true",
                        help="Output only the reference IDs of the best hits, one per line")
    parser.add_argument("--sketches-path", default=DEFAULT_SKETCHES_PATH, type=str,
                        help="Path to the mash sketch file of the references "
                             "(default: %(default)s)")
    parser.add_argument("--check-model-path", default=None, type=str,
                        help="Check that there is a gene model for each reference, "
                             "based on the model path given. "
                             "If any models are missing, an error is raised.")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel",
                        const=logging.INFO, help="Enable verbose mode")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format="%(levelname)-8s %(asctime)s %(message)s")
    args_to_pass = {k: v for k, v in vars(args).items() if k != 'loglevel'}
    main(**args_to_pass)
