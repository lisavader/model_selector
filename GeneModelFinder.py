from dataclasses import dataclass
import argparse
import logging
import os
import subprocess

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

def check_models(models_path: str, sketches_path: str):
    models = os.listdir(models_path)
    sketches = run_mash_info(sketches_path)
    logging.info("Found %d reference sketches.", len(sketches))
    # Check that there's a gene model for each reference sketch
    missing_ids = []
    for sketch in sketches:
        if not sketch.sketch_id in models:
            missing_ids.append(sketch.sketch_id)
    if missing_ids:
        missing_ids_formatted = "\n".join(missing_ids[:10])
        message = (f"For the following references, no gene model was found in {models_path} "
                   f"(showing the first 10): \n{missing_ids_formatted}")
        raise FileNotFoundError(message)

def run_mash_info(sketches_path: str):
    command = ["mash", "info", "-t", sketches_path]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("\n".join([f"Mash info returned non-zero exit status {e.returncode}.",
                                    f"Captured stderr: {e.stderr}"])) from e
    return parse_mash_info(result.stdout)

def parse_mash_info(result: str):
    if not result:
        return []
    sketches = []
    try:
        for row in result.rstrip().split("\n")[1:]:
            hashes, length, sketch_id, comment = row.split("\t")
            sketch = Sketch(hashes, length, sketch_id, comment)
            sketches.append(sketch)
    except ValueError as e:
        raise ValueError("Mash info output not formatted as expected (tabular format)") from e
    return sketches

def run_mash_dist(query: str, reference: str,
                  opts: list[str] = None):
    command = ["mash", "dist"]
    if opts is not None:
        command.extend(opts)
    command.extend([reference, query])
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("\n".join([f"Mash dist returned non-zero exit status {e.returncode}.",
                                    f"Captured stderr: {e.stderr}"])) from e

    return parse_mash_dist(result.stdout)

def parse_mash_dist(result: str):
    if not result:
        return []
    hits = []
    try:
        for row in result.rstrip().split("\n"):
            ref_id, query_id, mash_dist, p_value, hash_ratio = row.split("\t")
            matching_hashes, total_hashes = hash_ratio.split("/")
            hit = MashHit(ref_id, query_id, mash_dist, p_value, matching_hashes, total_hashes)
            hits.append(hit)
    except ValueError as e:
        raise ValueError("Mash dist output not formatted as expected (standard output format)") from e
    return hits

def get_best_hits(hits: list[MashHit], mode: str, n: int):
    if not hits:
        return []
    # Sort by lowest mash distance
    sorted_hits = sorted(hits, key=lambda hit: hit.mash_dist)
    if mode == "standard":
        best_hits = [sorted_hits[0]]
        for hit in sorted_hits[1:]:
            # Only add a hit if it has the same score as the best hit
            if hit.mash_dist != best_hits[-1].mash_dist:
                break
            best_hits.append(hit)
        return best_hits
    if mode == "best_n":
        return sorted_hits[:n]

def write_hits(best_hits: list[MashHit], models_path: str):
    if not best_hits:
        logging.info("No hits found.")
        return
    logging.info("Found %d hit(s).", len(best_hits))
    header = ["Hit", "Mash_distance", "Matching hashes", "p value"]
    print("\t".join(header))
    for hit in best_hits:
        hash_ratio = "/".join([hit.matching_hashes, hit.total_hashes])
        hit_info = [hit.ref_id, hit.mash_dist, hash_ratio, hit.p_value]
        print("\t".join(hit_info))

def select_models(query: str, mode: str, n: int, max_dist: float,
                  models_path: str, sketches_path: str, check: bool, loglevel: int):
    if check:
        check_models(models_path, sketches_path)
    opts = ["-d", str(max_dist)]
    hits = run_mash_dist(query, sketches_path, opts)
    best_hits = get_best_hits(hits, mode, n)
    write_hits(best_hits, models_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="GeneModelFinder")
    parser.add_argument("query", default=None, type=str, help=
                        "Path to a sequence file in .fasta (.fna) format")
    parser.add_argument("-m", "--mode", default="standard", type=str, help=
                        "Which mode to run. " \
                        "'standard': Outputs only the model(s) with the best score (= lowest mash distance)," \
                        "'best_n': Outputs the best n models (default: %(default)s)")
    parser.add_argument("-n", default=1, type=int, help=
                        "The number of models to output when running in 'best_n' mode (default: %(default)s)")
    parser.add_argument("-d", "--max_dist", default=0.3, type=float, help=
                        "The maximum mash distance to report (default: %(default)s)")
    parser.add_argument("--models_path", default="data/models", type=str, help=
                        "Path to the gene models directory (default: %(default)s)")
    parser.add_argument("--sketches_path", default="data/reference.msh", type=str, help=
                        "Path to the mash sketch file of the references (default: %(default)s)")
    parser.add_argument("-c", "--check", action="store_true", help=
                        "Check that there is a gene model for each reference")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.INFO, help=
                        "Enable verbose mode")
    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel, format="%(levelname)-8s %(asctime)s %(message)s")
    select_models(**vars(args))
