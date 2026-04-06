import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def canonicalize_smiles(smiles: str) -> str:
    if not smiles:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    return canonicalize_smiles(smiles) is not None


def pair_similarity(a: str, b: str) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    amol = Chem.MolFromSmiles(a) if isinstance(a, str) else a
    bmol = Chem.MolFromSmiles(b) if isinstance(b, str) else b
    if amol is None or bmol is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_fp_diversity(smiles_list: Sequence[str]) -> float:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    if len(mols) < 2:
        return 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in mols]
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    if not sims:
        return 0.0
    return 1 - (sum(sims) / len(sims))


def compute_scaffold_diversity(smiles_list: Sequence[str]) -> float:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            scaffolds.append(scaffold)
    if not scaffolds:
        return 0.0
    return len(set(scaffolds)) / len(scaffolds)


def _predict_tdc_batch(smiles_batch: Sequence[str], oracle_name: str) -> List[float]:
    from tdc import Oracle

    oracle = Oracle(name=oracle_name)
    return list(oracle(list(smiles_batch)))


def predict_tdc_parallel(
    smiles_list: Sequence[str],
    oracle_name: str,
    *,
    num_workers: int,
    batch_size: int,
) -> Dict[str, float]:
    smiles_list = list(smiles_list)
    if not smiles_list:
        return {}
    if num_workers <= 1 or len(smiles_list) <= batch_size:
        scores = _predict_tdc_batch(smiles_list, oracle_name)
        return {smiles_list[i]: float(scores[i]) for i in range(len(smiles_list))}

    batches = [smiles_list[i : i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    results = [0.0] * len(smiles_list)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_predict_tdc_batch, batch, oracle_name): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            batch_start = batch_index * batch_size
            scores = future.result()
            for offset, score in enumerate(scores):
                results[batch_start + offset] = float(score)
    return {smiles_list[i]: results[i] for i in range(len(smiles_list))}


def predict_local_properties_for_smiles(
    smiles_list: Iterable[str],
    properties: Sequence[str],
    *,
    cache_path: str = None,
) -> pd.DataFrame:
    props_dir = Path(__file__).resolve().parent / "data" / "props"
    if str(props_dir) not in sys.path:
        sys.path.insert(0, str(props_dir))
    from properties import drd2, penalized_logp, qed, sas

    property_list = [prop for prop in dict.fromkeys(properties) if prop not in {"sas", "jnk3", "gsk3b"}]
    smiles_list = [smiles for smiles in dict.fromkeys(smiles_list) if smiles]
    if not smiles_list:
        columns = ["smiles"] + property_list + ["sas"]
        return pd.DataFrame(columns=columns)

    cache = {}
    if cache_path:
        try:
            cache_df = pd.read_csv(cache_path)
            cache_df = cache_df.fillna("")
            for record in cache_df.to_dict(orient="records"):
                cache[record["smiles"]] = record
        except FileNotFoundError:
            pass

    rows = []
    missing = []
    for smiles in smiles_list:
        cached = cache.get(smiles)
        if cached and all(prop in cached and cached[prop] != "" for prop in property_list):
            rows.append(cached)
        else:
            missing.append(smiles)

    predicted_rows = []
    if missing:
        for smiles in missing:
            row = {"smiles": smiles}
            for prop in property_list:
                if prop == "drd2":
                    row[prop] = float(drd2(smiles))
                elif prop == "qed":
                    row[prop] = float(qed(smiles))
                elif prop == "plogp":
                    row[prop] = float(penalized_logp(smiles))
                else:
                    raise ValueError(f"Unsupported local property predictor: {prop}")
            row["sas"] = float(sas(smiles))
            predicted_rows.append(row)

    all_rows = rows + predicted_rows
    df = pd.DataFrame(all_rows)
    if cache_path and not df.empty:
        merged = df
        if cache:
            previous_df = pd.DataFrame(cache.values())
            merged = pd.concat([previous_df, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["smiles"], keep="last")
        merged.to_csv(cache_path, index=False)
    return df


def predict_tdc_properties_for_smiles(
    smiles_list: Iterable[str],
    properties: Sequence[str],
    *,
    num_workers: int,
    batch_size: int,
    cache_path: str = None,
) -> pd.DataFrame:
    property_list = [prop for prop in dict.fromkeys(properties) if prop in {"jnk3", "gsk3b"}]
    smiles_list = [smiles for smiles in dict.fromkeys(smiles_list) if smiles]
    if not property_list:
        return pd.DataFrame(columns=["smiles"])
    if not smiles_list:
        return pd.DataFrame(columns=["smiles"] + property_list)

    cache = {}
    if cache_path:
        try:
            cache_df = pd.read_csv(cache_path)
            cache_df = cache_df.fillna("")
            for record in cache_df.to_dict(orient="records"):
                cache[record["smiles"]] = record
        except FileNotFoundError:
            pass

    rows = []
    missing = []
    for smiles in smiles_list:
        cached = cache.get(smiles)
        if cached and all(prop in cached and cached[prop] != "" for prop in property_list):
            rows.append(cached)
        else:
            missing.append(smiles)

    predicted_rows = []
    if missing:
        tdc_targets = {}
        for prop in property_list:
            if prop == "jnk3":
                tdc_targets[prop] = predict_tdc_parallel(missing, "JNK3", num_workers=num_workers, batch_size=batch_size)
            elif prop == "gsk3b":
                tdc_targets[prop] = predict_tdc_parallel(missing, "GSK3B", num_workers=num_workers, batch_size=batch_size)
        for smiles in missing:
            row = {"smiles": smiles}
            for prop in property_list:
                row[prop] = float(tdc_targets[prop][smiles])
            predicted_rows.append(row)

    all_rows = rows + predicted_rows
    df = pd.DataFrame(all_rows)
    if cache_path and not df.empty:
        merged = df
        if cache:
            previous_df = pd.DataFrame(cache.values())
            merged = pd.concat([previous_df, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["smiles"], keep="last")
        merged.to_csv(cache_path, index=False)
    return df


def load_property_cache(cache_path: str) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        return pd.DataFrame(columns=["smiles"])
    return pd.read_csv(path)


def merge_property_caches(cache_paths: Sequence[str]) -> pd.DataFrame:
    merged = None
    for cache_path in cache_paths:
        if not cache_path:
            continue
        df = load_property_cache(cache_path)
        if df.empty:
            continue
        merged = df if merged is None else merged.merge(df, on="smiles", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["smiles"])
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def normalized_improvement(delta: float, threshold: float) -> float:
    if threshold <= 0:
        return delta
    return delta / threshold


def composite_score(improvements: Sequence[float]) -> float:
    if not improvements:
        return 0.0
    return sum(improvements) / len(improvements)


def pareto_filter(points: List[List[float]]) -> List[List[float]]:
    filtered = []
    for i, point in enumerate(points):
        dominated = False
        for j, other in enumerate(points):
            if i == j:
                continue
            if all(o >= p for o, p in zip(other, point)) and any(o > p for o, p in zip(other, point)):
                dominated = True
                break
        if not dominated:
            filtered.append(point)
    return filtered


def hypervolume(points: List[List[float]]) -> float:
    points = [point for point in points if point and all(value >= 0 for value in point)]
    if not points:
        return 0.0
    points = pareto_filter(points)
    dims = len(points[0])

    def hv_recursive(current_points: List[List[float]], dimension: int) -> float:
        if not current_points:
            return 0.0
        if dimension == 1:
            return max(point[0] for point in current_points)

        sorted_points = sorted(current_points, key=lambda point: point[dimension - 1])
        volume = 0.0
        previous = 0.0
        for index, point in enumerate(sorted_points):
            height = point[dimension - 1] - previous
            if height > 0:
                projection = [candidate[: dimension - 1] for candidate in sorted_points[index:]]
                volume += hv_recursive(projection, dimension - 1) * height
                previous = point[dimension - 1]
        return volume

    return hv_recursive(points, dims)


def safe_mean(values: Sequence[float]) -> float:
    values = [value for value in values if value is not None and not math.isnan(value)]
    if not values:
        return float("nan")
    return sum(values) / len(values)
