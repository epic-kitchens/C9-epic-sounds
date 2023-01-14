import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch


def load_results(results_path: Path) -> Dict[str, Any]:
    """
    Load results created by running `evaluate.py`
    Args:
        results_path: Path to a results file with either a .pt suffix (loadable via
            ``torch.load``) or with a .pkl suffix (loadable via
            ``pickle.load(open(path, 'rb'))``). The loaded data should be in one of the
            following formats:
                [
                    {
                        'interaction_output': np.ndarray of float32, shape [44],
                        'annotation_id': str, e.g. 'P01_101_1'
                    }, ... # repeated for all segments in the val/test set.
                ]
            or
                {
                    'interaction_output': np.ndarray of float32, shape [N, 44],
                    'annotation_id': np.ndarray of str, shape [N,]
                }
    Returns:
        A dictionary with the structure
        .. code-block::
            {
               'interaction_output': np.ndarray [N, 44],
               'annotation_id': np.ndarray [N,],
            }
    """
    # These are all in python lists, we turn them into np arrays for convenience
    # We first have to collate them.
    results: Union[List[Dict[str, Any]], Dict[str, Any]]
    if results_path.suffix.lower() == ".pkl":
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    elif results_path.suffix.lower() == ".pt":
        results = torch.load(results_path, map_location=torch.device("cpu"))
    else:
        raise ValueError(
            f"Unknown file extensions {results_path.suffix!r} in path "
            f"{results_path}"
        )

    if isinstance(results, list):
        new_results = dict()
        first_item = results[0]
        for key in first_item.keys():
            new_results[key] = np.array([r[key] for r in results])
        return new_results
    return results