import json
import numpy as np
import pickle

from pathlib import Path
from typing import Union, List, Dict, Any
from zipfile import ZIP_DEFLATED, ZipFile

def make_action_recognition_submission(
    interaction_scores: np.ndarray,
    annotation_ids: Union[np.ndarray, List[str]],
    challenge: str = "audio_based_interaction_recognition",
    sls_pt: int = 5,
    sls_tl: int = 5,
    sls_td: int = 5,
    t_mod: int = 0,
    version = "0.1"
):
    """
    Args:
        interaction_scores: Array containing verb scores of shape :math:`(N, 44)`.
        annotation_ids:  Array or list of length :math:`N` containing annotation IDs
            for each score.
        challenge: The challenge being submitted to.
        sls_pt:  Supervision level: pretraining (0--5)
        sls_tl:  Supervision level: training labels (0--5)
        sls_td:  Supervision level: training data (0--5)
        t_mod:  Training modality (0--2)
    Returns:
        Submission dictionary ready to be serialised to JSON.
    """
    return {
        "version": version,
        "challenge": challenge,
        "sls_pt": sls_pt,
        "sls_tl": sls_tl,
        "sls_td": sls_td,
        "t_mod": t_mod,
        "results": make_action_recognition_submission_scores_dict(
            interaction_scores, annotation_ids
        )
    }

def make_action_recognition_submission_scores_dict(
    interaction_scores: np.ndarray,
    annotation_ids: Union[np.ndarray, List[str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Args:
        interaction_scores: Array containing verb scores of shape :math:`(N, 44)`.
        annotation_ids:  Array or list of length :math:`N` containing annotation IDs
            for each score.
    Returns:
        Dictionary mapping annotation ids to a dictionary containing interaction
        scores, e.g.
        .. code-block:: python
            "P01_101_0": {
              "interaction": {
                "0": 1.223,
                "1": 4.278,
                ...
                "44": 0.023
              }
            }
    """
    results = dict()
    for example_interaction_scores, annotation_id in zip(
        interaction_scores, annotation_ids
    ):
        results[str(annotation_id)] = {
            "interaction": make_scores_dict(example_interaction_scores)
        }
    return results

def make_scores_dict(scores: np.ndarray) -> Dict[str, float]:
    assert scores.ndim == 1
    return {str(i): float(s) for i, s in enumerate(scores)}

def write_submission_file(submission_dict: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=5) as f:
        f.writestr("test.json", json.dumps(submission_dict))