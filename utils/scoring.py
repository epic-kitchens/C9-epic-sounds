from typing import Dict
from typing import List
from typing import Union

import numpy as np

def scores_dict_to_ranks(scores_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {key: scores_to_ranks(scores) for key, scores in scores_dict.items()}


def scores_to_ranks(scores: Union[np.ndarray, List[Dict[int, float]]]) -> np.ndarray:
    if isinstance(scores, np.ndarray):
        return _scores_array_to_ranks(scores)
    elif isinstance(scores, list):
        return _scores_dict_to_ranks(scores)
    raise ValueError("Cannot compute ranks for type {}".format(type(scores)))

def _scores_array_to_ranks(scores: np.ndarray):
    """
    The rank vector contains classes and is indexed by the rank

    Examples:
        >>> _scores_array_to_ranks(np.array([[0.1, 0.15, 0.25,  0.3, 0.5], \
                                             [0.5, 0.3, 0.25,  0.15, 0.1], \
                                             [0.2, 0.4,  0.1,  0.25, 0.05]]))
        array([[4, 3, 2, 1, 0],
               [0, 1, 2, 3, 4],
               [1, 3, 0, 2, 4]])
    """
    if scores.ndim != 2:
        raise ValueError(
            "Expected scores to be 2 dimensional: [n_instances, n_classes]"
        )
    return scores.argsort(axis=-1)[:, ::-1]

def _scores_dict_to_ranks(scores: List[Dict[int, float]]) -> np.ndarray:
    """
    Compute ranking from class to score dictionary

    Examples:
        >>> _scores_dict_to_ranks([{0: 0.15, 10: 0.75, 5: 0.1},\
                                   {0: 0.85, 10: 0.10, 5: 0.05}])
        array([[10,  0,  5],
               [ 0, 10,  5]])
    """
    ranks = []
    for score in scores:
        class_ids = np.array(list(score.keys()))
        score_array = np.array([score[class_id] for class_id in class_ids])
        ranks.append(class_ids[np.argsort(score_array)[::-1]])
    return np.array(ranks)