from typing import Dict, Tuple, Union, List

import numpy as np
import pandas as pd
import sklearn.metrics as met
from .scoring import scores_dict_to_ranks

def _check_label_predictions_preconditions(rankings: np.ndarray, labels: np.ndarray):
    if len(rankings) < 1:
        raise ValueError(
            f"Need at least one instance to evaluate, but input shape "
            f"was {rankings.shape}"
        )
    if not rankings.ndim == 2:
        raise ValueError(f"Rankings should be a 2D matrix but was {rankings.ndim}D")
    if not labels.ndim == 1:
        raise ValueError(f"Labels should be a 1D vector but was {labels.ndim}D")
    if not labels.shape[0] == rankings.shape[0]:
        raise ValueError(
            f"Number of labels ({labels.shape[0]}) provided does not match number of "
            f"predictions ({rankings.shape[0]})"
        )


def topk_accuracy(
    rankings: np.ndarray, labels: np.ndarray, ks: Union[Tuple[int, ...], int] = (1, 5)
) -> List[float]:
    """Computes TOP-K accuracies for different values of k
    Parameters:
    -----------
    rankings
        2D rankings array: shape = (instance_count, label_count)
    labels
        1D correct labels array: shape = (instance_count,)
    ks
        The k values in top-k, either an int or a list of ints.

    Returns:
    --------
    list of float: TOP-K accuracy for each k in ks

    Raises:
    -------
    ValueError
         If the dimensionality of the rankings or labels is incorrect, or
         if the length of rankings and labels aren't equal
    """
    if isinstance(ks, int):
        ks = (ks,)
    _check_label_predictions_preconditions(rankings, labels)

    # trim to max k to avoid extra computation
    maxk = np.max(ks)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)

    # trim to selected ks and compute accuracies
    accuracies = [tp[:, :k].max(1).mean() for k in ks]
    if any(np.isnan(accuracies)):
        raise ValueError(f"NaN present in accuracies {accuracies}")
    return accuracies


def map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes, or num_examples.
    Returns:
        mean_ap (int): final mAP score.
    """
    # Convert labels to one hot vector
    labels = np.eye(preds.shape[1])[labels] if labels.ndim == 1 else labels
    
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = met.average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap

def per_class_accuracy(preds, labels):
    preds_inds = preds.argmax(axis=1)
    matrix = met.confusion_matrix(labels, preds_inds)
    class_counts = matrix.sum(axis=1)
    correct = matrix.diagonal()[class_counts != 0]
    class_counts = class_counts[class_counts != 0]
    return correct / class_counts

def calculate_stats(output, target):
    """Calculate statistics including per class accuracy, mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Class-wise statistics
    for k in range(classes_num):
        # Average precision
        avg_precision = met.average_precision_score(
                target[:, k], output[:, k], average=None
            )

        # AUC
        auc = met.roc_auc_score(
                target[:, k], output[:, k], average=None
            )

        # Precisions, recalls
        (precisions, recalls, thresholds) = met.precision_recall_curve(
                target[:, k], output[:, k]
            )

        # FPR, TPR
        (fpr, tpr, thresholds) = met.roc_curve(
                target[:, k], output[:, k]
            )

        save_every_steps = 1000   # Sample statistics to reduce size
        stats_dict = {
                'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc
            }
        stats.append(stats_dict)

    return stats

def get_stats(preds, labels):
    per_class_acc = per_class_accuracy(preds, labels)

    # Convert labels to one hot vector
    labels = np.eye(preds.shape[1])[labels] if labels.ndim == 1 else labels

    # Only calculate for seen classes
    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]

    stats = calculate_stats(preds, labels)
    # Write out to log
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    m_PCA = np.mean([class_acc for class_acc in per_class_acc])

    stats_dict = {}
    stats_dict["mAP"] = mAP
    stats_dict["mAUC"] = mAUC
    stats_dict["mPCA"] = m_PCA

    return stats_dict

def compute_metrics(
    groundtruth_df: pd.DataFrame,
    scores: Dict[str, np.ndarray]
):
    """
    Parameters
    ----------
    groundtruth_df
        DataFrame containing 'class_id': int columns.
    scores
        Dictionary containing 'interaction' entries should map to a 2D
        np.ndarray of shape (instance_count, class_count) where each element is the predicted score
        of that class.

    Returns
    -------
    A dictionary containing nested metrics.

    Raises
    ------
    ValueError
        If the shapes of the score arrays are not correct, or the lengths of the groundtruth_df and the
        scores array are not equal, or if the grountruth_df doesn't have the specified columns.

    """
    np_labels = groundtruth_df.to_numpy().astype(int)
    preds = np.array(scores["interaction"], dtype=np.float32)
    metrics = get_stats(preds, np_labels)

    ranks = scores_dict_to_ranks(scores)
    top_k = (1, 5)

    all_accuracies = {
        "interaction": topk_accuracy(
            ranks["interaction"], np_labels, ks=top_k
        )
    }

    return {
        "accuracies": all_accuracies,
        "mAP": metrics["mAP"],
        "mAUC": metrics["mAUC"],
        "mCA": metrics["mPCA"]       
    }