import argparse
from pathlib import Path
from textwrap import dedent

import pandas as pd
import numpy as np
import torch
from pprint import pprint

from utils.metrics import compute_metrics
from utils.results import load_results

parser = argparse.ArgumentParser(
    description="Evaluate model results on the validation set",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "results",
    type=Path,
    help=dedent(
        """\
        Path to a results file with either a .pt suffix (loadable via `torch.load`) or 
        with a .pkl suffix (loadable via `pickle.load(open(path, 'rb'))`). The 
        loaded data should be in one of the following formats:
            [
                {
                    'interaction_output': np.ndarray of float32, shape [44],
                    'annotation_id': str, e.g. 'P01_101_1'
                }, ... # repeated entries 
            ]
        or 
            {
                'interaction_output': np.ndarray of float32, shape [N, 44],
                'annotation_id': np.ndarray of str, shape [N,]
            }
        """
    ),
)
parser.add_argument("labels", type=Path, help="Labels (pickled dataframe)")



def collate(results):
    return {k: [r[k] for r in results] for k in results[0].keys()}


def main(args):
    labels: pd.DataFrame = pd.read_pickle(args.labels)
    if "annotation_id" in labels.columns:
        labels.set_index("annotation_id", inplace=True)
    labels = labels["class_id"]

    results = load_results(args.results)
    results["interaction_output"] = np.sum(results["interaction_output"], axis=1)
    interaction_output = results["interaction_output"]
    annotation_ids = results["annotation_id"]
    scores = {
        "interaction": interaction_output,
    }
    
    metrics = compute_metrics(
        labels.loc[annotation_ids],
        scores
    )

    display_metrics = dict()
    task_accuracies = metrics["accuracies"]["interaction"]
    for k, task_accuracy in zip((1, 5), task_accuracies):
        display_metrics[f"all_interaction_accuracy_at_{k}"] = task_accuracy
    display_metrics[f"all_interaction_mCA"] = metrics["mCA"]
    display_metrics[f"all_interaction_mAP"] = metrics["mAP"]
    display_metrics = {metric: value * 100 for metric, value in display_metrics.items()}
    display_metrics[f"all_interaction_mAUC"] = metrics["mAUC"]
    pprint(display_metrics)


if __name__ == "__main__":
    main(parser.parse_args())