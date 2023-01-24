# C9-Audio-Based-Interaction-Recognition

## Challenge

To participate and submit to this challenge, register at the [EPIC-SOUNDS Audio-Based Interaction Recognition Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/9729).
The labelled train/val annoations, along with the recognition test set timestamps are available on the [EPIC-Sounds annotations repo](https://github.com/epic-kitchens/epic-sounds-annotations). The [baseline models](https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src) can also be found here, where the inference script `src/tools/test_net.py` can be used as a template to correctly format models scores for the `create_submission.py` and `evaluate.py` scripts.

This repo is a modified version of the existing [Action Recognition Challenge](https://github.com/epic-kitchens/C1-Action-Recognition).

**NOTE:** For this version of the challenge (version "0.1"), the class "background" (class_id=13) has been redacted from the test set. The arguement `--redact_background` is supported in `evaluate.py` to remove background labels from your validation set evaluation.

## Result data formats

We support two formats for model results.

- *List format*:
  
```{python}
[
    {
        'interaction_output': Iterable of float, shape [44],
        'annotation_id': str, e.g. 'P01_101_1'
    }, ... # repeated for all segments in the val/test set.
]
```

- *Dict format*:

```{python}

{
    'interaction_output': np.ndarray of float32, shape [N, 44],
    'annotation_id': np.ndarray of str, shape [N,]
}

```

Either of these formats can saved via `torch.save` with `.pt` or `.pyth` suffix or with
`pickle.dump` with a `.pkl` suffix.

Note that either of these layouts can be stored in a `.pkl`/`.pt` file--the dict
format doesn't necessarily have to be in a `.pkl`.

## Evaluating model results

We provide an evaluation script to compute the metrics we report in the paper on
the validation set. You will also need to clone the [annotations repo](https://github.com/epic-kitchens/epic-sounds-annotations).
