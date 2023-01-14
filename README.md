# C9-Audio-Based-Interaction-Recognition

## Challenge
To participate and submit to this challenge, register at the [EPIC-SOUNDS Audio-Based Interaction Recognition Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/9729).
The labelled train/val annoations, along with the recognition test set timestamps are available on the [EPIC-Sounds annotations repo](https://github.com/epic-kitchens/epic-sounds-annotations). The [baseline models](https://github.com/epic-kitchens/epic-sounds-annotations) can also be found here.

This repo is a modified version of the existing [Action Recognition Challenge](https://github.com/epic-kitchens/C1-Action-Recognition)

## Result data formats

We support two formats for model results.

- *List format*:
  ```
  [
      {
          'interaction_output': Iterable of float, shape [44],
          'annotation_id': str, e.g. 'P01_101_1'
      }, ... # repeated for all segments in the val/test set.
  ]
  ```
- *Dict format*:
  ```
  {
      'interaction_output': np.ndarray of float32, shape [N, 44],
      'annotation_id': np.ndarray of str, shape [N,]
  }
  ```

Either of these formats can saved via `torch.save` with `.pt` suffix or with
`pickle.dump` with a `.pkl` suffix.

Note that either of these layouts can be stored in a `.pkl`/`.pt` file--the dict
format doesn't necessarily have to be in a `.pkl`.


## Evaluating model results

We provide an evaluation script to compute the metrics we report in the paper on
the validation set. You will also need to clone the [annotations repo](https://github.com/epic-kitchens/epic-sounds-annotations)

