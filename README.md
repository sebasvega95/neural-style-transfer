# Neural style transfer

This is a TensorFlow implementation for performing style transfer using neural
networks. The code uses key ideas from the [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) and
[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) papers.

## Requirements

The code is written in Python 3.5+ and uses TensorFlow. The requirements
can be found in `top-requirements.txt` and installed using
`pip install -r top-requirements.txt`.

## Training

For training a model from scratch, we need to specify a folder containing the
content images and the path to the style image. For example, assuming we've
downloaded the [MS COCO 2014 dataset](http://cocodataset.org/#download) and put
the training images to `data/train`. We can train a model to apply the style of
`images/style/wave.jpg` with the command

```python
python3 train.py --style=images/style/wave.jpg --train_path=data/train --weights_path=weights/wave.hdf5
```

This will produce a weights file that we can use to generate new images from
unseen content images 🎉.

#### Arguments

- `--style` Path to style image. Required.
- `--train` Path to training (content) images. Required.
- `--weights` Path where to save the model's weights. Required.

## Styling

We can style new content images from a path or from a webcam 📷.

For example, for styling `images/content/viaducto.jpg` with weights located
in `weights/wave.hdf5` run

```python
python3 style.py  --content=images/content/viaducto.jpg --weights=weights/wave.hdf5
```

For styling from a webcam feed, run

```python
python3 style.py --weights=weights/wave.hdf5 --webcam
```

#### Arguments

- `--content` Path to content image. Required if `--webcam` isn't used.
- `--gen` Path where to save the generated image. If it isn't provided the image
will be shown on screen.
- `--weights` Path to model's weights. Required.
- `--webcam` Generate images from the webcam. Can't be used if a content image
is specified.

## Example

<div align="center">
  <img height="250px" src="images/content/viaducto.jpg" />
  <img height="250px" src="images/style/wave.jpg" />
  <img height="502px" src="images/generated/viaducto_wave.jpg" />
</div>

## Weights

These are the weights used for some style images (which can be found in
`images/style/`)

- `wave`:
  - Content weight: 1
  - Style weight: 100
  - Total variation weight: 1e-5
- `mosaic`:
  - content: 3
  - Style weight: 100
  - Total variation weight: 1e-5
