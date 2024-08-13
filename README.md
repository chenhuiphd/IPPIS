# Incorporating-Polarization-aware-Physical-Inference-and-SAM-for-Image-Dehazing
## Inference

```
python execute/infer_full.py -r checkpoint/full.pth --data_dir <path_to_input_data> --result_dir <path_to_result_data> default
```

## Visualization

Since the file format we use is `.npy`, we provide scrips for visualization:

* use `scripts/visualize_polarized_img.py` to visualize the polarized hazy images
* use `scripts/visualize_img.py` to visualize the unpolarized hazy images and synthetic results
* use `scripts/visualize_real_img.py` to visualize real results

## Preprocess your own data

Note that in our code implementation, the network input contains three components: `{I_alpha, I_hat, delta_I_hat}`:

* `I_alpha`: three polarized hazy images
* `I_hat`: the calculated unpolarized hazy image
* `delta_I_hat`: the calculated unpolarized hazy image multiplied by the degree of polarization

So, we should preprocess the data first to get the network input:

* for synthetic images (training and inference)
  1. use `scripts/preprocess_cityscapes.py` to preprocess the [Cityscapes Dataset](https://www.cityscapes-dataset.com/) (require leftImg8bit, gtFine, and leftImg8bit_transmittanceDBF) for generating `{image, depth, segmentation}` (or choose other source dataset if you want)
  2. use `scripts/make_dataset.py` to generate the synthetic dataset from `{image, depth, segmentation}`

* for real images (inference only)
  1. use `scripts/make_real_dataset_from_raw_format.py` to generate the real dataset from images (in `.raw` format) captured by a polarization camera (Lucid Vision Phoenix polarization camera (RGB) in our paper)

## Training your own model

1. ```
   python sam.py
   
   python execute/train.py -c config/subnetwork1.json
   ```

* All config files (`config/*.json`) and the learning rate schedule function (MultiplicativeLR) at `get_lr_lambda` in `utils/util.py` could be edited

