# Notes
This document serves as notes for the group on how to use the pod_compare package.

# Table of Contents
- [Notes](#notes)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Installation Steps](#installation-steps)
  - [Dataset](#dataset)
    - [BDD Dataset](#bdd-dataset)
  - [Code changes](#code-changes)
  - [Training](#training)
    - [Useful Screen Commands:](#useful-screen-commands)
- [Results](#results)
  - [Overview of trained models](#overview-of-trained-models)
  - [Evaluation with metrics](#evaluation-with-metrics)
    - [M1](#m1)
    - [M4: Loss Attenuation + Dropout](#m4-loss-attenuation--dropout)
    - [❗ M6: BayesOD + Dropout](#-m6-bayesod--dropout)
    - [❗ M7: Pre-NMS Ensembles](#-m7-pre-nms-ensembles)
      - [Random seed: 1](#random-seed-1)
      - [Random seed: 42](#random-seed-42)
      - [Random seed: 713](#random-seed-713)
      - [random seed: 100](#random-seed-100)
      - [Random seed: 1000](#random-seed-1000)
      - [Inference](#inference)
    - [M9](#m9)

# Installation
**The steps in the [README](README.md) should not be followed.** The following are the steps we used to reproduce the environment.

The main issue was finding compatible versions of:
- Python (3.7)
- PyTorch (1.10.0)
- CUDA (11.3)
- cuDNN (8.7.0)
- Detectron2 (0.6)
- The available modules on the cluster (see ```module avail```).
- The compute capabilities of the GPU (s86).

Where (.) is the final version of the package/component.

## Installation Steps

- Install Python 3.7 and make it the global Python version by running the following commands:
    ```bash
    pip install pyenv
    pyenv install 3.7
    pyenv global 3.7
    ```
- From the available modules, load the following:
    ```bash
    module load cuda-11.3
    module load cudnn-11.X-8.7.0
    ```
- In [requirements.txt](requirements.txt) outcomment ```torch```, ```torchvision```, and ```git+https://github.com/facebookresearch/detectron2.git```.
- Install the requirements using ```pip install -r requirements.txt```.
- Install GPU-compatible ```torch```  and ```torchvision``` using the following command:
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
- Install the ```detectron2``` package by running the command below. This is the only one compatible with the current version of ```torch``` and ```torchvision``` and the CUDA version.
    ```bash
    python -m pip install detectron2==0.6 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
    ```
- For some reason the 0.6 version of ```detectron2``` is not compatible with the current package's expectation of the implementation of retinanet. In order to mitigate, do the following:
    1. Go to [the detectron2 webpage](https://github.com/facebookresearch/detectron2/releases) and download release 0.4 as zip. 
    2. Unzip the file and go to the ```/Users/au691667/Downloads/detectron2-0.4/detectron2/modeling/meta_arch```folder. 
    3. Copy the ```retinanet.py``` file and paste it in the ```.pyenv/versions/3.7/site-packages/detectron2/modeling/meta_arch``` folder.

## Dataset

### BDD Dataset
To download the BDD dataset, run the following commands:
```bash
cd pod_compare
mkdir BDD_DATASET_ROOT
wget http://128.32.162.150/bdd100k/bdd100k_images_100k.zip
wget http://128.32.162.150/bdd100k/bdd100k_labels.zip
wget http://128.32.162.150/bdd100k/bdd100k_info.zip
unzip bdd100k_images_100k.zip
unzip bdd100k_labels.zip
unzip bdd100k_info.zip
```
Follow the instructions in the [README](README.md) file to see the directory structure of the dataset. Since the code expects all labels to be in one file, we have created [file_concatenation](file_concatenation.py) to concatenate all the labels into one file. Run this for both validation and training labels. 

The labels are not structured exactly the same as expected by the code. Therefore, in [convert_bdd_to_coco.py](src/core/datasets/convert_bdd_to_coco.py) replace:
```python
for annotation in annotations:
    if annotation['category'] in category_keys:
        bbox = annotation['bbox']
        bbox_coco = [
            bbox[0],
            bbox[1],
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]]
        annotations_list.append({'image_id': im_id,
                                    'id': count,
                                    'category_id': category_mapper[annotation['category']],
                                    'bbox': bbox_coco,
                                    'area': bbox_coco[2] * bbox_coco[3],
                                    'iscrowd': 0})
        count += 1

```
With:
```python
for annotation in annotations:
    for l in annotation['labels']:
        if l.get('category') in category_keys:
            bbox = list(l['box2d'].values())
            bbox_coco = [
                bbox[0],
                bbox[1],
                bbox[2] - bbox[0],
                bbox[3] - bbox[1]]
            annotations_list.append({'image_id': im_id,
                                    'id': count,
                                    'category_id': category_mapper[l['category']],
                                    'bbox': bbox_coco,
                                    'area': bbox_coco[2] * bbox_coco[3],
                                    'iscrowd': 0})
            count += 1

```

**Alternatively it seems that the authors have provided their converted files also linked to in the [README](README.md) file.**

## Code changes
In ```src/probabilistic_inference/probabilistic_inference.py``` in lines 207 and 449, change the following:
```python
n_fms = len(self.model.in_features)
```
to
```python
n_fms = len(self.model.head_in_features)
```

## Training

1. **Create session**  
   ```bash
   screen -S training
   ```

2. **Run training script**  
   ```bash
   python train.py ...
   ```
   - The command is:
        ``` train
        python src/train_net.py --num-gpus 2 --dataset-dir BDD_DATASET_ROOT --config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var.yaml --random-seed 42 --resume
        ```
3. **Detach session**  
   Press `Ctrl+A` then `D`

4. **Reattach later**  
   ```bash
   screen -r training
   ```

### Useful Screen Commands:
- List sessions: `screen -ls`
- Kill session: `screen -XS training quit`


# Results

## Overview of trained models
Unique ID | Method Name | Config File | Inference Config File | Trained
--- | --- | --- | --- | ---
M1 |Baseline RetinaNet | retinanet_R_50_FPN_1x.yaml| standard_nms.yaml | 🟩 
M2 |Output Redundancy| retinanet_R_50_FPN_1x.yaml | anchor_statistics.yaml | 🟩 
M3 |Loss Attenuation |retinanet_R_50_FPN_1x_reg_cls_var.yaml| standard_nms.yaml | 🟩
M4 |Loss Attenuation + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | mc_dropout_ensembles_pre_nms.yaml | 🟩
M5 |BayesOD | retinanet_R_50_FPN_1x_reg_cls_var.yaml | bayes_od.yaml | 🟩
M6 |BayesOD + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | bayes_od_mc_dropout.yaml | 🟩
M7 |Pre-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_pre_nms.yaml | 🟩
M8 |Post-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_post_nms.yaml | 🟩
M9 |Black Box| retinanet_R_50_FPN_1x_dropout.yaml | mc_dropout_ensembles_post_nms.yaml | 🟥

## Evaluation with metrics

### M1
Output from finished training:
```bash
[04/24 00:00:39 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/24 00:00:44 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 5.42 seconds.
[04/24 00:00:45 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/24 00:00:46 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 1.88 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.515
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.244
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
[04/24 00:00:46 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 29.556 | 53.620 | 27.393 | 7.606 | 31.272 | 51.482 |
[04/24 00:00:46 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.068 | bus        | 41.053 | truck      | 38.713 |
| person     | 27.297 | rider      | 18.987 | bike       | 19.721 |
| motor      | 16.052 |            |        |            |        |
[04/24 00:00:47 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[04/24 00:00:47 d2.evaluation.testing]: copypaste: Task: bbox
[04/24 00:00:47 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/24 00:00:47 d2.evaluation.testing]: copypaste: 29.5557,53.6202,27.3928,7.6064,31.2722,51.4816
```

### M4: Loss Attenuation + Dropout 
Command for inference:
```bash
 python src/apply_net.py --dataset-dir BDD_DATASET_ROOT --test-dataset bdd_val --config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml --inference-config Inference/mc_dropout_ensembles_pre_nms.yaml --random-seed 1
```

Output after inference:
```bash
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:34<00:00, 286.05it/s]
+------------------+---------------------+---------------------+---------------------+
|   Output Type    | Number of Instances | Cls Ignorance Score | Reg Ignorance Score |
+------------------+---------------------+---------------------+---------------------+
| True Positives:  |        72550        |        0.8119       |       14.9692       |
| False Positives: |        363110       |        0.1286       |       17.7588       |
| False Negatives: |         1125        |          -          |          -          |
+------------------+---------------------+---------------------+---------------------+
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
| Cls Marginal Calibration Error | Reg Expected Calibration Error | Reg Maximum Calibration Error | Cls Minimum Uncertainty Error | Reg Minimum Uncertainty Error |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
|             0.0362             |             0.0407             |             0.0851            |             0.1398            |             0.4361            |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
```

### ❗ M6: BayesOD + Dropout
Command for inference:
```bash
python src/apply_net.py --dataset-dir BDD_DATASET_ROOT --test-dataset bdd_val --config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml --inference-config Inference/bayes_od_mc_dropout.yaml --random-seed 1
```

Output after inference:
```bash
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [00:34<00:00, 290.14it/s]
+------------------+---------------------+---------------------+---------------------+
|   Output Type    | Number of Instances | Cls Ignorance Score | Reg Ignorance Score |
+------------------+---------------------+---------------------+---------------------+
| True Positives:  |        72463        |        0.8139       |       12.5524       |
| False Positives: |        363107       |        0.1286       |       16.8985       |
| False Negatives: |         1126        |          -          |          -          |
+------------------+---------------------+---------------------+---------------------+
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
| Cls Marginal Calibration Error | Reg Expected Calibration Error | Reg Maximum Calibration Error | Cls Minimum Uncertainty Error | Reg Minimum Uncertainty Error |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
|             0.0362             |             0.0121             |             0.0321            |             0.1384            |             0.2202            |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
```

### ❗ M7: Pre-NMS Ensembles

#### Random seed: 1
Output from finished training:
```bash
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.416
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661
[04/22 00:49:34 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 28.939 | 52.214 | 26.893 | 7.265 | 30.540 | 51.176 |
[04/22 00:49:34 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.255 | bus        | 40.761 | truck      | 39.056 |
| person     | 27.164 | rider      | 17.492 | bike       | 19.002 |
| motor      | 13.843 |            |        |            |        |
[04/22 00:49:35 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[04/22 00:49:35 d2.evaluation.testing]: copypaste: Task: bbox
[04/22 00:49:35 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/22 00:49:35 d2.evaluation.testing]: copypaste: 28.9391,52.2135,26.8932,7.2648,30.5397,51.1755
```
#### Random seed: 42
Output from finished training:
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.282
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.317
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.671
[04/30 03:36:42 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 30.012 | 53.546 | 28.211 | 7.679 | 31.732 | 52.729 |
[04/30 03:36:42 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.698 | bus        | 41.511 | truck      | 39.912 |
| person     | 27.875 | rider      | 18.820 | bike       | 20.300 |
| motor      | 15.970 |            |        |            |        |
[04/30 03:36:43 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[04/30 03:36:43 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 03:36:43 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 03:36:43 d2.evaluation.testing]: copypaste: 30.0121,53.5460,28.2110,7.6790,31.7320,52.7294
```
#### Random seed: 713
Output from finished training:
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.282
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.420
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
[04/30 21:15:40 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 29.594 | 52.770 | 28.199 | 7.603 | 31.214 | 52.231 |
[04/30 21:15:40 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.552 | bus        | 40.659 | truck      | 39.583 |
| person     | 27.748 | rider      | 18.769 | bike       | 19.792 |
| motor      | 15.053 |            |        |            |        |
[04/30 21:15:41 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[04/30 21:15:41 d2.evaluation.testing]: copypaste: Task: bbox
[04/30 21:15:41 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/30 21:15:41 d2.evaluation.testing]: copypaste: 29.5935,52.7704,28.1993,7.6031,31.2139,52.2310
```

#### random seed: 100
```bash
[05/01 11:18:07 d2.evaluation.evaluator]: Total inference time: 0:09:09.549562 (0.110020 s / iter per device, on 2 devices)
[05/01 11:18:07 d2.evaluation.evaluator]: Total inference pure compute time: 0:09:00 (0.108284 s / iter per device, on 2 devices)
[05/01 11:18:13 d2.evaluation.coco_evaluation]: Preparing results for COCO format ...
[05/01 11:18:13 d2.evaluation.coco_evaluation]: Saving results to /home/cv09f25/pod_compare/data/BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var/random_seed_100/inference/coco_instances_results.json
[05/01 11:18:17 d2.evaluation.coco_evaluation]: Evaluating predictions with unofficial COCO API...
Loading and preparing results...
DONE (t=2.70s)
creating index...
index created!
[05/01 11:18:20 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[05/01 11:18:25 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 5.42 seconds.
[05/01 11:18:25 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/01 11:18:27 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 1.55 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.536
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.316
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.246
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.506
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
[05/01 11:18:27 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 30.036 | 53.606 | 28.308 | 7.566 | 31.601 | 53.141 |
[05/01 11:18:27 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.599 | bus        | 41.753 | truck      | 39.913 |
| person     | 27.802 | rider      | 19.053 | bike       | 20.399 |
| motor      | 15.734 |            |        |            |        |
[05/01 11:18:27 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[05/01 11:18:27 d2.evaluation.testing]: copypaste: Task: bbox
[05/01 11:18:27 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/01 11:18:27 d2.evaluation.testing]: copypaste: 30.0360,53.6063,28.3084,7.5660,31.6008,53.1411
```

#### Random seed: 1000
```bash
[05/02 05:38:35 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 5.39 seconds.
[05/02 05:38:35 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[05/02 05:38:37 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 1.53 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.299
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.283
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.203
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.666
[05/02 05:38:37 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 29.915 | 53.234 | 28.267 | 7.868 | 31.517 | 52.487 |
[05/02 05:38:37 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 45.649 | bus        | 41.533 | truck      | 39.742 |
| person     | 27.895 | rider      | 18.869 | bike       | 19.647 |
| motor      | 16.068 |            |        |            |        |
[05/02 05:38:37 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[05/02 05:38:37 d2.evaluation.testing]: copypaste: Task: bbox
[05/02 05:38:37 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/02 05:38:37 d2.evaluation.testing]: copypaste: 29.9147,53.2340,28.2675,7.8676,31.5175,52.4867
```

#### Inference

**IMPORTANT:** Go to ```src/configs/Inference``` and change the ```RANDOM_SEED_NUMS``` variable to the random seeds used in the ensemble.

Command for inference:
```bash
python src/apply_net.py --dataset-dir BDD_DATASET_ROOT --test-dataset bdd_val --config-file BDD-Detection/retinanet/retinanet_R_50_FPN_1x_reg_cls_var.yaml --inf
erence-config Inference/ensembles_pre_nms.yaml
```

Output after inference:
```bash
```



### M9
trining output:
```bash
[04/22 18:36:34 d2.evaluation.fast_eval_api]: Evaluate annotation type *bbox*
[04/22 18:36:39 d2.evaluation.fast_eval_api]: COCOeval_opt.evaluate() finished in 5.16 seconds.
[04/22 18:36:39 d2.evaluation.fast_eval_api]: Accumulating evaluation results...
[04/22 18:36:41 d2.evaluation.fast_eval_api]: COCOeval_opt.accumulate() finished in 1.45 seconds.
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.074
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.241
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656
[04/22 18:36:41 d2.evaluation.coco_evaluation]: Evaluation results for bbox:
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 29.011 | 53.051 | 26.815 | 7.410 | 30.437 | 51.353 |
[04/22 18:36:41 d2.evaluation.coco_evaluation]: Per-category bbox AP:
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| car        | 44.699 | bus        | 40.970 | truck      | 38.736 |
| person     | 27.031 | rider      | 17.296 | bike       | 19.201 |
| motor      | 15.148 |            |        |            |        |
[04/22 18:36:41 d2.engine.defaults]: Evaluation results for bdd_val in csv format:
[04/22 18:36:41 d2.evaluation.testing]: copypaste: Task: bbox
[04/22 18:36:41 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[04/22 18:36:41 d2.evaluation.testing]: copypaste: 29.0114,53.0515,26.8150,7.4095,30.4369,51.3532
```