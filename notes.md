# Notes
This document serves as notes for the group on how to use the pod_compare package.

# Table of Contents
- [Notes](#notes)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Installation Steps](#installation-steps)
  - [Dataset](#dataset)
    - [BDD Dataset](#bdd-dataset)
  - [Training](#training)
    - [Useful Screen Commands:](#useful-screen-commands)
- [Results](#results)
  - [Overview of trained models](#overview-of-trained-models)
  - [Evaluation with metrics](#evaluation-with-metrics)
    - [M1](#m1)
    - [M3](#m3)

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

## Training

1. **Create session**  
   ```bash
   screen -S training
   ```

2. **Run training script**  
   ```bash
   python train.py ...
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
BDD-Detection/retinanet/retinanet_R_50_FPN_1x.yaml


## Overview of trained models
Unique ID | Method Name | Config File | Inference Config File | Trained
--- | --- | --- | --- | ---
M1 |Baseline RetinaNet | retinanet_R_50_FPN_1x.yaml| standard_nms.yaml | 🟨 (not fully) 
M2 |Output Redundancy| retinanet_R_50_FPN_1x.yaml | anchor_statistics.yaml | 🟨 (not fully) 
M3 |Loss Attenuation |retinanet_R_50_FPN_1x_reg_cls_var.yaml| standard_nms.yaml | 🟩
M4 |Loss Attenuation + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | mc_dropout_ensembles_pre_nms.yaml | 🟩
M5 |BayesOD | retinanet_R_50_FPN_1x_reg_cls_var.yaml | bayes_od.yaml | 🟩
M6 |BayesOD + Dropout | retinanet_R_50_FPN_1x_reg_cls_var_dropout.yaml | bayes_od_mc_dropout.yaml | 🟩
M7 |Pre-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_pre_nms.yaml | 🟩
M8 |Post-NMS Ensembles| retinanet_R_50_FPN_1x_reg_cls_var.yaml | ensembles_post_nms.yaml | 🟩
M9 |Black Box| retinanet_R_50_FPN_1x_dropout.yaml | mc_dropout_ensembles_post_nms.yaml | 🟥

## Evaluation with metrics
.


<!-- Unique ID | Method Name | Metric1 | Metric2 | Metric3 | Metric4...
--- | --- | --- | --- | --- | ---
M1 |Baseline RetinaNet | 0.123 | 0.456 | 0.789 | 0.012
M2 |Output Redundancy| 0.123 | 0.456 | 0.789 | 0.012
M3 |Loss Attenuation | 0.123 | 0.456 | 0.789 | 0.012
M4 |Loss Attenuation + Dropout | 0.123 | 0.456 | 0.789 | 0.012
M5 |BayesOD | 0.123 | 0.456 | 0.789 | 0.012
M6 |BayesOD + Dropout | 0.123 | 0.456 | 0.789 | 0.012
M7 |Pre-NMS Ensembles| 0.123 | 0.456 | 0.789 | 0.012
M8 |Post-NMS Ensembles| 0.123 | 0.456 | 0.789 | 0.012
M9 |Black Box| 0.123 | 0.456 | 0.789 | 0.012 -->

### M1
output from inference:
```bash
+------------------+---------------------+---------------------+---------------------+
|   Output Type    | Number of Instances | Cls Ignorance Score | Reg Ignorance Score |
+------------------+---------------------+---------------------+---------------------+
| True Positives:  |        71519        |        1.0239       |      3069.6577      |
| False Positives: |        372490       |        0.1394       |       -1.8071       |
| False Negatives: |         1424        |          -          |          -          |
+------------------+---------------------+---------------------+---------------------+
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
| Cls Marginal Calibration Error | Reg Expected Calibration Error | Reg Maximum Calibration Error | Cls Minimum Uncertainty Error | Reg Minimum Uncertainty Error |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
|             0.0350             |             0.0432             |             0.1287            |             0.1822            |             0.4984            |
+--------------------------------+--------------------------------+-------------------------------+-------------------------------+-------------------------------+
```


### M3
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