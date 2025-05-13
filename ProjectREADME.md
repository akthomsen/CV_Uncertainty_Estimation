# A Comparative Study of Uncertainty Estimation Methods in 2D Object Detection: BayesOD and Deep Ensembles
> Comparative study & reference implementation — Computer Vision, Aarhus University (2025)
## Overview
Deep neural‑network object detectors such as RetinaNet achieve state‑of‑the‑art accuracy, yet deploy poorly in safety‑critical systems because they lack calibrated *uncertainty* estimates.

This repository reproduces and compares two methods for incorporating uncertainty estimation on a common code-base and dataset (BDD100K): 
1. [BayesOD](https://www.researchgate.net/publication/344983540_BayesOD_A_Bayesian_Approach_for_Uncertainty_Estimation_in_Deep_Object_Detectors), introduced by Harakeh et al. in 2019
2. [Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), proposed by Lakshminarayanan et al. in 2017.



## Quick Start
### Prerequisites
- Python (3.7)
- PyTorch (1.10.0)
- CUDA (11.3)
- cuDNN (8.7.0)
- Detectron2 (0.6)
- The available modules on the cluster (see ```module avail```).
- The compute capabilities of the GPU (s86).

### Installation Steps
1. Install Python 3.7 and make it the global Python version by running the following commands:
    ```bash
    pip install pyenv
    pyenv install 3.7
    pyenv global 3.7
    ```
2. From the available modules, load the following:
    ```bash
    module load cuda-11.3
    module load cudnn-11.X-8.7.0
    ```
3. Install the requirements using ```pip install -r requirements.txt```.
4. Install GPU-compatible ```torch```  and ```torchvision``` using the following command:
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```
5. Install the ```detectron2``` package by running the command below. This is the only one compatible with the current version of ```torch``` and ```torchvision``` and the CUDA version.
    ```bash
    python -m pip install detectron2==0.6 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
    ```

6. For some reason the 0.6 version of ```detectron2``` is not compatible with the current package's expectation of the implementation of retinanet. In order to mitigate, do the following:
    1. Go to [the detectron2 webpage](https://github.com/facebookresearch/detectron2/releases) and download release 0.4 as zip. 
    2. Unzip the file and go to the ```/Users/au691667/Downloads/detectron2-0.4/detectron2/modeling/meta_arch```folder. 
    3. Copy the ```retinanet.py``` file and paste it in the ```.pyenv/versions/3.7/site-packages/detectron2/modeling/meta_arch``` folder.


## Dataset

### BDD Dataset
Download the Berkeley Deep Drive (BDD) Object Detection Dataset [here](https://bdd-data.berkeley.edu/). The BDD
dataset should have the following structure:
<br>
 
     └── BDD_DATASET_ROOT
         ├── info
         |   └── 100k
         |       ├── train
         |       └── val
         ├── labels
         └── images
                ├── 10K
                └── 100K
                    ├── test
                    ├── train
                    └── val


For all BDD dataset, labels need to be converted to COCO format. To do so, run the following:
```bash
python src/core/datasets convert_bdd_to_coco.py --dataset-dir /path/to/bdd/dataset/root
```
If the script to convert BDD labels to COCO format does not work, please use [these pre-converted labels](https://drive.google.com/file/d/1hOd3zX1Qt0_uV64uJBLidavjbtrv1tXI/view?usp=sharing).

## Training
To train the model(s) in the paper, run these commands:

1. **Create session**  
   ```bash
   screen -S training
   ```

2. **Run training script**  
   ```bash
   python train.py ...
   ```
   - Example of use of command:
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

## Evaluation
For running inference and evaluation of a model, run the following code:
```eval
python src/apply_net.py --dataset-dir /path/to/test/dataset/root --test-dataset test_dataset_name --config-file BDD-Detection/retinanet/name_of_config.yaml --inference-config Inference/name_of_inference_config.yaml
```

`--test-dataset` can be one of `bdd_val`, `kitti_val`, or `lyft_val`. `--dataset-dir` corresponds to the root directory of the dataset used.

Evaluation code will run inference on the test dataset and then will generate mAP, Negative Log Likelihood, Calibration Error, and Minimum Uncertainty Error results. If only evaluation of metrics is required,
add `--eval-only` to the above code snippet.

## Results

### Overview of trained models
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

### Evaluation with metrics
![Method comparison](docs\img\evaluation.PNG)

For a more detailed evaluation, see the project [notes](docs/notes.md).
