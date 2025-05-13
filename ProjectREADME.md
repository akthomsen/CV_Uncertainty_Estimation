# A Comparative Study of Uncertainty Estimation Methods in 2D Object Detection: BayesOD and Deep Ensembles
> Comparative study & reference implementation — Computer Vision, Aarhus University (2025)
## Overview
Deep neural‑network object detectors such as RetinaNet achieve state‑of‑the‑art accuracy, yet deploy poorly in safety‑critical systems because they lack calibrated *uncertainty* estimates.

This repository reproduces and compares two methods for incorporating uncertainty estimation on a common code-base and dataset (BDD100K): 
1. [BayesOD](https://www.researchgate.net/publication/344983540_BayesOD_A_Bayesian_Approach_for_Uncertainty_Estimation_in_Deep_Object_Detectors), introduced by Harakeh et al. in 2019
2. [Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html), proposed by Lakshminarayanan et al. in 2017.



## Get Started
### 1. Prerequisites
- Python (3.7)
- PyTorch (1.10.0)
- CUDA (11.3)
- cuDNN (8.7.0)
- Detectron2 (0.6)
- The available modules on the cluster (see ```module avail```).
- The compute capabilities of the GPU (s86).

### 2. Installation Steps
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