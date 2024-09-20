## Overall Architecture

The data processing code for 3D-SpecDIM is mainly divided into two parts. The spectral feature recognition based on ViT is implemented by python, and the other data processing is implemented by Matlab.

## Instruction for use
### Single molecule spectrum dynamics tracking with 3D-SpecDIM

The raw data for 3D-SpecDIM consists of two tdms files, prefixed TRXXX.tdms and SMXXX.tdms, which includes the 3D tracking information and spectral information for the tracking particle, respectively. Here we provide the raw data in Figure 2a-c ([Fluorescent Beads Data](https://drive.google.com/drive/folders/1Dw9yuSNdgijvf85wU_PdkR6uYqLkI3QU?usp=drive_link)), other data will be provided upon reasonable request due to file size.

#### 3D-SpecDIM Data Processing

Before starting the data processing, you should open the file `./matlab/traj3St_colormap_Spec.m` and specify the `[YourFilePath]` on Line 7.

If you have download the Fluorescent bead data, you can run `./matlab/traj3St_colormap_Spec.m`. A pop-up window will appear in the program, please select `230828 TR012.tdms` file. Then, the program will output the 3D trajectory of fluorescent bead diffusing in water solution with diffrent color coding, including spectrum and time.

For various spectral data processing methods, we offer corresponding implementations, including the analysis of spectral centroid using a normal distribution (`./matlab/AnalysisSpecImg_centroid.m`) and the proportion of fluorescent probes through intensity analysis (`./matlab/AnalysisSpecImg_pH.m`) or spectral unmixing (`./matlab/AnalysisSpecImg_pH_unmixng_v2.m`). To switch between these methods, simply modify the corresponding statement on Line 19.

#### Spectral Unmixing
For spectral unmixing, the original emission spectrum should be specified in advance. Due to the limitation of file size, we only provide the spectral image matlab file extracted from the corresponding original data in Fig3b-f ([Spectral Unmixing Data](https://drive.google.com/drive/folders/1WsiG2Vk7DymLLZQuZc745vVBpigkCdQi?usp=drive_link)). You can run  `./matlab/AnalysisSpecImg_pH.m` and `./matlab/AnalysisSpecImg_pH_unmixng_v2.m` separately to compare the difference between the spectral unmixing method and the wavelength-split detection method.

### SpecViT

#### Environment Preparation

Before training or testing, you need to prepare the deep-leanring environment:


```
NVIDIA GPU + CUDA CuDNN
conda create --name 3D-SpecDIM python=3.10.9
Linux OS
```

then, run the followng command in terminal:

```
conda activate 3D-SpecDIM
pip install -r requirement.txt
```

Installation will be completed quickly

#### Data Preparation

For the simulated datasets generation, changes the `[savedir]` on Line 11 in `./SpecViT/simu_data_gen/simu_data_gen_pos.py`. Then run the on the terminal:

```
python ./SpecViT/simu_data_gen/simu_data_gen_pos.py
```

#### Training process
 
Run
```
cd SpecViT
python ./train.py --dataroot [savedir] --model ViT_skip --batchSize 128 --gpu_ids 0,1 --pos_manner learned --pretrained_filename   --N_per_C 720 
```

Here we have provided the pretrained weights file ([pretrained_weights](https://drive.google.com/drive/folders/1wOQTmSC3L1ItfSwdXGO1EAIJoGM1puVS?usp=drive_link)), please download it and put it to the `./SpecViT/checkpoints/`

#### Domain Adaption

Modify the datatset to domain_dataset on Line 12 in `./SpecViT/train.py`:
```
from datasets.domain_spec_dataset import CustomDataModula
```

Download the domain_datasets ([domain_datasets](https://drive.google.com/drive/folders/1js-kvqsNWNKmCVebctWe9T1cHRPGH6xW?usp=drive_link)) and put it to the `./SpecViT/data/`, then run:

```
python ./train.py --dataroot ./data/domain_adaption --model MDD_ViT --batchSize 128 --gpu_ids 0,1 --pos_manner learned --pretrained_filename ./checkpoints/pretrained_model.ckpt
```

#### Test process

Download the test_datasets ([experimental_datasets](https://drive.google.com/drive/folders/14oCcPLTFosCsw6-ZsntoI6VTAQDREl3V?usp=drive_link)) and put it to the `./SpecViT/data/`, then run:

```
python ./test.py --dataroot ./data/test_dataset --model MDD_ViT --batchSize 1 --gpu_ids 0 --pretrained_filename ./checkpoints/MDD_ViT-learned-latest.ckpt
```
