# Real-Time Self-Adaptive Deep Stereo
Code for "Real-time self-adaptive deep stereo", if you use this code please cite 

```
@InProcedings{
    title = {Real-time self-adaptive deep stereo},
    author = {Tonioni, Alessio and Poggi, Matteo and Tosi, Fabio and Mattoccia, Stefano and Di Stefano, Luigi},
    booktitle = {Under Review...}
    month = {Oct},
    Years = {2018}    
}
```

## Requirements
This software has been tested with python3 and tensorflow 1.10. All required packages can be installed using pip and requirements.txt

```
pip3 install -r requirements.txt
```

## Pretrained Weights for Network
Pretrained weights for both DispNet and MADNet available [here](https://drive.google.com/open?id=1GwGxBOFx-NlUo9RAUgPlgPvaHCpGedlm).

## Online Adaptation Step by step

1. Create a csv file for your video sequence similar to `example_list.csv`. 
   
    Each row should contain absolute paths to the input data in the following order:

    "*path_to_left_rgb*,*path_to_right_rgb*,*path_to_groundtruth*"

    Ground truth data will only be used to compute the network performance not for the online adaptation.

2. Download pretrained network from [here](https://drive.google.com/open?id=1GwGxBOFx-NlUo9RAUgPlgPvaHCpGedlm).
3. Perform different kinds of online adaptation with `Stereo_Online_Adaptation.py`, to list all available options use  `python3 Stereo_Online_Adaptation.py -h`. 

Example of online adaptation using MADNet and MAD:
```bash
LIST="path/to/the/list/of/frames/" #the one described at step (1)
OUTPUT="path/to/output/folder"
WEIGHTS="path/to/pretrained/network"
MODELNAME="MADNet"
BLOCKCONFIG="block_config/MadNet_full.json"

python3 Stereo_Online_Adaptation.py \
    -l ${LIST} \
    -o ${OUTPUT} \
    --weights ${WEIGHTS} \
    --modelName MADNet \
    --blockConfig ${BLOCKCONFIG} \
    --mode MAD --sampleMode PROBABILITY 
```

Additional details on available parameters:

+ **--modelName**: choose which disparity inference model to use, currently supports either "DispNet" or "MADNet"
+ **--mode**: how to perform online adaptation: NONE (no online adaptation only inference), FULL (adaptation by full online backprop), MAD (adaptation by partial backprop using MAD)
+ **--blockConfig**: path to a json file that specify how to subdivide the network into independent blocks for adaptation using MAD, some example are provided inside the `block_config` folder. Each innermost array inside the json file should correspond to layers belonging to the same independent group.
+ **--sampleMode**: specify how to sample portions of the network to train when using MAD. 
+ **--weights**: path to the initial weights for the network, e.g., referring to the weights linked above and assuming of having extracted the zip file in the same folder as the python script, to load MADNet trained on synthetic data use `--weights MADNet/synthetic/weights.ckpt`

## Native Correlation layers

All the test in the paper were conducted using correlation layers implemented in pure tensorflow operations, however we provide also a native CUDA version of the same layer that might be usefull for deployment on memory constrained devices. To use the native correlation layer:
1. `cd Nets/Native`
2. `make`
3. `cd ..`
4. `vim sharedLayers.py`
5. Comment line 6, uncomment line from 12 to 19