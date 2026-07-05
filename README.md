<div align='center'>
<h1>BRICE: Brain-inspired Recurrent Illusory Contour Encoder for Point Cloud Completion</h1>
</div>

![example](./mvp_vis.png) 

## [BRICE]

This repo contains a PyTorch implementation for **BRICE: Brain-inspired Recurrent Illusory Contour Encoder for Point Cloud Completion**. 

## [News]
**[2026-07-05]**  We have uploaded the README.md document.

## [Installation]
❗Tips: If you have a configured virtual environment for [CRA-PCN](https://github.com/EasyRy/CRA-PCN), you can reuse it instead of installing a new one.
### Requirements
Our models have been tested on the configuration below:
- python == 3.8.20
- PyTorch == 2.4.1
- CUDA == 11.8

Step 1. Install requirements:
```
pip install -r requirements.txt
```

Step 2. Compile the C++ extension modules:
```
sh install.sh
```

## [Data preparation]
### PCN dataset
❗Tips: If you already have PCN dataset, you should change the data path in train_pcn.py and test_pcn.py:
```
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH   =  './data/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH  =  './data/PCN/%s/complete/%s/%s.pcd'
```
Otherwise, you need to download PCN dataset from [here](https://gateway.infinitescript.com/s/ShapeNetCompletion), 
and then unzip it and put it under ./data.

### MVP dataset
You can download MVP dataset from this [link](https://mvp-dataset.github.io/), and put these two .h5 files in MVP folder.
The input & output resolution is 2048.

### PlantPCom dataset
You can download PlantPCom dataset from this [link](https://drive.google.com/file/d/1aJSs3iaMcyP2C7Hkzo98SDxwBytxeigB/view?usp=sharing).


❗After data preparation, the overall directory structure should be:
```
│BRICE/
├──datasets/
├──data/
│   ├──PCN/
│   ├──PlantPCom/
│   ├──MVP/
│   │   ├──MVP_Test_CP.h5
│   │   ├──MVP_Train_CP.h5
├──.......
```


## [Acknowledgement]
This repo is heavily based on [PCN](https://github.com/qinglew/PCN-PyTorch), [SeedFormer](https://github.com/hrzhou2/seedformer), [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet), [AnchorFormer](https://github.com/chenzhik/AnchorFormer), [PointAttN](https://github.com/ohhhyeahhh/PointAttN), [ODGNet](https://github.com/corecai163/ODGNet), [CRA-PCN](https://github.com/EasyRy/CRA-PCN), and [SymmCompletion](https://github.com/HKUST-SAIL/SymmCompletion/tree/main).
We thank for their excellent works.
