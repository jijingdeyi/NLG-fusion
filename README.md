# Text-guided infrared and visible image fusion

This is the implementation code of our paper "".

To begin with, you need to create your own virtual environment. We provide our environment configuration in `environment.yml`. You just need to activate your environment and run
```shell
conda env create -f environment.yml
```
and then activate the new virtual environment:
```shell
conda activate reconet
```

## For testing
We provide our test datasets in `../data`, which includes four test sets from four public infrared and visible image fusion datasets. 

Pretrained parameters are provided in [Google drive](https://drive.google.com/drive/folders/10z5WKA-iJFtix_4hBMfkhdROyjZAtasW?usp=drive_link)

If you just want to test the performance of our model, you just need to run:

```shell
# set project path for python
export PYTHONPATH="${PYTHONPATH}:$ROOT"
python scripts/test.py --data $your_data_path (e.g.  data/TNO_test) --ckpt checkpoint/train_llvip_622.ckpt --dst $your_save_path
```
`$ROOT`: your project root

`$your_data_path`: path of your testing data 

`$your_save_path`: path of your fusion result 

Then you can obtain the fusion result which we already put in `../result`. 

## For training
If you want to train the model yourself:

```shell
# set project path for python
export PYTHONPATH="${PYTHONPATH}:$ROOT"
python scripts/train.py --data $DATA --ckpt $CHECKPOINT_PATH --lr 1e-3
python scripts/pred.py --data $DATA --ckpt $CHECKPOINT_PATH --dst $result_path
```

`$DATA`: path of your training or testing dataset

`$CHECKPOINT_PATH`: path of your checkpoint files

`$result_path`: path of your fusion result

## Experiment details
When we train on LLVIP dataset, the text for task is "This is an infrared and visible image fusion task.", the text for visible images is "low light degradation" and the text for infrared images is "low contrast and blurred"; when we predict on LLVIP and MSRS dataset, the text for task is "This is an infrared and visible image fusion task.", the text for visible images is "maybe low light degradation and overexposure degradation in visible images." and the text for infrared images is "low contrast issues"  ratio 0.6 0.2 0.2

when we predict on RoadScene dataset, the text for task is "This is an infrared and visible image fusion task. Should preserve more visible image information", the text for visible images is "The brightness is too high, and some scene lights are too bright; it has less noise" and the text for infrared images is "low contrast issues"  ratio 0.6 0.3 0.1

## Metric
The python code for calculating various metrics in our paper can refer to : [CSDN blog](https://blog.csdn.net/fovever_/article/details/129332278?spm=1001.2014.3001.5501)

