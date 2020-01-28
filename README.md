# vision4ofx
torchvision -> torchScript -> openFrameworks  

## Description
faster rcnn

## Requirement
windows

## Installation
[Anadonda](https://www.anaconda.com/distribution/) Python 3.7 version  
 
``` Installation.sh
$ conda create -n pt14 python=3.6 -y 
$ conda create -n pt14  
$ conda install -c pytorch pytorch=1.4.0 torchvision=0.5.0 -y
$ pip install opencv-python
```

[openFrameworks](https://openframeworks.cc/download/) Windows 

## demo on python
```
$ conda activate pt14
$ python py/demo_python.py
```
fasterrcnn: 12fps  
maskrcnn: 4fps  
keypointrcnn: 9fps  


## convert model
```
$ python py/convert.py
```

