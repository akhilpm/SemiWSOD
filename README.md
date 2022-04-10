# Semi-Weakly Supervised Object Detection by Sampling Pseudo Ground-Truth Boxes
This code is adapting the Faster R-CNN fully-supervised detector to the semi-weakly supervised settings by our sampling based training algorithm proposed in the paper "Semi-Weakly Supervised Object Detection by Sampling Pseudo Ground-Truth Boxes". 
The original Faster R-CNN implementation is from [loolzaaa](https://github.com/loolzaaa/faster-rcnn-pytorch). In this version, we are not using the python extension compiled for NMS, ROI layers, instead the RoI pooling and NMS of Pytorch is used. 
So there is no compilation steps.

Additionally, visualizations are created for training and testing process using visdom.

### Benchmarking on Fully supervised & Semi-supervised detection
<table>
  <tr>
    <td>Backbone</td>
    <td>Data</td>
    <td>L R & Decay steps</td>
    <td>Batch size</td>
    <td>mAP</td>
  </tr>
  <tr>
    <td colspan="5">Fully-supervised</td>
  </tr>
  <tr>
    <td> Vgg-16 </td>
    <td> VOC 2007</td>
    <td> 1e-3 [5,10]</td>
    <td> 1</td>
    <td> 69.9</td>
  </tr>
  <tr>
    <td> ResNet-101 </td>
    <td> VOC 2007</td>
    <td> 1e-3 [5,10]</td>
    <td> 1</td>
    <td> 74.4</td>
  </tr>
  <tr>
    <td colspan="5">Semi-weakly supervised</td>
  </tr>
  <tr>
    <td> Vgg-16 </td>
    <td> 10% VOC 2007</td>
    <td> 1e-3 [5,10]</td>
    <td> 1</td>
    <td> 60.3</td>
  </tr>
  <tr>
    <td> Vgg-16 </td>
    <td> 20% VOC 2007</td>
    <td> 1e-3 [5,10]</td>
    <td> 1</td>
    <td> 65.5</td>
  </tr>
  <tr>
    <td> ResNet-101 </td>
    <td> VOC 2007(L)+2012(WL)</td>
    <td> 1e-3 [5,10]</td>
    <td> 1</td>
    <td> 79.4</td>
  </tr>
</table>

Here L and WL stands for fully-labeled and weakly labeled respetively. More results are available
in the [paper](https://arxiv.org/pdf/2204.00147.pdf)


---
## Preparation
Clone this repo and create `data` folder in it:
```
git clone https://github.com/akhilpm/SemiWSOD.git
cd SemiWSOD && mkdir data
```

### Prerequisites
- Python 3.5+
- PyTorch 1.3+
- CUDA 8+
- easydict
- opencv-python(cv2)
- scikit-learn
- colorama
- visdom(can be ignored if no visualization is needed)

### Compilation
At present no compilation, pytorch codes are used RoI pooling and NMS etc.

### Pretrained model
1. Download PyTorch RGB pretrained models ([link](https://drive.google.com/drive/folders/1P4Q9jtsMB9C47l7imseK5JlTpgMX1pFh?usp=sharing))
2. Put them into `data/pretrained_model/`

**NOTE:** Please, remember that this network use caffe (*BGR color mode*) pretrained model **by default**. If you want to use PyTorch pretrained models, you must specify *RGB* color mode, image range = [0, 1], *mean = [0.485, 0.456, 0.406]* and *std = [0.229, 0.224, 0.225]* in additional parameters for run script. For example:
```
python run.py train ............. -ap color_mode=RGB image_range=1 mean="[0.485, 0.456, 0.406]" std="[0.229, 0.224, 0.225]"
```

### Data preparation
Prepare dataset as described [here](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) for Pascal VOC.
*Actually, you can use any dataset. Just download it and create softlinks in `library_root/data` folder.*

You can, *but not necessary*, specify directory name for dataset relative `./data` folder in addtional parameters for run script. For example:
- `python run.py train ............. --add_params devkit_path=VOC_DEVKIT_PATH` => ./data/VOC_DEVKIT_PATH
- `python run.py train ............. -ap data_path=COCO2014` => ./data/COCO2014

The pre-computed object proposals I used for VOC dataset using CAM and selective search
proposals is available [here](https://drive.google.com/drive/folders/12ajXPOKlSBnVGHsu6U49GQQX3f3ikAka?usp=sharing).

**NOTE:** Name of the parameter is different for datasets (`devkit_path` for Pascal VOC, `data_path` for COCO, etc.)

**WARNING! If you change any parameter of some dataset, you must remove cache files for this dataset in `./data/cache` folder!**

---
## Usage:
All interaction with the library is done through a `run.py` script. Just run:
```
python run.py -h
```
and follow help message.

### Train
To train Faster R-CNN network with Vggg16 backbone on Pascal VOC 2007 trainval dataset in 10 epochs, run next:
```
python run.py train --net vgg16 --dataset voc_2007_trainval --total-epoch 10 --cuda
```
Some parameters saved in default config file(located at lib/config.py), another parameters has default values. To change the percentage of
supervised datapoints and object proposals types, please edit the corresponcing line in the default config file.


### Test
If you want to evlauate the detection performance of above trained model, run next:
```
python run.py test --net vgg16 --dataset voc_2007_test --epoch $EPOCH --cuda
```
where *$EPOCH* is early saved checkpoint epoch (maximum =10 for training example above).

### Detect
If you want to run detection on your own images with above trained model:
* Put your images in `data/images` folder
* Run script:
```
python run.py detect --net vgg16 --dataset voc_2007_test --epoch $EPOCH --cuda --vis
```
where *$EPOCH* is early saved checkpoint epoch (maximum =10 for training example above).

After detect, you will find the detection results in folder `data/images/result` folder.

For more information about detect script, you need to run `python run.py detect -h` and follow help message.
