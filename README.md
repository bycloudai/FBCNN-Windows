## [Towards Flexible Blind JPEG Artifacts Removal (FBCNN, ICCV 2021)](https://arxiv.org/abs/2109.14573)
[![paper download](https://img.shields.io/badge/arxiv-paper-red.svg)](https://arxiv.org/abs/2109.14573)
[![GitHub Stars](https://img.shields.io/github/stars/jiaxi-jiang/FBCNN?style=social)](https://github.com/jiaxi-jiang/FBCNN)
[![download](https://img.shields.io/github/downloads/jiaxi-jiang/FBCNN/total.svg)](https://github.com/jiaxi-jiang/FBCNN/releases)
![python version](https://img.shields.io/badge/python-3.7+-yellow.svg) 
![pytorch version](https://img.shields.io/badge/pytorch-1.7-orange.svg) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)


[Jiaxi Jiang](https://jiaxi-jiang.github.io/), [Kai Zhang](https://cszn.github.io/), [Radu Timofte](http://people.ee.ethz.ch/~timofter/)

*[Computer Vision Lab](https://vision.ee.ethz.ch/), ETH Zurich, Switzerland*


### **This is a fork version that walks through how to evaluate FBCNN locally on images**
________
:fire::fire: This repository is the official PyTorch implementation of paper "Towards Flexible Blind JPEG Artifacts Removal". [![paper download](https://img.shields.io/badge/arxiv-paper-red.svg)](https://arxiv.org/abs/2109.14573)
FBCNN achieves **state-of-the-art performance** in blind JPEG artifacts removal on
- Single JPEG images (color/grayscale)
- Double JPEG images (aligned/non-aligned)
- Real-world JPEG images

🚀🚀 Some Visual Examples (Click for full images)
----------
|[<img src="figs/v1_.png" width="400px"/>](https://imgsli.com/NzA3NTk) |[<img src="figs/v3_.png" width="400px"/>](https://imgsli.com/NzA3NjI)|
________

# Run FBCNN

Follow this YouTube [tutorial]() or if you have any questions feel free to join my [discord](https://discord.gg/sE8R7e45MV) and ask there.
 
## Setup Environment
We are going to use Anaconda3, download [Anaconda3](https://www.anaconda.com/products/individual) if you don't have it.  

- Create conda environment:
```
conda create -n FBCNN python=3.7
conda activate FBCNN
```
- Setup conda environment for nvidia non-30 series GPU:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
- Setup conda environment for nvidia 30 series GPU:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
- Setup conda env for non-nvidia GPU:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
- Setup the rest of the conda environment:

**clone**/download this repository, and `cd` into the base folder in Anaconda Prompt of the folder you just cloned/downloaded. Then enter the following commands:
```
pip install -r requirements.txt
```

- *To reuse the created conda environment after you close the prompt, you just need to*:
```
conda activate FBCNN
```

## Setup Files

Download the following files:
- [fbcnn_color.pth](https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth)
- [fbcnn_gray.pth](https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray.pth)
- [fbcnn_gray_double.pth](https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_gray_double.pth)

File Structure:
```
📂FBCNN-main/ # this is root
├── 📂model_zoo/
├── 📜fbcnn_color.pth
├── 📜fbcnn_gray.pth
├── 📜fbcnn_gray_double.pth
│   │...
│...
```


Testing
----------

- Grayscale JPEG images

To test this, put your input images under `testsets/Classic5`
```bash
python main_test_fbcnn_gray.py
```

- Grayscale JPEG images, trained with double JPEG degradation model

To test this, put your input images under `testsets/LIVE1_gray`
```bash
python main_test_fbcnn_gray_doublejpeg.py
```

- Color JPEG images

To test this, put your input images under `testsets/LIVE1_color`
```bash
python main_test_fbcnn_color.py
```

- Real-World JPEG images

To test this, put your input images under `testsets/Real`
```bash
python main_test_fbcnn_color_real.py
```

All results would be under `test_results`, check the according folders for what type of testing you've done.

### This is the end of my fork edit.

___________

Contents
----------

* [Motivations](#motivations)
* [Network Architecture](#network-architecture)
* [Analysis of Double JPEG Restoration](#analysis-of-double-jpeg-restoration)
* [Experiments](#experiments)
* [Citation](#citation)
* [License and Acknowledgement](#license-and-acknowledgement)

Motivations
----------
JPEG is one of the most widely-used image compression algorithms and formats due to its simplicity and fast encoding/decoding speeds. However, it is a lossy compression algorithm and can introduce annoying artifacts. Existing methods for JPEG artifacts removal generally have four limitations in real applications: 

- Most existing learning-based methods [e.g. ARCNN, MWCNN, SwinIR] trained a specific model for each quality factor, lacking the flexibility to learn a single model for different JPEG quality factors. 

- DCT-based methods [e.g. DMCNN, QGAC] need to obtain the DCT coefficients or quantization table as input, which is only stored in JPEG format. Besides, when images are compressed multiple times, only the most recent compression information is stored. 

- Existing blind methods [e.g. DnCNN, DCSC, QGAC] can only provide a deterministic reconstruction result for each input, ignoring the need for user preferences. 

- Existing methods are all trained with synthetic images which assumes that the low-quality images are compressed only once. _**However, most images from the Internet are compressed multiple times.**_ Despite some progress for real recompressed images, e.g. from Twitter [ARCNN, DCSC], a detailed and complete study on double JPEG artifacts removal is still missing. 


Network Architecture
----------

We propose a flexible blind convolutional neural network (FBCNN) that predicts the quality factor of a JPEG image and embed it into the decoder to guide image restoration. The quality factor can be manually adjusted for flexible JPEG restoration according to the user's preference.
![architecture](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/architecture.png)

Analysis of Double JPEG Restoration
----------
###  1. What is non-aligned double JPEG compression?

Non-aligned double JPEG compression means that the 8x8 blocks of two JPEG compression are not aligned. For example, when we crop a JPEG image and save it also as JPEG, it is highly possible we get a non-aligned double JPEG image. 
![real](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/doraemon.png)
There are many other common scenarios including, but not limited to:
- take a picture by smartphone and upload it online. Most social media platforms, e.g. Wechat, Twitter, Facebook, resize the uploaded images by downsampling and then apply JPEG compression to save storage space.
- Edit a JPEG image that introduces cropping, rotation, or resizing, and save it as JPEG.
- Zoom in/out a JPEG image, then take a screenshot, save it as JPEG.
- Group different JPEG image and save it as a single JPEG image.
- Most memes are compressed many times with non-aligned cases.

###  2. Limitation of existing blind methods on restoration of non-aligned double JPEG images

We find that existing blind methods always do not work when the 8x8 blocks of two JPEG compression are not aligned and QF1 <= QF2, _**even with just a one-pixel shift.**_ Other cases such as non-aligned double JPEG with QF1>QF2, or aligned double JPEG compression, are actually equivalent to single JPEG compression.

Here is an example of the restoration result of DnCNN and QGAC on a JPEG image with different degradation settings. '*' means there is a one-pixel shift between two JPEG blocks.
![lena_doublejpeg](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/lena_doublejpeg.png)


###  3. Our solutions
We find for non-aligned double JPEG images with QF1 < QF2, FBCNN always predicts the quality factor as QF2. However, it is the smaller QF1 that dominants the compression artifacts. By manually changing the predicted quality factor to QF1, we largely improve the result.

Besides, to get a fully blind model, we propose two blind solutions to solve this problem:

(1) FBCNN-D: Train a model with a single JPEG degradation model + automatic dominant QF correction. By utilizing the property of JPEG images, we find the quality factor of a single JPEG image can be predicted by applying another JPEG compression. When QF1 = QF2, the MSE of two JPEG images is minimal. In our paper, we also extend this method to non-aligned double JPEG cases to get a fully blind model.

(2) FBCNN-A: Augment training data with double JPEG degradation model, which is given by:

<p align="center">
    y = JPEG(shift(JPEG(x, QF1)),QF2)
</p>

By reducing the misalignment of training data and real-world JPEG images, FBCNN-A further improves the results on complex double JPEG restoration. _**This proposed double JPEG degradation model can be easily integrated into other image restoration tasks, such as single image super-resolution (e.g. [BSRGAN](https://github.com/cszn/BSRGAN)), for better general real image restoration.**_


Experiments
----------

### 1. Single JPEG restoration
![single_table](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/single_table.png)
*: Train a specific model for each quality factor.
![single_compare](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/single_compare.png)

### 2. Non-aligned double JPEG restoration
There is a pixel shift of (4,4) between the blocks of two JPEG compression.
![double_table](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/double_table.png)
![double_compare](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/double_compare.png)

### 3. Real-world JPEG restoration
![real](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/real.png)

### 4. Flexibility of FBCNN
By setting different quality factors, we can control the trade-off between artifacts removal and details preservation, according to the user's preference.
![flexible](https://github.com/jiaxi-jiang/FBCNN/blob/main/figs/flexible.png)


Citation
----------

    @inproceedings{jiang2021towards,
      title={Towards Flexible Blind JPEG Artifacts Removal},
      author={Jiang, Jiaxi and Zhang, Kai and Timofte, Radu},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={4997--5006},
      year={2021}
      }

License and Acknowledgement
----------
This project is released under the Apache 2.0 license. This work was partly supported by the ETH Zürich Fund (OK) and a Huawei Technologies Oy (Finland) project.
