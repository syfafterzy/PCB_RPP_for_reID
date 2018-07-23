# Part-based Convolutional Baseline for Person Retrieval and the Refined Part Pooling

Code for the paper [Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)](https://arxiv.org/pdf/1711.09349.pdf). 

**This code is ONLY** released for academic use.

## Preparation
<font face="Times New Roman" size=4>

**Prerequisite: Python 2.7 and Pytorch 0.3+**

1. Install [Pytorch](https://pytorch.org/)

2. Download dataset
	a. Market-1501 [BaiduYun](https://pan.baidu.com/s/1ntIi2Op?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)
	b. DukeMTMC-reID[BaiduYun](https://pan.baidu.com/share/init?surl=jS0XM7Var5nQGcbf9xUztw) (password:bhbh)
	c. Move them to ```~/datasets/Market-1501/(DukeMTMC-reID)```
</font>

## train PCB
<font face="Times New Roman" size=4>

```sh train_PCB.sh```
With Pytorch 0.4.0, we shall get about 93.0% rank-1 accuracy and 78.0% mAP on Market-1501.
</font>

## train RPP
<font face="Times New Roman" size=4>

```sh train_RPP.sh```
With Pytorch 0.4.0, we shall get about 93.5% rank-1 accuracy and 81.5% mAP on Market-1501.
</font>

## Citiaion
<font face="times new roman" size=4>

Please cite this paper in your publications if it helps your research:
</font>

```
@inproceedings{sun2018PCB,
  author    = {Yifan Sun and
               Liang Zheng and
               Yi Yang and
			   Qi Tian and
               Shengjin Wang},
  title     = {Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)},
  booktitle   = {ECCV},
  year      = {2018},
}
```
