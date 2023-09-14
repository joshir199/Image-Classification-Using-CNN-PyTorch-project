# Image-Classification-Using-CNN-PyTorch-project
Creating End to end project flow for Image classification using CNN with PyTorch

This project trains on standard MNIST dataset using Convolutional Neural Network (CNN) model architecture.
It predicts the class which the digit shown as image in the input image files.

# Model summary

![](https://github.com/joshir199/Image-Classification-Using-CNN-PyTorch-project/blob/main/images/model_summary.png)

------------------------------------------------
# Training and Evaluating script:
```bash
usage: main.py [-h] [--seed S] [--outf OUTF] [--ckpf CKPF] [--degree P]
               [--batch-size N] [--train] [--evaluate]

CNN Model using PyTorch

optional arguments:
  -h, --help          show this help message and exit
  --seed S            random seed (default: 1)
  --outf OUTF         folder to output images and model checkpoints
  --ckpf CKPF         path to model checkpoint file (to continue training)
  --learning_rate lr  learning rate for training (default : 0.01)
  --train             training a CNN model
  --evaluate          Evaluate a [pre]trained model from a random tensor.
```
-------------------------------------
# Training
This project will download MNIST dataset from the standard dataset repo of PyTorch.
The dataset contains images of single digits from 0 and 9 and output label as digit. For example:

![](https://github.com/joshir199/Image-Classification-Using-CNN-PyTorch-project/blob/main/images/data_image_for_label_One.png)


Optimizer used in the project : SGD with learning rate as provided by user
loss function used here is CrossEntropy for multiclass classification.

To get proper understanding of model flow, please refer: https://github.com/joshir199/Image-Classification-Using-CNN-PyTorch-project/blob/main/output/model_summary_graph.py

Here's the commands to training, Please run the following commands by providing appropriate value for mentioned parameters.

full_path : full directory path to this folder where model weights will be saved after training
```bash
$ python main.py --train --seed 3 --learning_rate 0.05
```

-------------------------------
# Training Loss curve

![](https://github.com/joshir199/Image-Classification-Using-CNN-PyTorch-project/blob/main/images/loss_graph_CNN_Pytorch.png)
