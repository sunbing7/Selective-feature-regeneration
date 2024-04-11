# Imports
import sys
import caffe
import pickle
import numpy as np
from PIL import Image
import glob
import json
import argparse

#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=ResNet152 --test_len=100
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=VGG16
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=GoogLeNet
#python test_sfr.py --input_path=/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/ --dnn=ResNet152 --defense='_FRU'

parser = argparse.ArgumentParser()
parser.add_argument('--input_path',default="/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/",
                    help='path to the input images')
parser.add_argument('--dnn', default='ResNet152', choices=['CaffeNet', 'VGG_F', "GoogLeNet", "VGG16","ResNet152"],
                    help='DNN arch to be used')
parser.add_argument('--defense', default='', help='set to _FRU if apply defense')
parser.add_argument('--test_len', default=2000)
args = parser.parse_args()

img_crop = 224
label_path = "imagenet_labels.json"
with open(label_path) as f:
    label_dict = json.load(f)
model_def = 'Prototxt/' + args.dnn + '/deploy_' + args.dnn.lower() + args.defense + '.prototxt'
pretrained_model = 'Prototxt/' + args.dnn + '/' + args.dnn.lower() + args.defense + '.caffemodel'

#dataset
index_test = np.load(args.input_path + '/validation/index_test.npy').astype(np.int64)[:100]

# Create a net object
net = caffe.Net(model_def, pretrained_model, caffe.TEST)

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(1,         # batch size
                          3,         # 3-channel (BGR) images
                          img_crop, img_crop)  # image size is 227x227

correct = 0
i = 0
total = 0
for f in glob.iglob(args.input_path + "validation/val/*"):
    if i not in index_test:
        i = i + 1
        continue
    i = i + 1

    print('Processing image: {} {}'.format(i, f))
    img = Image.open(f).convert('RGB')
    w, h = img.size

    # image resizing for VGG16, VGG_F and ResNet152 maintains the original aspect ratio of the image
    #'''
    if w >= h:
        img = img.resize((256 * w // h, 256))
    else:
        img = img.resize((256, 256 * h // w))
    #'''
    #print('[DEBUG] ori image size: {}'.format(img.size))
    img = np.transpose(np.asarray(img), (2, 0, 1))
    img = img[[2, 1, 0], :, :]
    img = img.astype(np.float32)
    img = img[:, (img.shape[1] - img_crop) // 2:(img.shape[1] + img_crop) // 2,
               (img.shape[2] - img_crop) // 2:(img.shape[2] + img_crop) // 2]

    img = img[np.newaxis,:]

    # Mean subtraction values for ResNet152v2 model are different than the other 4 models provided
    if args.dnn=='ResNet152':
        img[:, 0, :, :] -= 102.98
        img[:, 1, :, :] -= 115.947
        img[:, 2, :, :] -= 122.772
    else:
        img[:, 0, :, :] -= 103.939
        img[:, 1, :, :] -= 116.779
        img[:, 2, :, :] -= 123.68

    net.blobs['data'].reshape(*img.shape)
    net.blobs['data'].data[...] = img

    net.forward()

    pred = net.blobs['prob'].data[0]
    pred_ids = np.argsort(pred)[-5:][::-1]

    print("Top 1 prediction: ",  label_dict[str(pred_ids[0])][1], ", Confidence score: ", str(np.max(pred)))
    print("Top 5 predictions: ", [label_dict[str(pred_ids[k])][1] for k in range(5)])

    correct = correct + (label_dict[str(pred_ids[0])][0] in f)
    total = total + 1




print('Clean sample top 1 accuracy: {}%'.format(correct / total * 100.))