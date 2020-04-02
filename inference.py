######################################################
#                    Master Thesis: 
#      Embedded Deep Learning for Apple detection
#       
#        Author: Felipe Verdes Wolukanis 302513
######################################################


import torch
import torchvision
import os


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from converter import convertToElu, convertToGelu, convertToLRelu, convertToPRelu, convertToRRelu


import utility.utils as utils
#import utility.transforms as T
import numpy as np
from PIL import Image
import torchvision.transforms as T



num_classes = 2
device = torch.device('cuda')
    

print("Loading model")

# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256

# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

model.eval()

img_path = '/home/felipevw/MyAppleDetector/inference/images/tree_apple.jpg'
img = Image.open(img_path)
img.show()


transform = T.Compose([T.ToTensor()])
img = transform(img)


pred = model([img])

















