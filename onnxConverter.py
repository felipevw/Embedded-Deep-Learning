######################################################
#                    Master Thesis: 
#      Embedded Deep Learning for Apple detection
#       
#        Author: Felipe Verdes Wolukanis 302513
######################################################



import torch.onnx
import torchvision
import os


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from converter import convertToElu, convertToGelu, convertToLRelu, convertToPRelu, convertToRRelu
from data.apple_dataset import AppleDataset

import utility.utils as utils
import utility.transforms as T
import numpy as np
from PIL import Image



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_prediction(img_path, target, model, ):
        
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img, target)
    pred = model([img])
    
    return pred


def main(args):
    num_classes = 2
    device = args.device
    
    
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
    
    
    # Select model version
    if args.model == 'elu':
        print('Elu activation function selected.')
        model = convertToElu(model)
        
    elif args.model == 'lrelu':
        print('LRelu activation function selected.')
        model = convertToLRelu(model)
        
    elif args.model == 'prelu':
        print('PRelu activation function selected.')
        model = convertToPRelu(model)
    
    elif args.model == 'rrelu':
        print('RRelu activation function selected.')
        model = convertToRRelu(model)
        
    elif args.model == 'gelu':
        print('LRelu activation function selected.')
        model = convertToGelu(model)
                          
    else:
        print('Relu activation function selected.')    
    
    print(model)
    
    # Load model parameters and keep on CPU
    model.load_state_dict(torch.load(args.weight_dir + args.weight_file, map_location=device))
    model.eval()
    
    # Load model parameters and keep on CPU
    model.load_state_dict(torch.load(args.weight_dir + args.weight_file, map_location=device))
    model.eval()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection')
    
    parser.add_argument('--data_path', 
                        default='/home/felipevw/MyAppleDetector/inference/images/20150919_174151_image1.png', 
                        help='path to the data to predict on')    
    parser.add_argument('--output_file', 
                        default='/home/felipevw/MyAppleDetector/output_inference', 
                        help='path where to write the prediction outputs')    
    parser.add_argument('--weight_dir', 
                        default='/home/felipevw/MyAppleDetector/inference/weight_file', 
                        help='path to the weight file')    
    parser.add_argument('--weight_file', 
                        default='/model_24.pth', 
                        help='filename of the weight file')   
    parser.add_argument('--device', 
                        default='cuda', 
                        help='device to use. Either cpu or cuda')
    
    parser.add_argument('--model', default='lrelu', help='model')
    
    #model = parser.add_mutually_exclusive_group(required=True)
    
    
    args = parser.parse_args()
    print(args.model)
    assert(args.model in ['relu', 'elu', 'lrelu', 'prelu', 'rrelu', 'gelu'])
    
    main(args)
