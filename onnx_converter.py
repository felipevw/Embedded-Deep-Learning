#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:12:05 2020

@author: felipevw
"""

import torch
import torchvision
import torch.onnx



# load an instance segmentation model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

batch_size = 1    # just a random number

# Input to the model
x = torch.randn(batch_size, 3, 360, 360, requires_grad=True)
#model = model(x)



torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "fasterrcnn_opset11_folding.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  )
