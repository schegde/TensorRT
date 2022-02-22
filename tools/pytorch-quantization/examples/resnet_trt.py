import torch
import torchvision
# import sys
# import os

# import collections
# sys.path.append("/objdet/vision/references/classification/")
# from train import evaluate, train_one_epoch, load_data

# data_path = "/objdet/imagenet/"
# batch_size = 32

# traindir = os.path.join(data_path, 'train')
# valdir = os.path.join(data_path, 'val')
# _args = collections.namedtuple('mock_args', ['model', 'distributed', 'cache_dataset', 'val_resize_size', 'val_crop_size', 'train_crop_size', 'interpolation', 'prototype'])
# dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, _args(model='resnet50', distributed=False, cache_dataset=False, val_resize_size=256, val_crop_size=224, train_crop_size=224, interpolation='bilinear', prototype=None))


# data_loader_test = torch.utils.data.DataLoader(
#     dataset_test, batch_size=batch_size,
#     sampler=test_sampler, num_workers=4, pin_memory=True)

# example_test = None
# for example in iter(data_loader_test):
#     example_test = example
#     break

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()
model = torchvision.models.resnet50(pretrained=False, progress=False)
model.eval()
model.cuda()

batch_size = 16
ONNX_FILE = "resnet50_quant.onnx"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load("/objdet/TensorRT/tools/pytorch-quantization/examples/quant_resnet50-calibrated.pth", map_location=device)
model.load_state_dict(state_dict)


dummy_input = torch.randn([16, 3, 224, 224], device=device)
torch.onnx.export(model, dummy_input, ONNX_FILE, verbose=True, opset_version=13, enable_onnx_checker=False)
