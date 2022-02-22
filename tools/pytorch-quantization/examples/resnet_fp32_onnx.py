import torch
import torchvision
model = torchvision.models.resnet50(pretrained=True, progress=False)
model.eval()
model.cuda()

ONNX_FILE = "resnet50_fp32.onnx"

batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy_input = torch.randn([16, 3, 224, 224], device=device)
torch.onnx.export(model, dummy_input, ONNX_FILE, verbose=True, opset_version=13, enable_onnx_checker=False)