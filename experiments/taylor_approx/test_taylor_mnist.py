import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from taylor_expansion_approx import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='PyTorch MNIST Taylor Approx Test Function')
parser.add_argument("--test-batch-size", type=int, default=100, help="testing batch size")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--project", default="project", help="wandb project")
parser.add_argument("--name", default="name", help="wandb name")
parser.add_argument("--model", default="lenet", help="nn model name")
parser.add_argument("--ckpt", default="./model.ckpt", help="trained checkpoint")
args = parser.parse_args()
torch.manual_seed(args.seed)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


wandb.init(project=args.project, name=args.name)
print("device:", device)
torch.backends.cudnn.benchmarks = True

# Data
print('==> Preparing data..')

if device == "cuda":
    kwargs = {"num_workers": 1, "pin_memory" : True}
else:
    kwargs = {}
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
if device == "cuda":
    kwargs = {"num_workers": 4, "pin_memory" : True}
else:
    kwargs = {}

testset = torchvision.datasets.MNIST(
    root=os.environ["DATASET_DIR"], train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

# Model
print('==> Building model..')

if args.model == "mlp_bn":
    import mlp_bn
    net = mlp_bn.MLPNet()
    net = net.to(device)
elif args.model == "lenet_bn":
    import lenet_bn
    net = lenet_bn.LeNet()
    net = net.to(device)
else:
    print("error: model not recognized")
    exit()
net.change_all_activations(ReLUTaylorApprox())
net.load_state_dict(torch.load(args.ckpt))
print(net)


def test(args, net, device, testloader):
    net.eval()
    global best_acc
    sum_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)[-1]
            sum_loss += nn.CrossEntropyLoss()(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    avg_loss = sum_loss / len(testset)
    acc = correct / len(testset)
    wandb.log({"Test Loss" : avg_loss,
               "Test Acc" : acc})


wandb.watch(net)
wandb.config.update(args)
test(args, net, device, testloader)
wandb.save("taylor_expansion_approx.py")
