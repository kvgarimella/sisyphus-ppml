import os
import math
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from approxminmax import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch MNIST QuaIL+AMM Training')
parser.add_argument("--train-batch-size", type=int, default=128, help="training batch size")
parser.add_argument("--test-batch-size", type=int, default=100, help="testing batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--momentum", type=int, default=0.5, help="SGD momentum value")
parser.add_argument("--weight-decay", type=float, default=0.00, help="L2 weight decay")
parser.add_argument("--nesterov", type=bool, default=True, help="nesterov")
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--log-interval", type=int, default=10, help="logging interval")
parser.add_argument("--activation", default="relu", help="activation function")
parser.add_argument("--project", default="project", help="wandb project")
parser.add_argument("--name", default="name", help="wandb name")
parser.add_argument("--model", default="lenet", help="nn model name")
parser.add_argument("--ckpt", default="./model.ckpt", help="trained checkpoint")
parser.add_argument("--approxmode", type=int, default=1, help="0: biggest, 1: average, 2: real max and min at test time")
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
    kwargs = {"num_workers": 4, "pin_memory" : True}
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
trainset = torchvision.datasets.MNIST(
    root=os.environ["DATASET_DIR"], train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

testset = torchvision.datasets.MNIST(
    root=os.environ["DATASET_DIR"], train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Model
print('==> Building model..')

class Quad(nn.Module):
  def __init__(self):
    super(Quad, self).__init__()

  def forward(self, x):
    return x*x
if args.model == "mlp_bn":
    import mlp_bn
    student = mlp_bn.MLPNet()
    student = student.to(device)
    teacher = mlp_bn.MLPNet()
    teacher = teacher.to(device)
    num_layers = 2
    channels = [(1,256)]
elif args.model == "lenet_bn":
    import lenet_bn
    student = lenet_bn.LeNet()
    student = student.to(device)
    teacher = lenet_bn.LeNet()
    teacher = teacher.to(device)
    ckpt_name = "lenet_bn_baseline.pth"
    num_layers = 5
    channels = [(2,6), (2,16), (1,120), (1,84)]
elif args.model == "resnet32_bn":
    import resnet32_bn
    student = resnet32_bn.ResNet32(num_classes=10)
    student = student.to(device)
    teacher = resnet32_bn.ResNet32(num_classes=10)
    teacher = teacher.to(device)
    num_layers = 32 
    channels = [(2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 16), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 32), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64), (2, 64)]
else:
    print("error: model not recognized")
    exit()

for i in range(num_layers-1):
    if channels[i][0] == 1:
        student.change_activation(i, nn.Sequential(ApproxMinMaxNorm1d(num_features=channels[i][1], mode=args.approxmode).to(device), Quad()))
    else:
        student.change_activation(i, nn.Sequential(ApproxMinMaxNorm2d(num_features=channels[i][1], mode=args.approxmode).to(device), Quad()))


student.load_state_dict(torch.load(args.ckpt), strict=False)
teacher.load_state_dict(torch.load(args.ckpt))

def train(args, student, teacher, device, trainloader, optimizer, epoch, layer):
    print('\nEpoch: %d' % epoch)
    student.train()
    teacher.eval()
    avg_sum_loss= 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        #optimizer.zero_grad()
        for param in student.parameters():
            param.grad = None

        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        student_outputs = student(inputs)

        loss = nn.MSELoss()(student_outputs[layer], teacher_outputs[layer])
        avg_sum_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = avg_sum_loss / len(trainset)
    wandb.log({"M.S. E. Train Loss" : avg_loss,
               "L2 Norm" : student.get_l2_norm().item(), 
               "custom_step" : epoch})
    if math.isnan(avg_loss) or math.isinf(avg_loss):
        print("Exiting.... NaN / Inf encountered during training..")
        exit()

def train_standard(args, student, teacher, device, trainloader, optimizer, epoch, layer):
    print('\nEpoch: %d' % epoch)
    student.train()
    teacher.eval()
    avg_sum_losses = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        #optimizer.zero_grad()
        for param in student.parameters():
            param.grad = None
        student_outputs = student(inputs) 
        loss = nn.CrossEntropyLoss()(student_outputs[-1], targets)
        avg_sum_losses += loss.item()
        loss.backward()
        optimizer.step()
    wandb_dict = {}
    wandb_dict["custom_step"] = epoch
    wandb_dict["C.E. Train Loss".format(layer)] = avg_sum_losses/ (batch_idx+1) 
    wandb_dict["L2 Norm"] = student.get_l2_norm().item()
    wandb.log(wandb_dict)
    if math.isnan(avg_sum_losses/(batch_idx+1)) or math.isinf(avg_sum_losses/(batch_idx+1)):
        print("Exiting.... NaN / Inf encountered during training..")
        exit()

def test(args, student, device, testloader, epoch):
    student.eval()
    correct  = 0
    num_seen = 0
    num_nans = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            num_examples = inputs.shape[0]
            inputs       = inputs.to(device)
            targets      = targets.to(device)
            outputs      = student(inputs)[-1]
            
            nan_idx       = torch.any(torch.isnan(outputs), dim=1)
            curr_num_nans = nan_idx.sum().item()

            num_seen += (num_examples - curr_num_nans)
            num_nans += curr_num_nans

            if curr_num_nans < num_examples:
                _, predicted = outputs[torch.logical_not(nan_idx)].max(1)
                correct      = correct + predicted.eq(targets[torch.logical_not(nan_idx)]).sum().item()

    if num_seen:
        acc = correct/num_seen
    else:
        acc = 0
    wandb.log({"Test Acc"    :acc ,"test_num_seen"  : num_seen, "test_num_nans" : num_nans, "custom_step" : epoch})



wandb.watch(student)
wandb.config.update(args)

## Mimic Stage
for layer in range(num_layers):
    student.freeze_all()
    student.unfreeze_layer(layer)
    student.print_all()
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)
    for epoch in range(0, args.epochs):
        train(args, student, teacher, device, trainloader, optimizer, epoch, layer)
        scheduler.step()
    if layer == (num_layers) - 1:
        print("Testing")
        test(args, student, device, testloader, epoch)
    del scheduler
    del optimizer

os.system("mkdir -p approx_quail_nets_mnist")
torch.save(student.state_dict(), "approx_quail_nets_mnist/{}_{}_stage1.pth".format(args.model, args.approxmode))

## Fine-tune Stage
student.freeze_all()
student.print_all()
for layer in range(num_layers-1,-1,-1):
    student.unfreeze_layer(layer)
    student.print_all()
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)

    for epoch in range(0, args.epochs):
        train_standard(args, student, teacher, device, trainloader, optimizer, epoch, layer)
        scheduler.step()
    del scheduler
    del optimizer

test(args, student, device, testloader, epoch)
torch.save(student.state_dict(), "approx_quail_nets_mnist/{}_{}_stage2.pth".format(args.model, args.approxmode))
wandb.save("approx_quail_nets_mnist/{}_{}_stage1.pth".format(args.model, args.approxmode))
wandb.save("approx_quail_nets_mnist/{}_{}_stage2.pth".format(args.model, args.approxmode))
