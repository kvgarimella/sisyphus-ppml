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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 QuaIL Training')
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
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
])
trainset = torchvision.datasets.CIFAR100(
    root=os.environ["DATASET_DIR"], train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR100(
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

if args.model == "alexnet_bn":
    import alexnet_bn
    student = alexnet_bn.AlexNet(num_classes=100)
    student = student.to(device)
    teacher = alexnet_bn.AlexNet(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 8
elif args.model == "vgg11_bn":
    import vgg11_bn
    student = vgg11_bn.VGG11(num_classes=100)
    student = student.to(device)
    teacher = vgg11_bn.VGG11(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 9
elif args.model == "vgg16_bn":
    import vgg16_bn
    student = vgg16_bn.VGG16(num_classes=100)
    student = student.to(device)
    teacher = vgg16_bn.VGG16(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 14
elif args.model == "resnet18_bn":
    import resnet18_bn
    student = resnet18_bn.ResNet18(num_classes=100)
    student = student.to(device)
    teacher = resnet18_bn.ResNet18(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 18
elif args.model == "mobilenetv1_bn":
    import mobilenetv1
    student = mobilenetv1.MobileNetV1(num_classes=100)
    student = student.to(device)
    teacher = mobilenetv1.MobileNetV1(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 28
elif args.model == "resnet32_bn":
    import resnet32_bn
    student = resnet32_bn.ResNet32(num_classes=100)
    student = student.to(device)
    teacher = resnet32_bn.ResNet32(num_classes=100)
    teacher = teacher.to(device)
    num_layers = 32 
else:
    print("error: model not recognized")
    exit()

student.change_all_activations(Quad())
student.load_state_dict(torch.load(args.ckpt))
teacher.load_state_dict(torch.load(args.ckpt))


# Training
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
    if math.isnan(avg_loss):
        print("Exiting.... NaN encountered during training..")
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
    if math.isnan(avg_sum_losses/(batch_idx+1)):
        print("Exiting.... NaN encountered during training..")
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

os.system("mkdir -p quail_nets_cifar100")
torch.save(student.state_dict(), "quail_nets_cifar100/{}_stage1.pth".format(args.model))

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
torch.save(student.state_dict(), "quail_nets_cifar100/{}_stage2.pth".format(args.model))
