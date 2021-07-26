import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='PyTorch MNIST QuaIL Training')
parser.add_argument("--train-batch-size", type=int, default=128, help="training batch size")
parser.add_argument("--test-batch-size", type=int, default=100, help="testing batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
parser.add_argument("--momentum", type=int, default=0.5, help="SGD momentum value")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="L2 weight decay")
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
elif args.model == "lenet_bn":
    import lenet_bn
    student = lenet_bn.LeNet()
    student = student.to(device)
    teacher = lenet_bn.LeNet()
    teacher = teacher.to(device)
    ckpt_name = "lenet_bn_baseline.pth"
    num_layers = 5
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
    wandb.log({"Train Loss" : avg_loss,
               "L2 Norm" : student.get_l2_norm().item(),
               "custom_step" : epoch})

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

def test(args, student, device, testloader, epoch):
    student.eval()
    global best_acc
    sum_loss = 0
    correct = 0
    num_seen = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = student(inputs)[-1]
            if not torch.isnan(outputs).any():
                sum_loss += nn.CrossEntropyLoss()(outputs, targets)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                num_seen+= inputs.shape[0]
    if num_seen:
        avg_loss = sum_loss / num_seen
        acc = correct / num_seen
    else:
        avg_loss = 0.0
        acc = 0.0
    wandb.log({"Test Loss" : avg_loss,
               "Test Acc" : acc, "custom_step" : epoch})


wandb.watch(student)
wandb.config.update(args)

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

os.system("mkdir -p quail_nets_mnist")
torch.save(student.state_dict(), "quail_nets_mnist/{}_stage1.pth".format(args.model))


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
torch.save(student.state_dict(), "quail_nets_mnist/{}_stage2.pth".format(args.model))
