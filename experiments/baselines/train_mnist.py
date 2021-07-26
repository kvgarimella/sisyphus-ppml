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


parser = argparse.ArgumentParser(description='PyTorch MNIST Baseline Training')
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
@torch.jit.script
def quad(x):
        return x*x 
mapping = {"relu": nn.ReLU(),
           "gelu": nn.GELU(),
           "silu": nn.SiLU(),
           "quad": quad}

if args.model == "mlp_bn":
    import mlp_bn
    net = mlp_bn.MLPNet()
    net = net.to(device)
    ckpt_name = "mlp_bn.pth"
elif args.model == "lenet_bn":
    import lenet_bn
    net = lenet_bn.LeNet()
    net = net.to(device)
    ckpt_name = "lenet_bn.pth"
else:
    print("error: model not recognized")
    exit()


optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, 
                      weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# Training
def train(args, net, device, trainloader, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    avg_sum_loss= 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        #optimizer.zero_grad()
        for param in net.parameters():
            param.grad = None

        outputs = net(inputs)[-1]

        loss = nn.CrossEntropyLoss()(outputs, targets)
        avg_sum_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
    avg_loss = avg_sum_loss / len(trainset)
    acc = correct / len(trainset)
    wandb.log({"Train Loss" : avg_loss,
               "Train Acc"  : acc, "custom_step" : epoch})



def test(args, net, device, testloader, epoch):
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
               "Test Acc" : acc, "custom_step" : epoch})

def get_l2_weights(net):
    params = []
    for param in net.parameters():
        params.append(param.view(-1))
    params = torch.cat(params).detach()
    return torch.linalg.vector_norm(params)


wandb.watch(net)
wandb.config.update(args)
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(args, net, device, trainloader, optimizer, epoch)
    test(args, net, device, testloader, epoch)
    l2 = get_l2_weights(net)
    wandb.log({"L2 Norm of Weights" : l2, "custom_step" : epoch})
    scheduler.step()
os.system("mkdir -p mnist")
torch.save(net.state_dict(), "mnist/" + ckpt_name)
wandb.save("mnist/" + ckpt_name)
