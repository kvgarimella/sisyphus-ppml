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

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Baseline Training')
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
print('==> Building model..')
@torch.jit.script
def quad(x):
        return x*x
mapping = {"relu": nn.ReLU(),
           "gelu": nn.GELU(),
           "silu": nn.SiLU(),
           "quad": quad}

if args.model == "alexnet_bn":
    import alexnet_bn
    net = alexnet_bn.AlexNet(num_classes=100)
    net = net.to(device)
    ckpt_name = "alexnet_bn.pth"
elif args.model == "vgg11_bn":
    import vgg11_bn
    net = vgg11_bn.VGG11(num_classes=100)
    net = net.to(device)
    ckpt_name = "vgg11_bn.pth"
elif args.model == "vgg16_bn":
    import vgg16_bn
    net = vgg16_bn.VGG16(num_classes=100)
    net = net.to(device)
    ckpt_name = "vgg16_bn.pth"
elif args.model == "resnet18_bn":
    import resnet18_bn
    net = resnet18_bn.ResNet18(num_classes=100)
    net = net.to(device)
    ckpt_name = "resnet18_bn.pth"
elif args.model == "mobilenetv1_bn":
    import mobilenetv1 
    net = mobilenetv1.MobileNetV1(num_classes=100)
    net = net.to(device)
    ckpt_name = "mobilenetv1_bn.pth"
elif args.model == "resnet32_bn":
    import resnet32_bn
    net = resnet32_bn.ResNet32(num_classes=100)
    net = net.to(device)
    ckpt_name = "resnet32_bn.pth"
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

wandb.watch(net)
wandb.config.update(args)
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(args, net, device, trainloader, optimizer, epoch)
    test(args, net, device, testloader, epoch)
    wandb.log({"L2 Norm of Weights" : net.get_l2_norm(), "custom_step" : epoch})
    scheduler.step()
os.system("mkdir -p cifar100")
torch.save(net.state_dict(), "cifar100/" + ckpt_name)
wandb.save("cifar100/" + ckpt_name)
