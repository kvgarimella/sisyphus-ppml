import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets



device = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='PyTorch Testing Code')
parser.add_argument("--train-batch-size", type=int, default=128, help="training batch size")
parser.add_argument("--test-batch-size", type=int, default=128, help="testing batch size")
parser.add_argument("--model", default="lenet_bn", help="nn model name")
parser.add_argument("--dataset", default="mnist", help="dataset")
parser.add_argument("--ckpt", default="./model.ckpt", help="trained checkpoint")
parser.add_argument("--activation", default='relu', help='relu or quad')
args = parser.parse_args()

print("device:", device)
if device == "cuda":
    kwargs = {"num_workers": 1, "pin_memory" : True}
else:
    kwargs = {}


if args.dataset == "mnist":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(
        root=os.environ["DATASET_DIR"], train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = torchvision.datasets.MNIST(
        root=os.environ["DATASET_DIR"], train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes=10

elif args.dataset == "cifar10":

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR10(
        root=os.environ["DATASET_DIR"], train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(
        root=os.environ["DATASET_DIR"], train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes=10
elif args.dataset == "cifar100":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))])

    trainset = torchvision.datasets.CIFAR100(
        root=os.environ["DATASET_DIR"], train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR100(
        root=os.environ["DATASET_DIR"], train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes=100
elif args.dataset == "tiny-imagenet":
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dir = os.environ["DATASET_DIR"] + "/tiny-imagenet-200/train"
    test_dir  = os.environ["DATASET_DIR"] + "/tiny-imagenet-200/val"

    trainloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(train_dir, transform=transform),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(
        datasets.ImageFolder(test_dir, transform=transform), 
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    num_classes=200

if args.model == "mlp_bn":
    import mlp_bn
    net = mlp_bn.MLPNet()
    net = net.to(device)
elif args.model == "lenet_bn":
    import lenet_bn
    net = lenet_bn.LeNet()
    net = net.to(device)
elif args.model == "alexnet_bn":
    import alexnet_bn
    net = alexnet_bn.AlexNet(num_classes=num_classes)
    net = net.to(device)
elif args.model == "vgg11_bn":
    import vgg11_bn
    net = vgg11_bn.VGG11(num_classes=num_classes)
    net = net.to(device)
elif args.model == "vgg16_bn":
    import vgg16_bn
    net = vgg16_bn.VGG16(num_classes=num_classes)
    net = net.to(device)
elif args.model == "resnet18_bn":
    import resnet18_bn
    net = resnet18_bn.ResNet18(num_classes=num_classes)
    net = net.to(device)
elif args.model == "mobilenetv1_bn":
    import mobilenetv1 
    net = mobilenetv1.MobileNetV1(num_classes=num_classes)
    net = net.to(device)
elif args.model == "resnet32_bn":
    import resnet32_bn
    net = resnet32_bn.ResNet32(num_classes=num_classes)
    net = net.to(device)
else:
    print("error: model not recognized")
    exit()

if num_classes == 200:
    net = nn.DataParallel(net).cuda()

net.eval()
net.load_state_dict(torch.load(args.ckpt))
sum_loss = 0
correct = 0
num_seen = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)[-1]
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
print("avg_loss:", avg_loss.item())
print("acc     :", acc)
