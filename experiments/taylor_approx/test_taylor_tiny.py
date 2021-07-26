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


from taylor_expansion_approx import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Taylor Approx Test Function')
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
    kwargs = {"num_workers": 4, "pin_memory" : True}
else:
    kwargs = {}
transform_train = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
train_dir = os.environ["DATASET_DIR"] + "/tiny-imagenet-200/train"
test_dir  = os.environ["DATASET_DIR"] + "/tiny-imagenet-200/val"

trainloader = torch.utils.data.DataLoader(datasets.ImageFolder(train_dir, transform=transform_train),batch_size=128, shuffle=True, **kwargs)

testloader = torch.utils.data.DataLoader(datasets.ImageFolder(test_dir, transform=transform_test), batch_size=args.test_batch_size, shuffle=False, **kwargs)
# Model
print('==> Building model..')
num_classes=200
if args.model == "alexnet_bn":
    import alexnet_bn
    net = alexnet_bn.AlexNet(num_classes=200)
    net = net.to(device)
elif args.model == "vgg11_bn":
    import vgg11_bn
    net = vgg11_bn.VGG11(num_classes=200)
    net = net.to(device)
elif args.model == "vgg16_bn":
    import vgg16_bn
    net = vgg16_bn.VGG16(num_classes=200)
    net = net.to(device)
elif args.model == "resnet18_bn":
    import resnet18_bn
    net = resnet18_bn.ResNet18(num_classes=200)
    net = net.to(device)
elif args.model == "mobilenetv1_bn":
    import mobilenetv1 
    net = mobilenetv1.MobileNetV1(num_classes=200)
    net = net.to(device)
elif args.model == "resnet32_bn":
    import resnet32_bn
    net = resnet32_bn.ResNet32(num_classes=200)
    net = net.to(device)
else:
    print("error: model not recognized")
    exit()

net.change_all_activations(ReLUTaylorApprox())
net = nn.DataParallel(net).cuda()
net.load_state_dict(torch.load(args.ckpt))
print(net)

def test(args, net, device, testloader):
    net.eval()
    global best_acc
    sum_loss = 0
    correct = 0
    num_seen = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            num_seen += inputs.shape[0]
            outputs = net(inputs)[-1]
            sum_loss += nn.CrossEntropyLoss()(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    avg_loss = sum_loss / num_seen 
    acc = correct / num_seen 
    wandb.log({"Test Loss" : avg_loss,
               "Test Acc" : acc})


wandb.watch(net)
wandb.config.update(args)
test(args, net, device, testloader)
wandb.save("taylor_expansion_approx.py")
