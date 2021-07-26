import os
import argparse

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound

from polynomial_regression_approx import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Poly Regression Test Function')
parser.add_argument("--train-batch-size", type=int, default=128, help="training batch size")
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
if args.model == "alexnet_bn":
    import alexnet_bn
    net = alexnet_bn.AlexNet(num_classes=100)
    net = net.to(device)
elif args.model == "vgg11_bn":
    import vgg11_bn
    net = vgg11_bn.VGG11(num_classes=100)
    net = net.to(device)
elif args.model == "vgg16_bn":
    import vgg16_bn
    net = vgg16_bn.VGG16(num_classes=100)
    net = net.to(device)
elif args.model == "resnet18_bn":
    import resnet18_bn
    net = resnet18_bn.ResNet18(num_classes=100)
    net = net.to(device)
elif args.model == "mobilenetv1_bn":
    import mobilenetv1 
    net = mobilenetv1.MobileNetV1(num_classes=100)
    net = net.to(device)
elif args.model == "resnet32_bn":
    import resnet32_bn
    net = resnet32_bn.ResNet32(num_classes=100)
    net = net.to(device)

else:
    print("error: model not recognized")
    exit()
net.load_state_dict(torch.load(args.ckpt))
print(net)

def get_model(train_x, train_y, state_dict=None, debug=False):
    gp  = SingleTaskGP(train_x, train_y).to(device)
    if debug:
        print("Prior hyperparams lengthscale & noise:        {}, {}".format(gp.covar_module.base_kernel.lengthscale.item(), gp.likelihood.noise.item()))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if state_dict is not None:
        gp.load_state_dict(state_dict) # speeds up fit
        if debug:
            print("After loading state_dict lengthscale & noise: {}, {}".format(gp.covar_module.base_kernel.lengthscale.item(), gp.likelihood.noise.item()))
    fit_gpytorch_model(mll) # performs the hyperparam fit
    if debug:
        print("Post hyperparams lengthscale & noise:         {}, {}".format(gp.covar_module.base_kernel.lengthscale.item(), gp.likelihood.noise.item()))
    return gp, mll

def get_train_acc(net, device, trainloader):
    net.eval()
    correct  = 0
    num_seen = 0
    num_nans = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            num_examples = inputs.shape[0]
            inputs       = inputs.to(device)
            targets      = targets.to(device)
            outputs      = net(inputs)[-1]
            
            nan_idx       = torch.any(torch.isnan(outputs), dim=1)
            curr_num_nans = nan_idx.sum().item()

            num_seen += (num_examples - curr_num_nans)
            num_nans += curr_num_nans

            if curr_num_nans < num_examples:
                _, predicted = outputs[torch.logical_not(nan_idx)].max(1)
                correct      = correct + predicted.eq(targets[torch.logical_not(nan_idx)]).sum().item()
    if num_seen:
        return correct / num_seen, num_nans, num_seen
    else:
        return 0, num_nans, num_seen

def get_test_acc(net, device, testloader):
    net.eval()
    correct  = 0
    num_seen = 0
    num_nans = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            num_examples = inputs.shape[0]
            inputs       = inputs.to(device)
            targets      = targets.to(device)
            outputs      = net(inputs)[-1]
            
            nan_idx       = torch.any(torch.isnan(outputs), dim=1)
            curr_num_nans = nan_idx.sum().item()

            num_seen += (num_examples - curr_num_nans)
            num_nans += curr_num_nans

            if curr_num_nans < num_examples:
                _, predicted = outputs[torch.logical_not(nan_idx)].max(1)
                correct      = correct + predicted.eq(targets[torch.logical_not(nan_idx)]).sum().item()
    if num_seen:
        return correct / num_seen, num_nans, num_seen
    else:
        return 0, num_nans, num_seen


wandb.watch(net)
wandb.config.update(args)

LOWER_BOUND_RANGE = 0.5
UPPER_BOUND_RANGE = 50.
LOWER_BOUND_ORDER = 1
UPPER_BOUND_ORDER = 9
NUM_BAYES_SEARCH  = 50 

BOUNDS = torch.tensor([[LOWER_BOUND_RANGE, LOWER_BOUND_ORDER], [UPPER_BOUND_RANGE, UPPER_BOUND_ORDER]], device=device)
NUM_RANDOM = 10 
train_x    = torch.FloatTensor(NUM_RANDOM, 1).uniform_(LOWER_BOUND_RANGE, UPPER_BOUND_RANGE)
train_obj  = torch.zeros_like(train_x)
train_x_2  = torch.FloatTensor(NUM_RANDOM, 1).uniform_(LOWER_BOUND_ORDER, UPPER_BOUND_ORDER)
train_x    = torch.cat([train_x, train_x_2], dim=1)
print(train_x, train_obj)
print(train_x.shape, train_obj.shape)

best_acc = 0.0
for i in range(NUM_RANDOM):
    curr_R     = train_x[i,0].item()
    curr_order = torch.round(train_x[i,1]).int().item()
    net.change_all_activations(ReLUPolyApprox(R=curr_R, order=curr_order))
    querried_train_acc, num_nans, num_seen = get_train_acc(net, device, trainloader)
    best_acc = max(best_acc, querried_train_acc)
    train_obj[i,:] = querried_train_acc
    wandb.log({"Train Best Acc" : best_acc, "Train Acc" : querried_train_acc,
               "Train Num Nans" : num_nans, "Train Num Seen" : num_seen})


train_x = train_x.to(device)
train_obj = train_obj.to(device)
model, mll = get_model(train_x, train_obj)
for i in range(NUM_BAYES_SEARCH):
    print("iter num {}...".format(i))
    UCB      = UpperConfidenceBound(model=model, beta=5.)
    new_point_analytic, acq_value_list = optimize_acqf(
                acq_function=UCB,
                bounds=BOUNDS,
                q=1,
                num_restarts=20,
                raw_samples=10000,
                options={},
                return_best_only=True,
                sequential=False
    )
    curr_R     = new_point_analytic[0,0].item()
    curr_order = torch.round(new_point_analytic[0,1]).int().item()
    net.change_all_activations(ReLUPolyApprox(R=curr_R, order=curr_order))
    querried_train_acc, num_nans, num_seen = get_train_acc(net, device, trainloader)
    best_acc = max(best_acc, querried_train_acc)
    obj = torch.zeros(1,1, device=device)
    obj[...] = querried_train_acc
    train_obj = torch.cat([train_obj, obj])
    train_x   = torch.cat([train_x, new_point_analytic])
    try:
        model, mll = get_model(train_x, train_obj, model.state_dict())
    except:
        print(train_obj)
        print(train_x)
        break
    wandb.log({"Train Best Acc" : best_acc, "Train Acc" : querried_train_acc,
               "Train Num Nans" : num_nans, "Train Num Seen" : num_seen})


print("Running best result on test data set")
best_result = train_x[torch.argmax(train_obj)]
best_R     = best_result[0].item()
best_order = torch.round(best_result[1]).int().item()
net.change_all_activations(ReLUPolyApprox(R=best_R, order=best_order))
querried_test_acc, num_nans, num_seen = get_test_acc(net, device, testloader)
wandb.log({"Test Acc" : querried_test_acc, "Test Num Nans" : num_nans, "Test Num Seen" : num_seen})
os.system("mkdir -p bayes_data_cifar100")
torch.save(train_x, "bayes_data_cifar100/xs_{}.pth".format(args.model))
torch.save(train_obj, "bayes_data_cifar100/ys_{}.pth".format(args.model))

