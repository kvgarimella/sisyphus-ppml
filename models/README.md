## models

### This directory contains the models used for Sisyphus. Currently, we use:
1. [MLP](https://d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html)
2. LeNet
3. AlexNet
4. VGG-[11,16]
5. ResNet-[18,32]
6. MobileNet-V1

### Docs:
Each network class contains the following functions:
1. `freeze_all()`: freeze all layers of the network
2. `unfreeze_layer(layer)`: unfreeze a specified layer of the network (0-based indexing)
3. `print_all()`: print all unfrozen layers of the network
4. `change_activation(layer, new_activation)`: change activation function of specified layer
5. `change_all_activations(new_activation)`: change all activation functions in the network
6. `get_l2_norm()`: return L2 norm of all weights in the network

Additionally, each network returns a list of tensors, rather than just the usual one tensor. This list contains the intermediate representation tensors after each activation layer as well as the final prediction.

### Example:
We will use the simple MLP model as an example. First, let's import the model:
```
#python
import torch
import mlp_bn
net = mlp_bn.MLPNet(num_classes=10)
```
Print the net using `print(net)`:
```
MLPNet(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linearL0_): Linear(in_features=784, out_features=256, bias=True)
  (bnL0_): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act0): ReLU()
  (linearL1_): Linear(in_features=256, out_features=10, bias=True)
)
```
Freeze all the parameters by running:
```
net.freeze_all()
```
Unfreeze layer 1:
```
net.unfreeze_layer(layer=1)
```
Change the activation function for a specified layer:
```
net.change_activation(layer=0, new_activation=torch.nn.SiLU())
>> self.act0 was previously: ReLU()
>> self.act0 is now        : SiLU()
```
Print the net again by calling `print(net)`:
```
MLPNet(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linearL0_): Linear(in_features=784, out_features=256, bias=True)
  (bnL0_): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (act0): SiLU()
  (linearL1_): Linear(in_features=256, out_features=10, bias=True)
)
```






