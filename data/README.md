## data

### This directory will contain the datasets used for Sisyphus. Currently, we use:
1. MNIST
2. CIFAR-10
3. CIFAR-100
4. TinyImageNet

### Downloading MNIST and CIFAR:
``` python
#python
import torchvision
torchvision.datasets.MNIST("./", download=True)
torchvision.datasets.CIFAR10("./", download=True)
torchvision.datasets.CIFAR100("./", download=True)
```

### Download TinyImageNet:
``` bash
#bash
sh tiny.sh
```
