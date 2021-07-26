import torch
import torch.nn as nn
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.convL0_    = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.bnL0_      = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0       = nn.ReLU()
        self.maxpoolL0_ = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.convL1_    = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.bnL1_      = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1       = nn.ReLU()
        self.maxpoolL1_ = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.linearL2_  = nn.Linear(in_features=256, out_features=120, bias=True)
        self.bnL2_      = nn.BatchNorm1d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2       = nn.ReLU()

        self.linearL3_  = nn.Linear(in_features=120, out_features=84, bias=True)
        self.bnL3_      = nn.BatchNorm1d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3       = nn.ReLU()

        self.linearL4_  = nn.Linear(in_features=84, out_features=num_classes, bias=True)
        self.flatten    = nn.Flatten()

    def freeze_all(self):
        for name, param in self.named_parameters():
            param.requires_grad = False

    def print_all(self):
        print("debugging..")
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print(name)

    def unfreeze_layer(self, layer):
        print("Unfreezing..", layer)
        for name, param in self.named_parameters():
            if ("L" + str(layer) +"_" in name):
                param.requires_grad = True

    def change_activation(self, layer, new_activation):
        print("self.act{} was previously: {}".format(layer, getattr(self, "act" + str(layer))))
        setattr(self, "act" + str(layer), new_activation)
        print("self.act{} is now        : {}".format(layer, getattr(self, "act" + str(layer))))

    def change_all_activations(self, new_activation):
        for layer in range(4):
            setattr(self, "act" + str(layer), new_activation)

    def get_l2_norm(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params).detach()
        return torch.norm(params)

    def forward(self, x):
        out = []

        x = self.convL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)
        out.append(x.clone())
        x = self.maxpoolL0_(x)

        x = self.convL1_(x)
        x = self.bnL1_(x)
        x = self.act1(x)
        out.append(x.clone())
        x = self.maxpoolL1_(x)

        x = self.flatten(x)

        x = self.linearL2_(x)
        x = self.bnL2_(x)
        x = self.act2(x)
        out.append(x.clone())

        x = self.linearL3_(x)
        x = self.bnL3_(x)
        x = self.act3(x)
        out.append(x.clone())

        x = self.linearL4_(x)
        out.append(x)
        return out


if __name__ == "__main__":
    net = LeNet()
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,1,28,28)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)
