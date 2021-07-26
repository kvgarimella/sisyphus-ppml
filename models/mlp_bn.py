import torch
import torch.nn as nn

class MLPNet(nn.Module):

    def __init__(self, num_classes=10):
        super(MLPNet, self).__init__()
        self.flatten   = nn.Flatten()
        self.linearL0_ = nn.Linear(in_features=784, out_features=256, bias=True) 
        self.bnL0_     = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0      = nn.ReLU()

        self.linearL1_ = nn.Linear(in_features=256, out_features=num_classes, bias=True)

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
        for layer in range(1):
            setattr(self, "act" + str(layer), new_activation)

    def get_l2_norm(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params).detach()
        return torch.norm(params)

    def forward(self, x):
        out = []

        x = self.flatten(x)

        x = self.linearL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)
        out.append(x.clone())

        x = self.linearL1_(x)
        out.append(x)
        return out

if __name__ == "__main__":
    net = MLPNet()
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,1,28,28)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)
