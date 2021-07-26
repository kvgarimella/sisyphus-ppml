import torch
import torch.nn as nn
class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.convL0_     = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL0_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0        = nn.ReLU()
        self.maxpoolL0_  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.convL1_     = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL1_       = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1        = nn.ReLU()
        self.maxpoolL1_  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.convL2_     = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL2_       = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2        = nn.ReLU()
        
        
        self.convL3_     = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL3_       = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3        = nn.ReLU()
        self.maxpoolL3_  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.convL4_     = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL4_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act4        = nn.ReLU()
        
        
        self.convL5_     = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL5_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act5        = nn.ReLU()
        self.maxpoolL5_  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        
        self.convL6_     = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL6_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act6        = nn.ReLU()
        
        self.convL7_     = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bnL7_       = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act7        = nn.ReLU()
        self.maxpoolL7_  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        #self.avgpoolL7_  = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avgpoolL7_   = nn.AdaptiveAvgPool2d((1,1))
        
        self.linearL8_   = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        self.flatten     = nn.Flatten()

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
        for layer in range(8):
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

        x = self.convL2_(x)
        x = self.bnL2_(x)
        x = self.act2(x)
        out.append(x.clone())

        x = self.convL3_(x)
        x = self.bnL3_(x)
        x = self.act3(x)
        out.append(x.clone())
        x = self.maxpoolL3_(x)

        x = self.convL4_(x)
        x = self.bnL4_(x)
        x = self.act4(x)
        out.append(x.clone())

        x = self.convL5_(x)
        x = self.bnL5_(x)
        x = self.act5(x)
        out.append(x.clone())
        x = self.maxpoolL5_(x)

        x = self.convL6_(x)
        x = self.bnL6_(x)
        x = self.act6(x)
        out.append(x.clone())

        x = self.convL7_(x)
        x = self.bnL7_(x)
        x = self.act7(x)
        out.append(x.clone())
        x = self.maxpoolL7_(x)
        x = self.avgpoolL7_(x)

        x = self.flatten(x)

        x = self.linearL8_(x)
        out.append(x)
        return out 

if __name__ == "__main__":
    net = VGG11()
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,3,32,32)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)
