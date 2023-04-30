import torch
import torch.nn as nn

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        # pre-layer conv
        self.convL0_      = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL0_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act0         = nn.ReLU()


        # Layer 1 - Block 1
        self.convL1_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL1_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act1         = nn.ReLU()

        self.convL2_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL2_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act2         = nn.ReLU()

        # Layer 1 - Block 2
        self.convL3_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL3_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act3         = nn.ReLU()

        self.convL4_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL4_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act4         = nn.ReLU()

        # Layer 1 - Block 3
        self.convL5_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL5_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act5         = nn.ReLU()

        self.convL6_      = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL6_        = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act6         = nn.ReLU()



        # Layer 2 - Block 1
        self.convL7_      = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL7_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act7         = nn.ReLU()

        self.convL8_      = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL8_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcutL8_  = nn. Sequential(
                            nn.Conv2d(16, 32, kernel_size=(1, 1), stride=(2, 2), bias=False),
                            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.act8         = nn.ReLU()

        # Layer 2 - Block 2
        self.convL9_      = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL9_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act9         = nn.ReLU()

        self.convL10_      = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL10_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act10         = nn.ReLU()

        # Layer 2 - Block 3 
        self.convL11_      = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL11_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act11         = nn.ReLU()

        self.convL12_      = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL12_        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act12         = nn.ReLU()



        # Layer 3 - Block 1
        self.convL13_      = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bnL13_        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act13         = nn.ReLU()

        self.convL14_     = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL14_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.shortcutL14_ = nn. Sequential(
                            nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(2, 2), bias=False),
                            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.act14        = nn.ReLU()


        # Layer 3 - Block 2
        self.convL15_     = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL15_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act15        = nn.ReLU()

        self.convL16_     = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL16_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act16        = nn.ReLU()

        # Layer 3 - Block 3
        self.convL17_     = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL17_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act17        = nn.ReLU()

        self.convL18_     = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bnL18_       = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.act18        = nn.ReLU()
        self.avgpoolL18_  = nn.AdaptiveAvgPool2d(output_size=(1,1))


        self.linearL19_   = nn.Linear(in_features=64, out_features=num_classes, bias=True)
        self.flatten      = nn.Flatten()

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
        for layer in range(19):
            setattr(self, "act" + str(layer), new_activation)

    def get_l2_norm(self):
        params = []
        for param in self.parameters():
            params.append(param.view(-1))
        params = torch.cat(params).detach()
        return torch.norm(params)

    def forward(self, x):
        out = []
        # pre-layers
        x = self.convL0_(x)
        x = self.bnL0_(x)
        x = self.act0(x)
        skip = x.clone()
        out.append(x.clone())
        
        # layer 1.1
        x = self.convL1_(x)
        x = self.bnL1_(x)
        x = self.act1(x)
        out.append(x.clone())

        x = self.convL2_(x)
        x = self.bnL2_(x)
        x = x + skip
        x = self.act2(x)
        skip = x.clone()
        out.append(x.clone())
        
        # layer 1.2
        x = self.convL3_(x)
        x = self.bnL3_(x)
        x = self.act3(x)
        out.append(x.clone())

        x = self.convL4_(x)
        x = self.bnL4_(x)
        x = x + skip
        x = self.act4(x)
        skip = x.clone()
        out.append(x.clone())

        # layer 1.3
        x = self.convL5_(x)
        x = self.bnL5_(x)
        x = self.act5(x)
        out.append(x.clone())

        x = self.convL6_(x)
        x = self.bnL6_(x)
        x = x + skip
        x = self.act6(x)
        skip = x.clone()
        out.append(x.clone())


        # layer 2.1
        x = self.convL7_(x)
        x = self.bnL7_(x)
        x = self.act7(x)
        out.append(x.clone())

        x = self.convL8_(x)
        x = self.bnL8_(x)
        x = x + self.shortcutL8_(skip)
        x = self.act8(x)
        skip = x.clone()
        out.append(x.clone())
        

        # layer 2.2
        x = self.convL9_(x)
        x = self.bnL9_(x)
        x = self.act9(x)
        out.append(x.clone())

        x = self.convL10_(x)
        x = self.bnL10_(x)
        x = x + skip
        x = self.act10(x)
        skip = x.clone()
        out.append(x.clone())


        # layer 2.3
        x = self.convL11_(x)
        x = self.bnL11_(x)
        x = self.act11(x)
        out.append(x.clone())

        x = self.convL12_(x)
        x = self.bnL12_(x)
        x = x + skip
        x = self.act12(x)
        skip = x.clone()
        out.append(x.clone())


        # layer 3.1
        x = self.convL13_(x)
        x = self.bnL13_(x)
        x = self.act13(x)
        out.append(x.clone())

        x = self.convL14_(x)
        x = self.bnL14_(x)
        x = x + self.shortcutL14_(skip)
        x = self.act14(x)
        skip = x.clone()
        out.append(x.clone())


        # layer 3.2
        x = self.convL15_(x)
        x = self.bnL15_(x)
        x = self.act15(x)
        out.append(x.clone())
        
        x = self.convL16_(x)
        x = self.bnL16_(x)
        x = x + skip
        x = self.act16(x)
        skip = x.clone()
        out.append(x.clone())

        # layer 3.3
        x = self.convL17_(x)
        x = self.bnL17_(x)
        x = self.act17(x)
        out.append(x.clone())
        
        x = self.convL18_(x)
        x = self.bnL18_(x)
        x = x + skip
        x = self.act18(x)
        out.append(x.clone())
        x = self.avgpoolL18_(x)

        x = self.flatten(x)

        x = self.linearL19_(x)
        out.append(x)
        return out

if __name__ == "__main__":
    net = ResNet20(num_classes=10)
    net.eval();
    torch.manual_seed(1)
    x   = torch.randn(2,3,32,32)
    out = net(x)
    print("len(out):", len(out))
    print("out[-1].shape:", out[-1].shape)
