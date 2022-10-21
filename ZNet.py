
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from modelsummary import summary

class ZNet_v2(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def __init__(self):
        super(ZNet_v2, self).__init__()
        self.down1 = conv_block(in_channels=3,out_channels=64,kernel_size=(7, 7),stride=(2, 2),padding=(3, 3),
        )
        self.down2a = mod_Inception_block(in_channels=64, out_1x1=20, red_3x3=16, out_3x3=24, red_5x5=10, out_5x5=20 ,red_7x7=10, out_7x7=20)
        self.down2b = Inception_block(in_channels=84, out_1x1=20, red_3x3=16, out_3x3=24, red_5x5=10, out_5x5=20 ,red_7x7=10, out_7x7=20)
        # self.down2b = Inception_block(in_channels=64, out_1x1=10, red_3x3=16, out_3x3=24, red_5x5=10, out_5x5=20, out_1x1pool=10)
        self.down2c = Inception_block(in_channels=84, out_1x1=40, red_3x3=16, out_3x3=48, red_5x5=10, out_5x5=40 ,red_7x7=10, out_7x7=40)

        self.down3a = mod_Inception_block(in_channels=168, out_1x1=41, red_3x3=36, out_3x3=57, red_5x5=15, out_5x5=30 ,red_7x7=15, out_7x7=30)
        self.down3d = Inception_block(in_channels=158, out_1x1=82, red_3x3=36, out_3x3=114, red_5x5=15, out_5x5=60 ,red_7x7=36, out_7x7=60)
        
        
        self.down4a = mod_Inception_block(in_channels=316, out_1x1=121, red_3x3=40, out_3x3=70, red_5x5=36, out_5x5=65 ,red_7x7=40, out_7x7=65)
        self.down4e = Inception_block(in_channels=321, out_1x1=242, red_3x3=40, out_3x3=140, red_5x5=36, out_5x5=130 ,red_7x7=36, out_7x7=130)

        self.neck_a = Inception_block(in_channels=642, out_1x1=242, red_3x3=40, out_3x3=140, red_5x5=36, out_5x5=130 ,red_7x7=36, out_7x7=130)

        self.up1 = upsample_block(in_channels=642, out_1x1=82, red_3x3=36, out_3x3=114, red_5x5=15, out_5x5=60 ,red_7x7=15, out_7x7=60)
        self.up2 = upsample_block(in_channels=632, out_1x1=40, red_3x3=16, out_3x3=48, red_5x5=10, out_5x5=40 ,red_7x7=10, out_7x7=40)
        self.up3 = upsample_block(in_channels=336, out_1x1=20, red_3x3=12, out_3x3=28, red_5x5=4, out_5x5=16 ,red_7x7=4, out_7x7=16)
        self.output = upsample_block(in_channels=144, out_1x1=2, red_3x3=8, out_3x3=2, red_5x5=8, out_5x5=2 ,red_7x7=8, out_7x7=2)

        self.apply(self.weight_init)

        
    def forward(self, m):
                            
        x0 = self.down1(m)  
        

        m = self.down2a(x0) 
        m = self.down2b(m)
        x1 = self.down2c(m)  

       
        m = self.down3a(x1) 
        x2 = self.down3d(m) 

        m = self.down4a(x2) 
        m = self.down4e(m) 


        m = self.neck_a(m)  

        m = self.up1(m)   
        m = torch.cat([m,x2], dim=1)
        m = self.up2(m)   
        m = torch.cat([m,x1], dim=1)
        m = self.up3(m)   
        m = torch.cat([m,x0], dim=1)
        m = self.output(m) 
        return m


class mod_Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5,red_7x7, out_7x7
    ):
        super(mod_Inception_block, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels, out_1x1, kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)), 
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),           
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )
        self.branch4 = nn.Sequential(
            depthwise_separable_conv7(in_channels,3,out_7x7),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        )                                                           

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
            ) 



class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5,red_7x7, out_7x7
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=(1, 1))

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),         
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
        )                                                                    

        self.branch4 = depthwise_separable_conv7(in_channels,3,out_7x7)                                                                   

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class depthwise_separable_conv7(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv7, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=7, padding=3, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(nout)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return self.relu(self.batchnorm(out))

class upsample_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5,red_7x7, out_7x7):
        super(upsample_block, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels, out_1x1, kernel_size=(1, 1)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        )

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=(1, 1)), 
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=(1, 1)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)            
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=(1, 1)),
            conv_block(red_5x5, out_5x5, kernel_size=(5, 5), padding=(2, 2)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.branch4 = nn.Sequential(
            conv_block(in_channels, red_7x7, kernel_size=(1, 1)),
            conv_block(red_7x7, out_7x7, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
                                                                    
    

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

    





if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)
    model = ZNet_v2()
    summary(model,x)
