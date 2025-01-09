from torch import nn

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = nn.Conv2d(inp_dim, int(out_dim/2),kernel_size=3,stride=1,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = nn.Conv2d(int(out_dim/2), int(out_dim/2),kernel_size=3,stride=1,padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = nn.Conv2d(int(out_dim/2), out_dim,kernel_size=1,stride=1,padding=0,bias=True)
        self.skip_layer = nn.Conv2d(inp_dim, out_dim,kernel_size=1,stride=1,padding=0,bias=True)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 
   
class Hourglass(nn.Module):
    def __init__(self, channel=3):
        super(Hourglass, self).__init__()
        self.Pool = nn.MaxPool2d(2, 2)
        self.Usample = nn.Upsample(scale_factor=2, mode='nearest')
        self.first_layer = nn.Sequential(nn.Conv2d(channel, 16,kernel_size=3,stride=1,padding=1,bias=True),nn.BatchNorm2d(16))
        # stack1
        self.up1 = Residual(16, 16)
        self.low1 = Residual(16, 32)
        
        # stack2
        self.up2 = Residual(32, 32)
        self.low2 = Residual(32, 64)

        # stack3
        self.up3 = Residual(64, 64)
        self.low3 = Residual(64, 128)

        # stack4
        self.up4 = Residual(128, 128)
        self.low4 = Residual(128, 256)

        self.middle = nn.Sequential(Residual(256, 256), Residual(256, 128), Residual(128, 128), Residual(128, 128))       
        
        self.decoder2 = Residual(128, 64)
        self.decoder3 = Residual(64, 32)
        self.decoder4 = Residual(32, 16)
        self.decoder5 = nn.Sequential(nn.Conv2d(16, 1,kernel_size=1,stride=1,padding=0,bias=True),nn.BatchNorm2d(1))
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        x = self.first_layer(x)

        up1  = self.up1(x)#16/1
        pool1 = self.Pool(x)#16/2
        low1 = self.low1(pool1)#32/2

        up2 = self.up2(low1)#32/2
        pool2 = self.Pool(low1)#32/4
        low2 = self.low2(pool2)#64/4
        
        up3 = self.up3(low2)#64/4
        pool3 = self.Pool(low2)#64/8
        low3 = self.low3(pool3)#128/8

        up4 = self.up4(low3)#128/8
        pool4 = self.Pool(low3)#128/16
        low4 = self.low4(pool4)#256/16
        
        high_feature = self.middle(low4)#128/16
        
        feature1 = up4 + self.Usample(high_feature) #128/8      
        out1 = self.decoder2(feature1)#64/8
        feature2 = up3 + self.Usample(out1)#64/4
        out2 = self.decoder3(feature2)#32/4
        feature3 = up2 + self.Usample(out2)#32/2
        out3 = self.decoder4(feature3)#16/2
        feature4 = up1 + self.Usample(out3)#16/1
        final_out = self.decoder5(rgb_3)#1/1
        rgb_prediction = self.sigmoid (final_out)     
        return feature1, feature2, feature3, feature4, rgb_prediction#128,64,32,16,1

     


