from torch import nn


class Light_depth(nn.Module):
    def __init__(self, channel):
        super(Light_depth, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(channel,16,kernel_size=7,stride =1,padding=3,bias=True), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=5,stride =2,padding=2,bias=True), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,stride =2,padding=1,bias=True), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride =2,padding=1,bias=True), nn.ReLU())
        self.transform = nn.Conv2d(128,128,kernel_size=1,stride =1,padding=0,bias=True)
        
    def forward(self, d_in):
        # feature descriptor on the global spatial information
        out1 = self.conv1(d_in)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out = self.transform(out4)
        
        return  out, out3, out2, out1
