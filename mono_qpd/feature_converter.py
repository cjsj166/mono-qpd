import torch
from torch import nn

class PixelShuffleConverter(nn.Module):
    def __init__(self):
        super(PixelShuffleConverter, self).__init__()
        # c_h, c_w = c_res

        self.basic_shuffle = nn.PixelShuffle(2)
        # c_h, c_w = (c_h + c_h % 2) / 2, (c_w + c_w % 2) / 2
        # c_h, c_w = c_h // 2 + c_h % 2, c_w // 2 + c_h % 2
        # c_h, c_w = c_h // 2 + c_h % 2, c_w // 2 + c_h % 2
        # self.adapool1 = nn.AdaptiveAvgPool2d((c_h, c_w)) # Biggest resolution
        # c_h, c_w = c_h // 2 + c_h % 2, c_w // 2 + c_h % 2
        # self.adapool2 = nn.AdaptiveAvgPool2d((c_h, c_w))
        # c_h, c_w = c_h // 2 + c_h % 2, c_w // 2 + c_h % 2
        # self.adapool3 = nn.AdaptiveAvgPool2d((c_h, c_w))

        # self.conv1 = nn.Conv2d(128, 128, 3, padding=1) # Biggest resolution
        # self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        # self.conv3 = nn.Conv2d(256, 128, 3, padding=1) 

        self.conv1 = nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)) # Biggest resolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2))
        )
        

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.basic_shuffle(x1)
        x2 = self.basic_shuffle(x2)
        x3 = self.basic_shuffle(x3)

        # x1 = self.adapool1(x1)
        # x2 = self.adapool2(x2)
        # x3 = self.adapool3(x3)

        patch_h, patch_w = x1.shape[2] // 2, x1.shape[3] // 2
        x1 = nn.functional.interpolate(x1, size=(7*patch_h, 7*patch_w), mode='bilinear', align_corners=False)
        x2 = nn.functional.interpolate(x2, size=(7*patch_h, 7*patch_w), mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(x3, size=(7*patch_h, 7*patch_w), mode='bilinear', align_corners=False)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x1 = self.relu1(x1)
        x2 = self.relu2(x2)
        x3 = self.relu3(x3)
        return [x1, x2, x3]

class ConvConverter(nn.Module):
    def __init__(self):
        super(ConvConverter, self).__init__()

        self.trans_conv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=7, stride=7)
        self.trans_conv2 = nn.ConvTranspose2d(1024, 1024, kernel_size=7, stride=7)
        self.trans_conv3 = nn.ConvTranspose2d(1024, 1024, kernel_size=7, stride=7)

        self.basic_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, stride=(1, 1)),
            nn.Conv2d(512, 256, 3, padding=1, stride=(1, 1))
        )

        self.basic_conv2 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, stride=(1, 1)),
            nn.Conv2d(512, 256, 3, padding=1, stride=(1, 1))
        )

        self.basic_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, stride=(1, 1)),
            nn.Conv2d(512, 256, 3, padding=1, stride=(1, 1))
        )

        self.relu1_1 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu3_1 = nn.ReLU()        

        self.conv1 = nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)) # Biggest resolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2)),
            nn.Conv2d(128, 128, 3, padding=(1,1), stride=(2,2))
        )        

        self.relu1_2 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu3_2 = nn.ReLU()

    def forward(self, x):
        x1, x2, x3 = x

        x1 = self.trans_conv1(x1)
        x2 = self.trans_conv2(x2)
        x3 = self.trans_conv3(x3)

        x1 = self.basic_conv1(x1)
        x2 = self.basic_conv2(x2)
        x3 = self.basic_conv3(x3)

        x1 = self.relu1_1(x1)
        x2 = self.relu2_1(x2)
        x3 = self.relu3_1(x3)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x1 = self.relu1_2(x1)
        x2 = self.relu2_2(x2)
        x3 = self.relu3_2(x3)

        return [x1, x2, x3]


class DecConverter(nn.Module):
    def __init__(self):
        super(DecConverter, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=(1,1), stride=(1,1))
        self.conv2 = nn.Conv2d(256, 128, 3, padding=(1,1), stride=(1,1))
        self.conv3 = nn.Conv2d(256, 128, 3, padding=(1,1), stride=(1,1))

    def forward(self, x):
        x1, x2, x3 = x

        x1_size = x1.shape[-2:]
        x2_size = x2.shape[-2:]
        x3_size = x3.shape[-2:]

        x1 = nn.functional.interpolate(x1, size=(x1_size[0]*7//8, x1_size[1]*7//8), mode='bilinear', align_corners=False)
        x2 = nn.functional.interpolate(x2, size=(x2_size[0]*7//8, x2_size[1]*7//8), mode='bilinear', align_corners=False)
        x3 = nn.functional.interpolate(x3, size=(x3_size[0]*7//8, x3_size[1]*7//8), mode='bilinear', align_corners=False)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        return [x1, x2, x3]

if __name__ == '__main__':
    # N = 456
    H, W = 200, 300
    H, W = 400, 500
    H, W = 456, 756

    def print_shape(x):
        for i in x:
            print(i.shape)

    fc = PixelShuffleConverter((H, W))

    x1 = torch.rand(1, 1024, 16, 16)
    x2 = torch.rand(1, 1024, 32, 32)
    x3 = torch.rand(1, 512, 16, 16)
    x = [x1, x2, x3]

    print_shape(x)

    x = fc(x)
    print('')

    print_shape(x)


