import torch
import torch.nn as nn
import torchvision.models as models

class UNetXBD(nn.Module):
    def __init__(self, num_classes=3): 
        super(UNetXBD, self).__init__()
        
        # 1. Load Pretrained ResNet34
        resnet = models.resnet34(pretrained=True)
        
        # 2. Extract Encoder Layers (Exactly 3 layers as per paper)
        # The default resnet conv1 works perfectly for 3-channel RGB xBD images
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3 
        
        # 3. Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            # Final Upsample to restore to 1024x1024 original dimension
            nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2), 
            nn.Conv2d(16, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size() # [Batch, 3, 1024, 1024]
        
        # Encode
        e0 = self.encoder0(x) 
        e1 = self.encoder1(e0)         
        e2 = self.encoder2(e1)         
        e3 = self.encoder3(e2)         
        
        # Decode
        out = self.decoder(e3)
        
        # Interpolate to ensure exact output size matches input size
        out = nn.functional.interpolate(out, size=(input_size[2], input_size[3]), mode='bilinear', align_corners=False)
        return out
