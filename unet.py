import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, features):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = features, kernel_size = 3, padding = 1, bias = True)
        self.conv2 = nn.Conv2d(in_channels = features, out_channels = features, kernel_size = 3, padding = 1, bias = True)
        self.batchNorm1 = nn.BatchNorm2d(features)
        self.batchNorm2 = nn.BatchNorm2d(features)
        self.act_f1 = nn.ReLU(inplace = True)
        self.act_f2 = nn.ReLU(inplace = True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.act_f1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.act_f2(x)

        return x

class Encoder(nn.Module):
    
    def __init__(self, in_channels = 3, init_features = 32, classify = False):
        super(Encoder, self).__init__()

        self.classify = classify

        features = init_features

        self.encoder1 = Block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = Block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3 = Block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4 = Block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = Block(features * 8, features * 16)
    
    def forward(self, x):
        if self.classify:
            x = self.encoder1(x)
            x = self.encoder2(self.pool1(x))
            x = self.encoder3(self.pool2(x))
            x = self.encoder4(self.pool3(x))

            x = self.bottleneck(self.pool4(x))

            return x
        else:
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            enc5 = self.bottleneck(self.pool4(enc4))

            return enc5, enc4, enc3, enc2, enc1

class Decoder(nn.Module):

    def __init__(self, out_channels = 3, init_features = 32):
        super(Decoder, self).__init__()

        features = init_features

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, enc5, enc4, enc3, enc2, enc1):
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

class UNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 3, init_features = 32):
        super(UNet, self).__init__()

        self.encoder = Encoder(in_channels, init_features)
        self.decoder = Decoder(out_channels, init_features)

    def forward(self, x):
        enc5, enc4, enc3, enc2, enc1 = self.encoder(x)
        x = self.decoder(enc5, enc4, enc3, enc2, enc1)

        return x

class Classifier(nn.Module):

    def __init__(self, in_channels = 3, init_features = 32, num_classes = 3):
        super(Classifier, self).__init__()

        self.encoder = Encoder(in_channels, init_features, True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace = True),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x