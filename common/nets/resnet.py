import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
# from torchvision.models.resnet import model_urls
from torchvision import models

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type = 50):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, stage=None):
        if stage != 'late':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        if stage == 'early':
            return x
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def init_weights(self):
        # org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        # org_resnet.pop('fc.weight', None)
        # org_resnet.pop('fc.bias', None)
        
        # self.load_state_dict(org_resnet)
        print("Initialize resnet from model zoo")



class Resnet50Encoder(nn.Module):
    """
    Returns spatial features before global avgpool (output of layer4).
    If you want the pooled 2048-D vector, pass pool=True in forward.
    """
    def __init__(self, resume_path=None, pretrained=True, **kwargs):
        super().__init__()
        # New torchvision uses weights arg; fall back if older
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.feature_encoder = models.resnet50(weights=weights)
        except Exception:
            self.feature_encoder = models.resnet50(pretrained=pretrained)

        self.use_gesture_logits = kwargs.get('use_gesture_logits', False)
        num_main_classes = kwargs.get('num_main_classes', 0)
        num_sub_classes = kwargs.get('num_sub_classes', 0)
        
        # Remove final classifier; keep avgpool around in case you want pooled vector
        self.in_features = self.feature_encoder.fc.in_features
        self.feature_encoder.fc = nn.Identity()
        
        
                        
        if self.use_gesture_logits:
            if num_main_classes > 0:
                self.main_classifier = nn.Sequential(
                    nn.Linear(self.in_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, num_main_classes)
                )
            if num_sub_classes > 0:
                self.sub_classifier = nn.Sequential(
                    nn.Linear(self.in_features, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                nn.Linear(1024, num_sub_classes)
                )
            
            
        # Optionally load your old classifier checkpoint; we only care about backbone
        if resume_path is not None:
            ckpt = torch.load(resume_path, map_location="cpu")

            # Easiest: load into the whole module with strict=False.
            # This will load all matching keys under feature_encoder.* and ignore old heads.
            missing, unexpected = self.load_state_dict(ckpt, strict=False)
            # (optional) print what didn’t load
            if len(missing) or len(unexpected):
                print("[load_state_dict] missing:", missing)
                print("[load_state_dict] unexpected:", unexpected)


    def forward(self, x, pool=False):
        # Manually run backbone to get pre-avgpool maps
        x = self.feature_encoder.conv1(x)   # [B, 64, H/2, W/2]
        x = self.feature_encoder.bn1(x)
        x = self.feature_encoder.relu(x)
        x = self.feature_encoder.maxpool(x) # [B, 64, H/4, W/4]

        x = self.feature_encoder.layer1(x)  # [B, 256, H/4,  W/4]
        x = self.feature_encoder.layer2(x)  # [B, 512, H/8,  W/8]
        x = self.feature_encoder.layer3(x)  # [B, 1024, H/16, W/16]
        feat = self.feature_encoder.layer4(x)  # [B, 2048, H/32, W/32]  <-- pre-avgpool

        main_logits = None
        sub_logits = None
        if self.use_gesture_logits:
            # Return the 2048-D vector if you ever need it
            x = self.feature_encoder.avgpool(feat)   # [B, 2048, 1, 1]
            x = torch.flatten(x, 1)               # [B, 2048]
            main_logits = self.main_classifier(x)
            sub_logits = self.sub_classifier(x)
            return feat, main_logits, sub_logits
        else:
            return feat

    def init_weights(self):
        pass