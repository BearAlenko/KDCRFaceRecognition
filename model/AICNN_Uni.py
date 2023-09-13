import torch
import torch.nn as nn
import torchvision

class AICNN_Uni(torch.nn.Module):
    def __init__(self):
      super(AICNN_Uni, self).__init__()
      self.f_extractor = torchvision.models.resnet50()
      self.f_extractor.fc = nn.Linear(in_features= 2048, out_features = 512, bias = True)
      #parameter = nn.ParameterList(f_extractor, age_feature_transform, age_classify)

    def forward(self, x):
      image = x
      raw_features = self.f_extractor(image)
      return raw_features, 0