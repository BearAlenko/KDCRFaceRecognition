import torch
import torch.nn as nn
import torchvision

class AICNN(torch.nn.Module):
    def __init__(self):
      super(AICNN, self).__init__()
      self.f_extractor = torchvision.models.resnet50()
      self.f_extractor.fc = nn.Linear(in_features= 2048, out_features = 512, bias = True)
      self.age_feature_transform = nn.Linear(in_features=512, out_features = 512, bias = True)
      self.age_classify = nn.Linear(in_features=512, out_features=6, bias = True)
      #parameter = nn.ParameterList(f_extractor, age_feature_transform, age_classify)

    def forward(self, x):
      image = x
      raw_features = self.f_extractor(image)
      age_feature = self.age_feature_transform(raw_features)
      id_feature = raw_features - age_feature
      age_out = self.age_classify(age_feature)
      return id_feature, age_out

#model = InceptionResnetV1(pretrained='vggface2').eval()

# For a model pretrained on CASIA-Webface
#model = InceptionResnetV1(pretrained='casia-webface').eval()