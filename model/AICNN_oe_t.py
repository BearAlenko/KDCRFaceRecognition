import torch
import torch.nn as nn
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1

class AICNN_oe_t(torch.nn.Module):
    def __init__(self):
      super(AICNN_oe_t, self).__init__()
      resnet = InceptionResnetV1(num_classes = 8631, classify=True)
      resnet.load_state_dict(torch.load('/project/6003167/zzh2015/trained_model/inceptionvgg2.pt'))
      self.f_extractor = resnet
      self.f_extractor.logits = nn.Identity(512)
      self.id_feature_transform = nn.Linear(in_features=512, out_features = 512, bias = True)
      self.age_classify = nn.Linear(in_features=1, out_features=6, bias = True)
      #parameter = nn.ParameterList(f_extractor, age_feature_transform, age_classify)

    def forward(self, x):
      image = x
      raw_features = self.f_extractor(image)
      age_feature = torch.norm(input=raw_features, dim = 1, keepdim=True, p = 2.0)
      id_feature = nn.functional.normalize(raw_features, p=2.0, dim=1, eps=1e-12)
      id_feature = self.id_feature_transform(id_feature)
      age_out = self.age_classify(age_feature)
      return id_feature, age_out