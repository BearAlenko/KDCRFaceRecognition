import torch
import torch.nn as nn
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1

class AICNN_Uni_t(torch.nn.Module):
    def __init__(self):
      super(AICNN_Uni_t, self).__init__()
      resnet = InceptionResnetV1(num_classes = 8631, classify=True)
      resnet.load_state_dict(torch.load('/project/6003167/zzh2015/trained_model/inceptionvgg2.pt'))
      self.f_extractor = resnet
      self.f_extractor.logits = nn.Identity(512)
      #parameter = nn.ParameterList(f_extractor, age_feature_transform, age_classify)

    def forward(self, x):
      image = x
      raw_features = self.f_extractor(image)
      return raw_features, 0