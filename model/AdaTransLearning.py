from .iresnet import *
from .losses import *
from .Attention import *
import torch
import torch.nn as nn
import torchvision
from facenet_pytorch import InceptionResnetV1


class ElasticFace_backbone(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_backbone, self).__init__()
      self.backbone = iresnet100()
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_state_dict(torch.load(backbone_path))
      print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False

class SoftFace_backbone(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(SoftFace_backbone, self).__init__()
      self.backbone = iresnet100()
      self.head = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, labels = None):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_state_dict(torch.load(backbone_path))
      print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
    
class SoftFace_backboneraw(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(SoftFace_backboneraw, self).__init__()
      self.backbone = iresnet100()
      self.head = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, labels = None):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      pass

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
    
class ElasticFaceD_block(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFaceD_block, self).__init__()
      self.backbone = iresnet100()
      self.age_trans = nn.Linear(in_features=512, out_features = 512)
      self.g = nn.Linear(in_features=512, out_features = 512)
      self.age_classify = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
      image = x
      id_features = self.backbone(image)
      age_features = self.age_trans(id_features)
      age_out = self.age_classify(age_features)
      id_features = id_features - self.g(age_features)
      id_features = nn.functional.normalize(id_features)
      return id_features, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_state_dict(torch.load(backbone_path))
      print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
    
class ElasticFaceD_backbone(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFaceD_backbone, self).__init__()
      self.backbone = ElasticFaceD_block(num_classes = num_classes, margin=margin, scale=scale)
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels=None):
      image = x
      id_features, age_out = self.backbone(image)
      if labels == None:
         return id_features, age_out
      id_class = self.head(id_features, labels)
      return id_features, id_class, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_init_backbone()
      print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      self.backbone.set_backbone_trainable(trainable=trainable)

    def check_trainable(self):
      flag = self.backbone.check_trainable()
      return flag
class AdaTransLearn_Elastic(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Elastic, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFace_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_state_dict(torch.load(backbone_path))
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False
    
class AdaTransLearn_Soft(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Soft, self).__init__()
      self.teacher = SoftFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = SoftFace_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_SoftFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_state_dict(torch.load(backbone_path))
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class AdaTransLearn_ElasticD(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_ElasticD, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFaceD_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s, age_out = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_init_backbone()
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False
    
class AdaTransLearn_ElasticD2(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_ElasticD2, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFaceD_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s, age_out = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      student_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFaceD_backbone_1.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_state_dict(torch.load(student_path))
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False
    
class AdaTransLearn_ElasticD3(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_ElasticD3, self).__init__()
      self.teacher = ElasticFaceD_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFaceD_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t, _ = self.teacher(image, labels)
      id_features_s, id_class_s, age_out = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFaceD_backbone_1.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_state_dict(torch.load(backbone_path))
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False
    
class ElasticFace_backbone_inception(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_backbone_inception, self).__init__()
      self.backbone = InceptionResnetV1(num_classes = num_classes, classify=False)
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class


class ElasticFace_backbone_ResNet100(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_backbone_ResNet100, self).__init__()
      self.backbone = iresnet100()
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      pass
      #self.backbone.load_state_dict(torch.load(backbone_path))
      #print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
class ElasticFace_backbone_ResNet50(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_backbone_ResNet50, self).__init__()
      self.backbone = iresnet50()
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      pass
      #self.backbone.load_state_dict(torch.load(backbone_path))
      #print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False

class ElasticFace_backbone_ResNet34(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_backbone_ResNet34, self).__init__()
      self.backbone = iresnet34()
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      pass
      #self.backbone.load_state_dict(torch.load(backbone_path))
      #print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False

class SoftFace_backbone_ResNet50raw(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(SoftFace_backbone_ResNet50raw, self).__init__()
      self.backbone = iresnet50()
      self.head = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x, labels):
      image = x
      id_features = self.backbone(image)
      id_features = nn.functional.normalize(id_features)
      id_class = self.head(id_features)
      return id_features, id_class

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      pass

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
class AdaTransLearn_Elastic_raw(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Elastic_raw, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFace_backbone_inception(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class AdaTransLearn_Elastic_raw50(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Elastic_raw50, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFace_backbone_ResNet50(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class AdaTransLearn_Elastic_raw34(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Elastic_raw34, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFace_backbone_ResNet34(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class AdaTransLearn_Soft_raw50(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Soft_raw50, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=0.35, scale=scale)
      self.student = SoftFace_backbone_ResNet50raw(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class AdaTransLearn_Elastic_raw50FT(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_Elastic_raw50FT, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = ElasticFace_backbone_ResNet50(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_FT64.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False

class age_elastichead(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(age_elastichead, self).__init__()
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)
      self.id_feature_transform = nn.Linear(in_features=512, out_features = 512, bias = True)
      self.age_classify = nn.Linear(in_features=1, out_features=7, bias = True)


    def forward(self, embedding, labels=None):
      age_features = torch.norm(input=embedding, dim = 1, keepdim=True, p = 2.0)
      id_features = nn.functional.normalize(embedding)
      
      id_features = self.id_feature_transform(id_features)
      age_out = self.age_classify(age_features)
      if labels == None:
         return id_features, age_out, None
      id_class = self.head(id_features, labels)
      return id_features, id_class, age_out

class ElasticFace_OE(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(ElasticFace_OE, self).__init__()
      self.backbone = iresnet100()
      self.head = age_elastichead(num_classes, margin=margin, scale=scale)


    def forward(self, x, labels=None):
      image = x
      id_features = self.backbone(image)
      id_features, id_class, age_out = self.head(id_features, labels)
      return id_features, id_class, age_out

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_state_dict(torch.load(backbone_path))
      print('pretrained weights loading finished')
    
    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False

class AttentionFace_backbone(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AttentionFace_backbone, self).__init__()
      self.backbone = iresnet100()
      self.fsm = AttentionModule()
      self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(512 * (112 // 16) ** 2, 512),
            nn.BatchNorm1d(512))
      self.age_trans = AgeEstimationModule(input_size=112, age_group=7)
      self.head = ElasticArcFace(in_features=512, out_features=num_classes, s=scale, m=margin)

    def forward(self, x, labels):
      image = x
      image = self.backbone.conv1(image)
      image = self.backbone.bn1(image)
      image = self.backbone.prelu(image)
      image = self.backbone.layer1(image)
      image = self.backbone.layer2(image)
      image = self.backbone.layer3(image)
      image = self.backbone.layer4(image)
      id_features, age_features = self.fsm(image)
      id_features = self.output_layer(id_features)
      _, age_group = self.age_trans(age_features)
      id_class = self.head(id_features, labels)
      return id_features, id_class, age_group

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/295672backbone.pth'):
      self.backbone.load_state_dict(torch.load(backbone_path))
      print('pretrained weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.backbone.parameters():
          param.requires_grad = trainable

    def check_trainable(self):
      for param in self.backbone.parameters():
          if param.requires_grad:
            return True
      return False
    
class AdaTransLearn_ElasticAttention(torch.nn.Module):
    def __init__(self, num_classes, margin=0.1, scale=64):
      super(AdaTransLearn_ElasticAttention, self).__init__()
      self.teacher = ElasticFace_backbone(num_classes, margin=margin, scale=scale)
      self.student = AttentionFace_backbone(num_classes, margin=margin, scale=scale)

    def forward(self, x, labels):
      image = x
      id_features_t, id_class_t = self.teacher(image, labels)
      id_features_s, id_class_s, age_group = self.student(image, labels)
      return id_features_s, id_features_t, id_class_s, id_class_t, age_group

    def load_init_backbone(self, backbone_path='/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1.pt'):
      backbone_path = '/project/6003167/zzh2015/trained_model/saved_ElasticFace_backbone_1_Ori.pt'
      self.teacher.load_state_dict(torch.load(backbone_path))
      print('pretrained teacher weights loading finished')
      self.student.load_init_backbone()
      print('pretrained student weights loading finished')

    def set_backbone_trainable(self, trainable=True):
      for param in self.teacher.parameters():
          param.requires_grad = trainable
    
    def check_trainable(self):
      for param in self.teacher.parameters():
          if param.requires_grad:
            return True
      return False