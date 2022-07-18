import torch
import torch.nn as nn
import torch
import torchvision.models as models
import time

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

vgg_model = models.vgg19(pretrained=True)
class FeatureExtractor(nn.Module):
    def __init__(self, model, feature_layer,device=torch.device('cpu')):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)]).cuda()
        for k, v in self.features.named_parameters():
            v.requires_grad = False
    def forward(self, x):
        output = self.features(x)
        return output


class Vgg19(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg19, self).__init__()
        self.net = models.vgg19(pretrained=True).features.eval().cuda()
        for k, v in self.net.named_parameters():
            v.requires_grad = False
    def forward(self, x):
        out = []
        for i in range(len(self.net)):
            x = self.net[i](x)
            if i in [1, 6, 11, 20, 29]:
                # print(self.net[i])
                out.append(x)
        return out
class Vgg19_29(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg19_29, self).__init__()
        # self.net = models.vgg19(pretrained=True).features.eval().cuda()
        self.features = nn.Sequential(*list(vgg_model.features.children())[:(30)]).cuda()
        for k, v in self.features.named_parameters():
            v.requires_grad = False
    def forward(self, x):
        out = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if i in [1, 6, 11, 20, 29]:
                # print(self.net[i])
                out.append(x)
        return out

class FeatureExtractor_VGG19(nn.Module):
    def __init__(self):
        super(FeatureExtractor_VGG19, self).__init__()
        # self.feature1= FeatureExtractor(vgg_model,0)
        # self.feature2 = FeatureExtractor(vgg_model, 5)
        # self.feature3 = FeatureExtractor(vgg_model, 10)
        # self.feature4 = FeatureExtractor(vgg_model, 19)
        # self.feature5 = FeatureExtractor(vgg_model, 28)
        self.feature= Vgg19()
        self.feature_vgg=Vgg19_29()

    def forward(self,x,y,z):
        length=5

        d=[0]*length
        d1 = [0] * length
        d2 = [0] * length
        starttime = time.perf_counter()
        # a1=self.feature1(x)
        # a2= self.feature2(x)
        # a3 = self.feature3(x)
        # a4 = self.feature4(x)
        # a5 = self.feature5(x)
        # ai=[a1,a2,a3,a4,a5]
        # b1 = self.feature1(y)
        # b2 = self.feature2(y)
        # b3 = self.feature3(y)
        # b4 = self.feature4(y)
        # b5 = self.feature5(y)
        # bi = [b1, b2, b3, b4, b5]
        # c1 = self.feature1(z)
        # c2 = self.feature2(z)
        # c3 = self.feature3(z)
        # c4 = self.feature4(z)
        # c5 = self.feature5(z)
        # ci = [c1, c2, c3, c4, c5]
        # endtime = time.perf_counter()
        # time1 = endtime-starttime
        # starttime = time.perf_counter()
        ai=self.feature(x)
        bi=self.feature(y)
        ci=self.feature(z)
        # ai = self.feature_vgg(x)
        # bi = self.feature_vgg(y)
        # ci = self.feature_vgg(z)
        endtime = time.perf_counter()
        time1 = endtime-starttime
        starttime = time.perf_counter()
        for i in range(5):
            d1[i]=torch.dist(ai[i], ci[i], p=1)
            d2[i] = torch.dist(bi[i], ci[i], p=1)
            d[i]=d2[i]/d1[i]
        output =1/32 *d[0]+ 1/16 *d[1] +1/8 *d[2] +1/4* d[3]+d[4]
        endtime = time.perf_counter()
        time2 = endtime - starttime
        return output,time1,time2

