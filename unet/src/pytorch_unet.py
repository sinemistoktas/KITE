import torch
import torch.nn as nn
import torch.nn.functional as F

#from gtda.images import HeightFiltration
#from gtda.homology import CubicalPersistence
from scipy.stats import wasserstein_distance_nd
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np
import time

    
class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
#         if weight is not None:
#             weight = torch.Tensor(weight)
#             self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C, H, W = predict.size()
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, -1) # (N, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        predict = predict[:,1,:] # (N, *)
        
#         ## convert target(N, 1, *) into one hot vector (N, C, *)
#         target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
#         target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target, dim=1)  # (N, *)
        union = torch.sum(predict + target, dim=1)  # (N, *)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, *)

#         if hasattr(self, 'weight'):
#             if self.weight.type() != predict.type():
#                 self.weight = self.weight.type_as(predict)
#                 dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = (1-dice_coef) # 1

        return dice_loss
    

class DiceLoss_TUnet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        N, H, W = score.size()
        target = target.float()
        
        predict = score.view(N, -1) # (N, *)
        target = target.view(N, -1) # (N, *)
        
        smooth = 1e-5
        intersection = torch.sum(predict * target, dim=1)  # (N, *)
        y_sum = torch.sum(target * target, dim=1)
        z_sum = torch.sum(predict * predict, dim=1)
        loss = (2 * intersection + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        
#         class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]

        return loss / self.n_classes
    
    

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class Decoder(nn.Module):
    def __init__(self, n_class, f_size):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dconv_up4 = double_conv(f_size*8 + f_size*16, f_size*8)
        self.dconv_up3 = double_conv(f_size*4 + f_size*8, f_size*4)
        self.dconv_up2 = double_conv(f_size*2 + f_size*4, f_size*2)
        self.dconv_up1 = double_conv(f_size + f_size*2, f_size)
        self.conv_last = nn.Conv2d(f_size, n_class, 1) 
        
    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        
        x = self.dconv_up4(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)   

        x = self.dconv_up2(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)         
        
        x = self.dconv_up1(x)       
        return self.conv_last(x)
        

class UNet(nn.Module):
    def __init__(self, n_class, f_size, task_no):
        super().__init__()
                
        self.reg_task = task_no
        self.dconv_down1 = double_conv(1, f_size)
        self.dconv_down2 = double_conv(f_size, f_size*2)
        self.dconv_down3 = double_conv(f_size*2, f_size*4)
        self.dconv_down4 = double_conv(f_size*4, f_size*8)  
        self.dconv_down5 = double_conv(f_size*8, f_size*16) 
        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.dconv_up4 = double_conv(f_size*8 + f_size*16, f_size*8)
        self.dconv_up3 = double_conv(f_size*4 + f_size*8, f_size*4)
        self.dconv_up2 = double_conv(f_size*2 + f_size*4, f_size*2)
        self.dconv_up1 = double_conv(f_size + f_size*2, f_size)
        self.conv_last = nn.Conv2d(f_size, n_class, 1)
        
        self.decoder_1 = Decoder(n_class, f_size)
        """
        if self.reg_task>1:
            self.decoder_2 = Decoder(1, f_size)   
        """
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3) 
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4) 
        
        x_bottleneck = self.dconv_down5(x)
        
        out1 = self.decoder_1(x_bottleneck, conv1, conv2, conv3, conv4) 
        out2 = None

        """
        if self.reg_task>1:
            out2 = self.decoder_2(x_bottleneck, conv1, conv2, conv3, conv4) 
        """

        return out1
