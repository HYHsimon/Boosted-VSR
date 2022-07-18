import torch
import torch.nn as nn

class class_loss_2class(nn.Module):
    #Class loss
    def __init__(self):
        super(class_loss_2class, self).__init__()

    def forward(self, type_res):

        loss = torch.mean(1 - abs(type_res[:, 0]-type_res[:, 1]))

        return loss


class average_loss_2class(nn.Module):
    #Average loss
    def __init__(self):
        super(average_loss_2class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0])
        type_all = type_res
        sum1 = 0
        sum2 = 0

        for i in range(n):
            sum1 += type_all[i][0]
            sum2 += type_all[i][1]

        return (abs(sum1-n/m) + abs(sum2-n/m)) / ((n/m)*4)

class class_loss_3class(nn.Module):
    #Class loss
    def __init__(self):
        super(class_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0]) - 1
        type_all = type_res
        loss = 0
        for i in range(n):
            sum_re = abs(type_all[i][0]-type_all[i][1]) + abs(type_all[i][0]-type_all[i][2]) + abs(type_all[i][1]-type_all[i][2])
            loss += (m - sum_re)
        return loss / n


class average_loss_3class(nn.Module):
    #Average loss
    def __init__(self):
        super(average_loss_3class, self).__init__()

    def forward(self, type_res):
        n = len(type_res)
        m = len(type_res[0])
        type_all = type_res
        sum1 = 0
        sum2 = 0
        sum3 = 0

        for i in range(n):
            sum1 += type_all[i][0]
            sum2 += type_all[i][1]
            sum3 += type_all[i][2]

        return (abs(sum1-n/m) + abs(sum2-n/m) + abs(sum3-n/m)) / ((n/m)*4)