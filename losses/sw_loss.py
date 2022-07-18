import torch
import torch.nn as nn
import torchvision.models as models
import os 

class Slicing(nn.Module):
    def __init__(self, device, num_slices, input_shape=None):
        super().__init__()
        # Number of directions
        self.num_slices = num_slices
        self.dim_slices = num_slices # input_shape[1] channel
        self.device = device
        self.update_slices()
    
    def update_slices(self):
        """ update random directions """

        # Generate random directions
        self.directions = torch.normal(mean=0.0, std=1.0, size=(self.num_slices, self.dim_slices))
        # Normalize directions
        norm = torch.reshape(torch.sqrt(torch.sum(torch.square(self.directions), dim=-1)), (self.num_slices, 1))
        self.directions = torch.div(self.directions, norm)
        if torch.cuda.is_available():
            self.directions = self.directions.to(self.device)
        self.flatten = nn.Flatten()
    
    def forward(self, input):
        """ implementation of figure 2 """
        tensor = torch.reshape(input, (input.shape[0], input.shape[1], -1))
        # project each pixel feature onto directions (batch dot product)
        # sliced = torch.bmm(self.directions, tensor.permute(0, 1, 2))
        sliced = torch.randn((input.shape[0], self.directions.shape[0], tensor.shape[-1]))
        if torch.cuda.is_available():
            sliced = sliced.to(self.device)
        for i in range(input.shape[0]):
            sliced[i] = torch.matmul(self.directions, tensor[i])
        # sort projections for each direction
        sliced, indices = torch.sort(sliced)

        return self.flatten(sliced)


class VGGLayers(nn.Module):
    """ create a vgg model that returns a list of intermediate output values """
    def __init__(self, layer_idx):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        vgg.eval()

        self.layer_idx = layer_idx
        self.features = nn.ModuleList(vgg).eval()
    
    def forward(self, x):
        results = list()
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in self.layer_idx:
                results.append(x)
        return results
    
class ExtractorVggModel(nn.Module):
    """ extract stats using a pretrained vgg and return slices vectors """
    def __init__(self, device, layer_idx, input_size=(3, 128, 128)):
        super().__init__()
        self.vgg = VGGLayers(layer_idx)
        self.device = device
        self.vgg.eval()
        if torch.cuda.is_available():
            self.vgg = self.vgg.to(device)

        data_ = torch.randn(1, input_size[0], input_size[1], input_size[2])
        if torch.cuda.is_available():
            data_ = data_.to(device)
        self.slicing_losses = [Slicing(self.device, num_slices=l.shape[1]) for i, l in enumerate(self.vgg(data_))]

        # device_ids = [i for i in range(torch.cuda.device_count())]
        # if len(device_ids)>1:
        #     self.vgg = nn.DataParallel(self.vgg, device_ids = device_ids)
    
    def update_slices(self):
        for slice_loss in self.slicing_losses:
            slice_loss.update_slices()

    def forward(self, inputs):
        outputs = self.vgg(inputs)
        outputs = [self.slicing_losses[i](output) for i, output in enumerate(outputs)]
        return outputs

class SWLoss(nn.Module):
    def __init__(self, device, input_size=(3, 128, 128), reduction='mean'):
        super().__init__()
        # VGG layers used for the loss
        # layers = ['block1_conv1',
        #         'block1_conv2',
        #         'block2_conv1',
        #         'block2_conv2',
        #         'block3_conv1', 
        #         'block3_conv2',
        #         'block3_conv3',
        #         'block3_conv4',
        #         'block4_conv1', 
        #         'block4_conv2',
        #         'block4_conv3',
        #         'block4_conv4',
        #         'block5_conv1',
        #         'block5_conv2'
        #             ]
        layer_idx = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30]
        self.extractor = ExtractorVggModel(device, layer_idx, input_size)
        self.loss = torch.nn.MSELoss(reduction=reduction)
    
    def forward(self, inputs, targets):
        slice_inputs = self.extractor(inputs)
        slice_targets = self.extractor(targets)

        losses = [self.loss(slice_[0], slice_[1]) for slice_ in zip(slice_inputs, slice_targets)]
        output = sum(losses)
        return output

    def __repr__(self):
        return "Sliced Wasserstein Loss"


if __name__ == "__main__":
    input = torch.randn((1, 3, 128, 128))
    # target = torch.randn((1, 3, 128, 128))
    target = input.clone()

    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()

    sw_loss = SWLoss()
    loss = sw_loss(input, target)
    print("loss={}".format(loss.item()))