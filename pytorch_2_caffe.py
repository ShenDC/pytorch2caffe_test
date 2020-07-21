import torch
import torch.nn as nn
layer_name_dict = {}


def module_hook(module, input, output):
    input = []

    name = layer_name_dict[module]
    # try to trans pytorch to caffe or other freamwork model
    if isinstance(module, nn.Conv2d):
        out = Conv(input[0], module.kernel_size, module.out_channels,
                   module.stride, module.padding, group_size=module.groups, name=name)
    elif isinstance(module, nn.ConvTranspose2d):
        out = Conv(input[0], module.kernel_size, module.out_channels,
                   module.stride, module.padding, group_size=module.groups, name=name, transpose=True)
    elif isinstance(module, nn.BatchNorm2d):
        out = Norm(input[0], 'batch_norm', name=name)
    elif isinstance(module, nn.Linear):
        out = fc(input[0], module.out_features, name=name)
    elif isinstance(module, nn.MaxPool2d):
        out = pool(input[0], module.kernel_size, module.stride, module.padding,
                   name=name, pool_type='max')
    elif isinstance(module, nn.AvgPool2d):
        out = pool(input[0], module.kernel_size, module.stride, module.padding,
                   name=name, pool_type='avg')
    elif isinstance(module, nn.ReLU):
        out = Activation(input[0], 'relu', name=name)
    elif isinstance(module, nn.LeakyReLU):
        out = Activation(input[0], 'relu', name=name)
    elif isinstance(module, nn.Sigmoid):
        out = Activation(input[0], 'sigmoid', name=name)
    elif isinstance(module, nn.Conv3d):
        out = Conv(input[0], module.kernel_size, module.out_channels,
                   module.stride, module.padding, group_size=module.groups, name=name)

    if out:
        pass
    else:
        print('WARNING: skip Module {}'.format(module))

def trans(net, input):

    for name,layer in net.named_modules():
        layer_name_dict[layer] = name

    net.apply(net.register_forward_hook(module_hook))
    net.forward(input)


if __name__ == "__main__":
    input = torch.rand(1,3,416,416)
    net = yolov3_tiny()
    a,b = trans(net, input)
    print('done')
