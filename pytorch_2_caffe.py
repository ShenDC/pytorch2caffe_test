import torch
import torch.nn as nn
layer_name_dict = {}


def module_hook(module, input, output):
    input = []

    name = layer_name_dict[module]
    # try to trans pytorch to caffe or other freamwork model
   

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
