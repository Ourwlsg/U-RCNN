import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import yaml

import tensorwatch as tw
from easydict import EasyDict
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
from src.urcnn.backbone.myResnet import resnet34
from src.urcnn.rpn import AnchorGenerator
from src.urcnn.urcnn import URCNN
from src.utils.gradcam_utils import visualize_cam, Normalize
from src.utils.gradcam import GradCAM, GradCAMpp

import torch
from torch.autograd import Variable
import torch.nn as nn
from graphviz import Digraph
from torchsummary import summary


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot


if __name__ == '__main__':
    img_path = "/home/zhucc/U-RCNN/data/JPEGImages/9999.jpg"
    pil_img = PIL.Image.open(img_path)
    # plt.figure()
    # plt.imshow(pil_img)
    # plt.show()

    with open("cfgs/train.yml", 'r', encoding="utf-8") as f:
        args = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    norm_mean = [0.4976264, 0.45133978, 0.3993562]
    norm_std = [0.11552592, 0.10886826, 0.10727626]
    # normalizer = Normalize(mean=norm_mean, std=norm_std)
    # torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    # torch_img = F.upsample(torch_img, size=(512, 512), mode='bilinear', align_corners=False)
    # normed_torch_img = normalizer(torch_img)

    writer = SummaryWriter(log_dir=args.board_dir + f"/fold-{0}",
                           comment=f'FOLD-{0}/LR-{args.lr}_'
                                   f'BS-{args.batch_size}_'
                                   f'EPOCHS-{args.epochs}_'
                                   f'SIZE-{args.inputsize}')
    backbone = resnet34(in_channel=3, out_channel=1, pretrain=True)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    anchor_generator = AnchorGenerator(sizes=((4, 8, 16),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = URCNN(backbone,
                  num_classes=args.num_classes,
                  min_size=args.inputsize,
                  max_size=args.inputsize,
                  image_mean=norm_mean,
                  image_std=norm_std,
                  rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler,
                  mask_classes=2)
    model.eval()
    summary(model=model,input_size=(3,512,512),device="cpu")
    # graph_inputs = torch.randn(1, 3, args.inputsize, args.inputsize)
    # writer.add_graph(model, input_to_model=graph_inputs, verbose=True)

    # tw.draw_model(model, [1, 3, 512, 512])
    # x = Variable(torch.randn(1, 1, 512, 512))
    # y = model(x)
    # g = make_dot(y)
    # g.view()
    #
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))

model.cuda()

for index, child in enumerate(model.children()):
    print(index, child)

cam_dict = dict()

resnet_model_dict = dict(type='resnet_unet', arch=model, layer_name='backbone', input_size=(512, 512))
resnet_gradcam = GradCAM(resnet_model_dict, True)
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)


mask, _ = resnet_gradcam(normed_torch_img)
heatmap, result = visualize_cam(mask, torch_img)

mask_pp, _ = resnet_gradcampp(normed_torch_img)
heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)


images = torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0)


plt.figure()
plt.imshow(images)
plt.show()
