import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os


class SSD(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 配置config
        self.cfg = (coco, voc)[num_classes == 21]
        # 初始化先验框
        self.priorbox = PriorBox(self.cfg)        # layers/functions/prior_box.py
        self.priors = Variable(self.priorbox.forward(), volatile=True)   # from torch.autograd import Variable
        self.size = size

        # SSD network
        # basebone 网络
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3  conv4_3后面的网络，L2正则化
        self.L2Norm = L2Norm(512, 20)  # 由于conv4_3与其他的输出相比波动大  # layers/modules/l2norm.py
        self.extras = nn.ModuleList(extras)

        # 回归和分类网络
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)    # 用于囧穿概率
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)  # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化
                                                                    # layers/functions/detection.py

    def forward(self, x):
        """# 定义forward函数, 将设计好的layers和ops应用到输入图片 x 上

        # 参数: x, 输入的batch 图片, Shape: [batch, 3, 300, 300]

        Return:
            Depending on phase:
        # test: 预测的类别标签, confidence score, 以及相关的location.
        #       Shape: [batch, topk, 7]
        # train: 关于以下输出的元素组成的列表
        #       1: confidence layers, Shape: [batch*num_priors, num_classes]
        #       2: localization layers, Shape: [batch, num_priors*4]
        #       3: priorbox layers, Shape: [2, num_priors*4]
        """
        sources = list()  # 存储的是参与预测的卷积层的输出，这里指的是6个
        loc = list()     # 存储预测的边框信息 ，列表中一个元素对应一个特征图
        conf = list()     # 预测类别信息

        # 前向传播至conv4_3 relu 得到第一个特征图
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)  # 将 conv4_3 的特征层输出添加到 sources 中, 后面会根据 sources 中的元素进行预测

        # 继续前向传播vgg到fc7得到第二个特征图
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)    # 得到的x尺度为[1,512,38,38]
        sources.append(x)

        # 在 extra layers 中前向传播得到其他四个特征图
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)                # 这里需要添加F.relu() 而vgg已经添加。得到的x尺寸为[1,1024,19,19]
            if k % 2 == 1:   # 在extras_layers中, 第1,3,5,7,9(从第0开始)的卷积层的
                                # 输出会用于预测box位置和类别, 因此, 将其添加到 sources列表中
                sources.append(x)
        # 上述sources中保存的数据如下：用于边框提取的特征图
        # torch.Size([1, 512, 38, 38])
        # torch.Size([1, 1024, 19, 19])
        # torch.Size([1, 512, 10, 10])
        # torch.Size([1, 256, 5, 5])
        # torch.Size([1, 256, 3, 3])
        # torch.Size([1, 256, 1, 1])
        # x = Variable(torch.randn(1, 3, 300, 300))

        # apply multibox head to source layers
        # 将各个特征图中的定位和分类结果append进列表中
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 $(N, H, W, C)$.
            # contiguous返回内存连续的tensor, 由于在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的,
            # 然后 view 操作是基于连续地址的, 因此, 需要调用contiguous语句.
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())   # 6（总共6层）*(N,C,H,W)->6*(N,H,W,C)  C=k*4
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())    # 6*(N,C,H,W)->6*(N,H,W,C)  C=k*num_class
            # loc: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
            # conf: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
        # cat 是 concatenate 的缩写, view返回一个新的tensor, 具有相同的数据但是不同的size, 类似于numpy的reshape
        # 在调用view之前, 需要先调用contiguous
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)   # [N,-1]  1表示按列拼接
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes（default boxes数量）*4]
        # 变成[batch，38*38*16+19*19*24+……+1*1*16]=[batch，8732 * 4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # 同理, 最终的shape为(两维):[batch, num_boxes*num_classes]
        if self.phase == "test":
            # 如果是测试阶段，就需要对定位和分类的结果进行分析得到最终的结果
            # detect对象, 该对象主要由于接预测出来的结果进行解析, 以获得方便可视化的边框坐标和类别编号
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds   ->[N,num_priors,4]
                # 意味着输出8732个先验框和每个先验框调整坐标
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds     [N,num_priors,num_classes] 最后一维softmax
                # 变成[batch, 8732, 21],对每一行进行softmax
                self.priors.type(type(x.data))                  # default boxes    [num_priors,4] 4:[cx,cy,w,h]
            )                                                   # output: [N,num_classes,num_remain*5]
        else:
            # 如果是训练阶段，就直接输出此时的定位和分类结果以计算损失函数
            output = (
                loc.view(loc.size(0), -1, 4),                        # [N,num_priors,4]
                conf.view(conf.size(0), -1, self.num_classes),       # [N,num_priors,num_classes]
                self.priors                                          # [num_priors,4]   num_priors表示的是ancuors的数量 8732
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i   # VGG第一层输入是RGB三通道的图像
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # 4-1 前面的需要输出38*38
        else:        # 如果遇到的不是M或C 那么就创建 多少入多少出的conv2d
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]    # 将普通的池化层变为池化加激活
            in_channels = v    # VGG 里面，输入输出通道除了第一层以外都是相等的
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 第五层的池化不进行下采样
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)  # 表示空洞卷积
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)   # 这里使用的卷积核为1
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers    # 35层，其中22和-2层需要输入到Multi-box Layers中

# 在vgg backbone后面加入了卷积网络用来后续的多尺度提取分析，要输入到multibox网络中
def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':  # 等于S的时候，将S后面的数值作为这一层卷积的输出
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers     # 总共8层，其中1，3，5，7层（index为0， 2...)要输入到Multi-box Layers

# 6层（4 + 2）多尺度提取的网络，每层分别对 loc 和 conf 进行卷积，得到相应的输出,
# 下面的cfg表示 每个特征图中各个像素定义的default boxes数量
# 这里生成 map * map * （先验框个数 * 4）的卷积shape
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]  # 21：conv4_3中最后一个3x3x512 -2：conv7relu之前的1x1卷积,剩下的是在添加的层里实现的
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        # 定义分类层, 和定位层差不多, 只不过输出的通道数不一样, 因为对于每一个像素点上的每一个default box,
        # 都需要预测出属于任意一个类的概率, 因此通道数为 default box 的数量乘以类别数.
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # 每个特征地图位置的盒子数量
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
