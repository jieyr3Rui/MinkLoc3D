from time import sleep
from turtle import forward
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from datasets.oxford import OxfordDataset
import matplotlib.pyplot as plt

def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor

class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'

    def __init__(self,
                inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=None,
                bn_momentum=0.1,
                D=3):
        super(BasicBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample
        self.relu = ME.MinkowskiReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = 'BN'


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = 'IN'


def get_block(norm_type,
                inplanes,
                planes,
                stride=1,
                dilation=1,
                downsample=None,
                bn_momentum=0.1,
                D=3
            ):
    if norm_type == 'BN':
        return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    elif norm_type == 'IN':
        return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
    else:
        raise ValueError(f'Type {norm_type}, not defined')


class ResUNet2(ME.MinkowskiNetwork):
    NORM_TYPE = None
    BLOCK_NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 32, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                in_channels=3,
                out_channels=32,
                bn_momentum=0.1,
                normalize_feature=None,
                conv1_kernel_size=None,
                D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        self.normalize_feature = normalize_feature
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv4_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv2_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2_tr = get_block(
            BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv1_tr = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D)

        # self.block1_tr = BasicBlockBN(TR_CHANNELS[1], TR_CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=True,
            dimension=D)

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)
        print("after conv1 ", out.F.size())

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)
        print("after conv2 ", out.F.size())

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)
        print("after conv3 ", out.F.size())

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)
        print("after conv4 ", out.F.size())

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        # 特征拼接
        out = ME.cat(out_s4_tr, out_s4)
        print("after conv4_tr ", out.F.size())
        

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)
        print("after conv3_tr ", out.F.size())

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)
        print("after conv2_tr ", out.F.size())

        out = self.conv1_tr(out)
        out = MEF.relu(out)
        print("after conv1_tr ", out.F.size())

        out = self.final(out)
        

        if self.normalize_feature:
            return ME.SparseTensor(
                out.F / torch.norm(out.F, p=2, dim=1, keepdim=True),
                coordinate_map_key=out.coordinate_map_key,
                coordinate_manager=out.coordinate_manager)
        else:
            return out


class ResUNetBN2(ResUNet2):
    NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]


class ResUNetBN2D(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
    NORM_TYPE = 'BN'
    CHANNELS = [None, 128, 128, 128, 256]
    TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
    NORM_TYPE = 'BN'
    BLOCK_NORM_TYPE = 'IN'


class MinkPyramidGeM(ME.MinkowskiNetwork):
    def __init__(self,
            in_channels=1,
            bn_momentum=0.1,
            D=3
        ):
        """
        data:     [stride, channel]
        network:  (..., ...)
        (cbr[k]): (convs[k], bns[k], resblocks[k])

        MinkLoc3d:
        [1, 1] -> (conv0, bn0) -> [1, 32] -> (cbr[0]) -> [2, 32] -> (cbr[1]) -> [4, 64] --> (cbr[2]) --> [8, 64]
                                                                                   |                        |
                                                                                  \|/                      \|/
                                                                              (1x1conv[1])             (1x1conv[2])
                                                                                   |                        |
                                                                                  \|/ (+)                  \|/
                                                                                [4, 256] <- (tcbr[2]) <- [8, 256]
                                                                                   |
                                                                                  \|/
                                                                               (GeM pool)
                                                                                   |
                                                                                  \|/
                                                                                 [256]
        MinkPyramid:
        [1, 1] -> (conv0, bn0) -> [1, 32] -> (cbr[0]) -> [2, 32] -> (cbr[1]) -> [4, 64] --> (cbr[2]) --> [8, 64]
                                                                                   |                        |
                                                                                  \|/                      \|/
                                                                              (1x1conv[1])             (1x1conv[2])
                                                                                   |                        |
                                                                                  \|/ (+)                  \|/
                                                                                [4, 256] <- (tcbr[2]) <- [8, 256]
                                                                                   |                        |
                                                                                  \|/                      \|/
                                                                               (GeM pool)               (GeM pool)
                                                                                   |                        |
                                                                                  \|/                      \|/
                                                                                 [256]                    [256]
        
        
        """
        ME.MinkowskiNetwork.__init__(self, D)

        # 1024×64, 256×128, 64×256, and 16×512,
        N = 4
        CHANNELS     = [32, 64, 128, 256, 512]
        TR_CHANNELS  = [32, 64, 128, 256, 512]
        KERNEL_SIZES = [3, 3, 3, 3]
        STRIDES      = [2, 2, 2, 2]
        conv0_kernel_size = 7

        self.relu = ME.MinkowskiReLU(inplace=True)

        # 第一层卷积区别对待
        self.conv0 = ME.MinkowskiConvolution(
                        in_channels=in_channels,
                        out_channels=CHANNELS[0],
                        kernel_size=conv0_kernel_size,
                        stride=1,
                        dilation=1,
                        bias=False,
                        dimension=D
                     )
        self.bn0 = ME.MinkowskiBatchNorm(CHANNELS[1], momentum=bn_momentum)


        self.convs  = nn.ModuleList()
        self.bns    = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()

        for i in range(N):
            # convs
            self.convs.append(
                ME.MinkowskiConvolution(
                    in_channels=CHANNELS[i],
                    out_channels=CHANNELS[i+1],
                    kernel_size=3,
                    stride=2,
                    dilation=1,
                    bias=False,
                    dimension=D
                )
            )
            # bns
            self.bns.append(
                ME.MinkowskiBatchNorm(
                    CHANNELS[i+1], 
                    momentum=bn_momentum
                ),
            )
            # res blocks
            self.blocks.append(
                BasicBlockBN(
                    inplanes=CHANNELS[i+1], 
                    planes=CHANNELS[i+1], 
                    stride=1, 
                    dilation=1, 
                    downsample=None, 
                    bn_momentum=0.1, 
                    D=3
                )
            )
            # pool
            self.pools.append(GeM())


        self.fmlp = nn.Sequential(
            nn.Linear(
                in_features = 
                    # CHANNELS[0] + CHANNELS[1] + CHANNELS[2] + CHANNELS[3] + 
                    TR_CHANNELS[0] + TR_CHANNELS[1] + TR_CHANNELS[2] + TR_CHANNELS[3], 
                out_features=1024, 
                bias=False
            ),
            # 这一层的norm必要，否则梯度爆炸
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(
                in_features  = 1024, 
                out_features = 512, 
                bias=False
            ),

            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(
                in_features=512, 
                out_features=256, 
                bias=False
            ),
            # 最后输出sigmoid实测有用
            nn.Sigmoid()
            # nn.BatchNorm1d(256),
        )

        # 权重初始化
        self.weight_initialization()
            

    def weight_initialization(self):
        """
        # 网络权重初始化
        """
        for m in self.modules():
            # 卷积
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            # 反卷积
            if isinstance(m, ME.MinkowskiConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
            # 归一化
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
        return


    def forward(self, batch):
        x = batch




        return 

class MinkPyramid(ME.MinkowskiNetwork):
    def __init__(self,
            in_channels=1,
            feature_size=256,
            bn_momentum=0.1,
            D=3
        ):
        ME.MinkowskiNetwork.__init__(self, D)

        # 1024×64, 256×128, 64×256, and 16×512,
        CHANNELS    = [32, 32, 64, 64]


        self.relu = ME.MinkowskiReLU()

        # 0层卷积 对1通道的输入卷积成32通道特征
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm0  = ME.MinkowskiBatchNorm(CHANNELS[0])

        # conv1
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm1  = ME.MinkowskiBatchNorm(CHANNELS[1])
        self.block1 = BasicBlockBN(
            inplanes=CHANNELS[1], 
            planes=CHANNELS[1], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )


        # conv2
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm2  = ME.MinkowskiBatchNorm(CHANNELS[2])
        self.block2 = BasicBlockBN(
            inplanes=CHANNELS[2], 
            planes=CHANNELS[2], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )

        self.conv2_1x1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=feature_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )

        # conv3
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm3  = ME.MinkowskiBatchNorm(CHANNELS[3])
        self.block3 = BasicBlockBN(
            inplanes=CHANNELS[3], 
            planes=CHANNELS[3], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )

        self.conv3_1x1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=feature_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )

        # 反卷积
        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        # pooling层
        self.pool = GeM()

        # 权重初始化，有用
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, batch):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        # x = batch
        xconv0 = self.conv0(x)
        xconv0 = self.norm0(xconv0)
        xconv0 = self.relu(xconv0)
        
        # print("xconv0 ", xconv0.F.size())

        xconv1 = self.conv1(xconv0)
        xconv1 = self.norm1(xconv1)
        xconv1 = self.relu(xconv1)
        xconv1 = self.block1(xconv1)
        # print("xconv1 ", xconv1.F.size())

        xconv2 = self.conv2(xconv1)
        xconv2 = self.norm2(xconv2)
        xconv2 = self.relu(xconv2)
        xconv2 = self.block2(xconv2)
        # print("xconv2 ", xconv2.F.size())


        xconv3 = self.conv3(xconv2)
        xconv3 = self.norm3(xconv3)
        xconv3 = self.relu(xconv3)
        xconv3 = self.block3(xconv3)
        # print("xconv3 ", xconv3.F.size())

        xconv3_1x1 = self.conv3_1x1(xconv3)

        xconv2_tr  = self.conv3_tr(xconv3_1x1)

        xconv2_1x1 = self.conv2_1x1(xconv2)

        feats = xconv2_tr + xconv2_1x1

        feats = self.pool(feats)

        return feats, None

class MinkPyramidDirect(ME.MinkowskiNetwork):
    def __init__(self,
            in_channels=1,
            feature_size=256,
            bn_momentum=0.1,
            D=3
        ):
        ME.MinkowskiNetwork.__init__(self, D)

        # 1024×64, 256×128, 64×256, and 16×512,
        CHANNELS    = [32, 32, 64, 64]


        self.relu = ME.MinkowskiReLU()

        # 0层卷积 对1通道的输入卷积成32通道特征
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=CHANNELS[0],
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D,
        )
        self.norm0  = ME.MinkowskiBatchNorm(CHANNELS[0])

        # conv1
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[0],
            out_channels=CHANNELS[1],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm1  = ME.MinkowskiBatchNorm(CHANNELS[1])
        self.block1 = BasicBlockBN(
            inplanes=CHANNELS[1], 
            planes=CHANNELS[1], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )


        # conv2
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm2  = ME.MinkowskiBatchNorm(CHANNELS[2])
        self.block2 = BasicBlockBN(
            inplanes=CHANNELS[2], 
            planes=CHANNELS[2], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )

        self.conv2_1x1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=feature_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )

        # conv3
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D
        )
        self.norm3  = ME.MinkowskiBatchNorm(CHANNELS[3])
        self.block3 = BasicBlockBN(
            inplanes=CHANNELS[3], 
            planes=CHANNELS[3], 
            stride=1, 
            dilation=1, 
            downsample=None, 
            bn_momentum=0.1, 
            D=3
        )

        self.conv3_1x1 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=feature_size,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=False,
            dimension=D
        )

        # 反卷积
        self.conv3_tr = ME.MinkowskiConvolutionTranspose(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=2,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)

        # pooling层
        self.pool0 = GeM()
        self.pool1 = GeM()
        self.pool2 = GeM()
        self.pool3 = GeM()

        self.mlp = nn.Sequential(
            nn.Linear(192, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        # 权重初始化，有用
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiConvolutionTranspose):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, batch):
        x = ME.SparseTensor(batch['features'], coordinates=batch['coords'])
        # x = batch
        xconv0 = self.conv0(x)
        xconv0 = self.norm0(xconv0)
        xconv0 = self.relu(xconv0)
        
        # print("xconv0 ", xconv0.F.size())

        xconv1 = self.conv1(xconv0)
        xconv1 = self.norm1(xconv1)
        xconv1 = self.relu(xconv1)
        xconv1 = self.block1(xconv1)
        # print("xconv1 ", xconv1.F.size())

        xconv2 = self.conv2(xconv1)
        xconv2 = self.norm2(xconv2)
        xconv2 = self.relu(xconv2)
        xconv2 = self.block2(xconv2)
        # print("xconv2 ", xconv2.F.size())


        xconv3 = self.conv3(xconv2)
        xconv3 = self.norm3(xconv3)
        xconv3 = self.relu(xconv3)
        xconv3 = self.block3(xconv3)
        # print("xconv3 ", xconv3.F.size())

        xconv0 = self.pool0(xconv0)
        xconv1 = self.pool0(xconv1)
        xconv2 = self.pool0(xconv2)
        xconv3 = self.pool0(xconv3)

        feats = torch.cat([xconv0, xconv1, xconv2, xconv3], dim=1)
        feats = self.mlp(feats)

        return feats, None



def testResUNet():

    oxford = OxfordDataset(
        dataset_path="/nas/slam/datasets/PointNetVLAD/DataMinkLoc3D",
        query_filename="training_queries_baseline.pickle",
        image_path=None,
        lidar2image_ndx=None,
        transform=None,
        set_transform=None,
        image_transform=None,
        use_cloud=True
    )

    
    
    bs, pn, d, fs = 16, 4096, 3, 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # quantization_size 相当于体素化
    coords = [ME.utils.sparse_quantize(coordinates=oxford[i][0], quantization_size=0.01)
                for i in range(bs)]

    feats  = [torch.rand((coords[b].size()[0], fs), dtype=torch.float32) for b in range(bs)] 

    # 生成SparseTensors
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)

    minknet_input = ME.SparseTensor(coordinates=coords, features=feats, device=device,  requires_grad=True)

    # 获取模型
    model = MinkPyramidDirect(in_channels=fs).to(device)

    minknet_output = model(minknet_input)

    
    # for idx in range(bs):
    #     coord1 = xconv1.decomposed_coordinates[idx].cpu().numpy().T
    #     coord2 = xconv2.decomposed_coordinates[idx].cpu().numpy().T
    #     coord3 = xconv3.decomposed_coordinates[idx].cpu().numpy().T
    #     coord4 = xconv4.decomposed_coordinates[idx].cpu().numpy().T

    #     fig = plt.figure(figsize=(20, 20))

    #     ax = plt.subplot(2,2, 1, projection = '3d')
    #     ax.scatter(coord1[0], coord1[1], coord1[2], color="blue", marker="o")
    #     plt.title("coord1: {}".format(coord1.shape[1]), fontsize=30)

    #     ax = plt.subplot(2,2, 2, projection = '3d')
    #     ax.scatter(coord2[0], coord2[1], coord2[2], color="red", marker="o")
    #     plt.title("coord2: {}".format(coord2.shape[1]), fontsize=30)

    #     ax = plt.subplot(2,2, 3, projection = '3d')
    #     ax.scatter(coord3[0], coord3[1], coord3[2], color="green", marker="o")
    #     plt.title("coord3: {}".format(coord3.shape[1]), fontsize=30)

    #     ax = plt.subplot(2,2, 4, projection = '3d')
    #     ax.scatter(coord4[0], coord4[1], coord4[2], color="orange", marker="o")
    #     plt.title("coord4: {}".format(coord4.shape[1]), fontsize=30)

    #     plt.savefig("/home/jieyr/code/MinkLoc3D/models/result.png")
    #     sleep(5)
    # print(minknet_output)
    return

if __name__ == "__main__":
    # 1024×64, 256×128, 64×256, and 16×512,
    testResUNet()