import torch
import torch.nn as nn
import MinkowskiEngine as ME

class PointWiseAttention(ME.MinkowskiNetwork):
    def __init__(self, in_channel=256, out_channel=1, dimension=3):
        """
        # 点级别的Attention网络\n
        输入每个点卷积得到的特征\n
        输出每个点的分数 [0,1]\n
        """
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(in_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(512, 128, bias=False),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(128, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            # [0,1] 的权重
            # ME.MinkowskiReLU(),
            ME.MinkowskiSigmoid(),
        )

    def forward(self, pcfeats: ME.TensorField):
        x = self.conv1(pcfeats)
        x = self.conv2(x)
        weight = self.conv3(x)
        return weight


class PointFeatWiseAttention(ME.MinkowskiNetwork):
    def __init__(self, in_channel=256, out_channel=256, dimension=3):
        """
        # 每个点的特征级别的Attention网络\n
        输入每个点卷积得到的特征\n
        输出每个特征的分数 [0,1]\n
        """
        ME.MinkowskiNetwork.__init__(self, dimension)
        self.conv1 = nn.Sequential(
            ME.MinkowskiLinear(in_channel, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiLinear(512, 512, bias=False),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiLinear(512, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            # [0,1] 的权重
            # ME.MinkowskiReLU(),
            ME.MinkowskiSigmoid(),
        )

    def forward(self, pcfeats: ME.TensorField):
        x = self.conv1(pcfeats)
        x = self.conv2(x)
        weight = self.conv3(x)
        return weight

class PointConvAttention(ME.MinkowskiNetwork):
    def __init__(self, in_channel=256, out_channel=1, dimension=3):
        """
        # 每个点的特级别的Attention网络\n
        输入每个点卷积得到的特征\n
        输出每个特征的分数 [0,1]\n
        """
        ME.MinkowskiNetwork.__init__(self, dimension)
        # Decreases spatial resolution (conv stride > 1)
        # conv stride > 1 时就可以让点数变少
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, 512, kernel_size=5, stride=3, dimension=dimension),
            ME.MinkowskiBatchNorm(512),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(512, 1024, kernel_size=5, stride=3, dimension=dimension),
            ME.MinkowskiBatchNorm(1024),
            ME.MinkowskiReLU(),
        )
        self.tconv2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(1024, 512, kernel_size=5, stride=3, dimension=dimension),
            ME.MinkowskiBatchNorm(512),
            # ME.MinkowskiReLU(),
        )
        self.tconv1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(512, in_channel, kernel_size=5, stride=3, dimension=dimension),
            ME.MinkowskiBatchNorm(256),
            # ME.MinkowskiReLU(),
        )
        self.linear1 = nn.Sequential(
            ME.MinkowskiLinear(in_channel, 64, bias=False),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.linear2 = nn.Sequential(
            ME.MinkowskiLinear(64, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiSigmoid(),
        )

    def forward(self, x: ME.TensorField):
        # print("0: ", x.F.size())
        x = self.conv1(x)
        # print("1: ", x.F.size())
        x = self.conv2(x)
        # print("2: ", x.F.size())
        x = self.tconv2(x)
        # print("3: ", x.F.size())
        x = self.tconv1(x)
        # print("4: ", x.F.size())
        x = self.linear1(x)
        # print("5: ", x.F.size())
        x = self.linear2(x)
        # print("6: ", x.F.size())
        return x


def testPointWiseAttenion():
    # 生成数据
    bs, pn, d, fs = 1, 4, 3, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # quantization_size 相当于体素化
    coords = [ME.utils.sparse_quantize(coordinates=torch.rand((pn,d)), quantization_size=0.01)
                for _ in range(bs)]
    feats  = [torch.rand((coords[b].size()[0], fs), dtype=torch.float32) for b in range(bs)] 

    # 生成SparseTensors
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    minknet_input = ME.SparseTensor(coordinates=coords, features=feats, device=device,  requires_grad=True)

    pointWiseAttention = PointWiseAttention(in_channel=fs, out_channel=1).cuda()


    print("input: ", minknet_input)

    minknet_output = pointWiseAttention(minknet_input)


    print("weights: ", minknet_output)

    minknet_input = minknet_input*minknet_output

    print("output: ", minknet_input)

    return


def testPointFeatWiseAttenion():
    # 生成数据
    bs, pn, d, fs = 1, 4, 3, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # quantization_size 相当于体素化
    coords = [ME.utils.sparse_quantize(coordinates=torch.rand((pn,d)), quantization_size=0.01)
                for _ in range(bs)]
    feats  = [torch.rand((coords[b].size()[0], fs), dtype=torch.float32) for b in range(bs)] 

    # 生成SparseTensors
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    minknet_input = ME.SparseTensor(coordinates=coords, features=feats, device=device,  requires_grad=True)

    pointWiseAttention = PointFeatWiseAttention(in_channel=fs, out_channel=fs).cuda()


    print("input: ", minknet_input)

    minknet_output = pointWiseAttention(minknet_input)


    print("weights: ", minknet_output)

    minknet_input = minknet_input*minknet_output

    print("output: ", minknet_input)

    return

def testPointConvAttenion():
    # 生成数据
    bs, pn, d, fs = 2, 4096, 3, 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # quantization_size 相当于体素化
    coords = [ME.utils.sparse_quantize(coordinates=torch.rand((pn,d)), quantization_size=0.01)
                for _ in range(bs)]
    feats  = [torch.rand((coords[b].size()[0], fs), dtype=torch.float32) for b in range(bs)] 

    # 生成SparseTensors
    coords, feats = ME.utils.sparse_collate(coords=coords, feats=feats)
    minknet_input = ME.SparseTensor(coordinates=coords, features=feats, device=device,  requires_grad=True)

    pointWiseAttention = PointConvAttention(in_channel=fs, out_channel=1).cuda()


    # print("input: ", minknet_input)

    minknet_output = pointWiseAttention(minknet_input)


    # print("weights: ", minknet_output)

    minknet_input = minknet_input*minknet_output

    # print("output: ", minknet_input)

    return

if __name__ == "__main__":
    # testPointWiseAttenion()
    # testPointFeatWiseAttenion()
    testPointConvAttenion()