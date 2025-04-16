from cuda.shift import Shift
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import math
sys.path.append("./model/Temporal_shift/")


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def gn_update(joint_data, bone_data, methods='elements_plus'):
    if methods == 'elements_plus':
        data = (joint_data + bone_data)/2
    elif methods == 'elements_mul':
        data = joint_data * bone_data
    elif methods == 'elements_concat':
        data = torch.cat((joint_data, bone_data), 1)
    else:
        raise ValueError('Method not found')

    return data


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        # print(x.shape)
        x = self.bn(self.conv(x))
        return x


class Shift_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels,
                               stride=stride, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        return x


class CeN(nn.Module):
    # self.cen = CeN(in_channels, num_joints=N, clip_len=T)
    def __init__(self, in_channels, num_joints, clip_len=300):
        super().__init__()
        self.num_joints = num_joints
        self.conv_c = nn.Conv2d(
            in_channels=in_channels, out_channels=1, kernel_size=1)
        self.conv_t = nn.Conv2d(
            in_channels=clip_len, out_channels=1, kernel_size=1)
        self.conv_v = nn.Conv2d(
            in_channels=num_joints,
            out_channels=num_joints * num_joints,
            kernel_size=1)
        self.bn = nn.BatchNorm2d(self.num_joints)

    def forward(self, x):
        x = self.conv_c(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # N T V C
        x = self.conv_t(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # N V C T

        x = self.bn(x)
        x = self.conv_v(x)

        n = x.size(0)
        A = x.view(n, self.num_joints, self.num_joints)
        d = torch.sum(torch.pow(A, 2), dim=1, keepdim=True)
        A = torch.div(A, torch.sqrt(d))
        return A


class STCAttention(nn.Module):

    def __init__(self, out_channels, num_joints):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # s attention
        ker_jpt = num_joints - 1 if not num_joints % 2 else num_joints
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        y = x
        # spatial attention
        se = y.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class JointProject(nn.Module):

    def __init__(self, in_channels, in_joints=25, out_joints=15):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints

        self.proj_mat = nn.Parameter(torch.empty(in_joints, out_joints))
        self.bn = nn.BatchNorm2d(in_channels)

        nn.init.kaiming_normal_(self.proj_mat)
        bn_init(self.bn, 1)

    def forward(self, x):
        joint_data, bone_data = torch.split(x, 1, dim=1)
        joint_data = torch.squeeze(joint_data, dim=1)
        bone_data = torch.squeeze(bone_data, dim=1)

        n, c, t, v = joint_data.size()
        # joint
        x = joint_data.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat)
        y = y.view(n, c, t, -1)
        y_j = self.bn(y)
        # bone
        x = bone_data.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat)
        y = y.view(n, c, t, -1)
        y_b = self.bn(y)
        y = torch.stack((y_j, y_b), dim=1)
        return y

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints,
                                                    self.out_joints)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, attention=True, clip_len=300, num_joints=25, update_A=True, type='joint'):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.clip_len = clip_len
        self.num_jpts = num_joints
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        self.update_A = update_A  # first layer Bone_GCN or not
        self.type = type

        self.PA = nn.Parameter(torch.empty(3, self.num_jpts, self.num_jpts))
        nn.init.constant_(self.PA, 1e-6)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_c = CeN(in_channels, num_joints=self.num_jpts,
                          clip_len=self.clip_len)

        if attention:
            self.attention_M = STCAttention(out_channels, self.num_jpts)
        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        joint_data, bone_data = torch.split(x, 1, dim=1)

        joint_data = torch.squeeze(joint_data, dim=1)
        bone_data = torch.squeeze(bone_data, dim=1)
        N, C, T, V = joint_data.size()  # [128, 64, 300, 25]

        y = None
        A1 = None
        A = self.PA
        down = None
        if not self.update_A:  # 如果是不需要更新Bone_GCN
            if self.type == 'joint':
                A1 = self.conv_c(joint_data)
            elif self.type == 'bone':
                # print(bone_data.shape, joint_data.shape)  # [64, 64, 300, 25])
                A1 = self.conv_c(bone_data)
            else:
                raise ValueError('Type not found')
        else:
            x = gn_update(joint_data, bone_data)
            A1 = self.conv_c(x)

        for i in range(self.num_subset):
            if self.type == 'joint':
                A1 = A[i] + A1 * self.alpha  # 到此邻接矩阵构成完毕
                A2 = joint_data.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
                down = self.down(joint_data)
            elif self.type == 'bone':
                A1 = A[i] + A1 * self.alpha  # 到此邻接矩阵构成完毕
                A2 = bone_data.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
                down = self.down(bone_data)
            else:
                raise ValueError('Type not found')
        y = self.bn(y)
        y += down
        y = self.relu(y)
        if self.attention:
            y = self.attention_M(y)
        # if self.type == 'joint':
        #     y = torch.stack((y, bone_data))
        # else:
        #     y = torch.stack((joint_data, y))
        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, attention=True, clip_len=300, adj_len=25):
        super(TCN_GCN_unit, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels or in_channels == 256:
            self.gcn_j = unit_gcn(in_channels, out_channels, A, clip_len=clip_len,
                                  attention=attention, num_joints=adj_len, update_A=False, type='joint')
            self.tcn_j = unit_tcn(out_channels, out_channels, stride=stride)
            self.gcn_b = unit_gcn(in_channels, out_channels, A, clip_len=clip_len,
                                  attention=attention, num_joints=adj_len, update_A=False, type='bone')
            self.tcn_b = unit_tcn(out_channels, out_channels, stride=stride)
            self.relu = nn.ReLU(inplace=True)
        else:  # GN层
            self.gcn_j = unit_gcn(in_channels, out_channels, A, clip_len=clip_len,
                                  attention=attention, num_joints=adj_len, type='joint')
            self.tcn_j = unit_tcn(out_channels, out_channels, stride=stride)
            self.gcn_b = unit_gcn(in_channels, out_channels, A, clip_len=clip_len,
                                  attention=attention, num_joints=adj_len, type='bone')
            self.tcn_b = unit_tcn(out_channels, out_channels, stride=stride)
            self.relu = nn.ReLU(inplace=True)

        self.attention = attention

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        joint_data, bone_data = torch.split(x, 1, dim=1)
        joint_data = torch.squeeze(joint_data, dim=1)
        bone_data = torch.squeeze(bone_data, dim=1)
        y_j = None
        y_b = None
        self.data = x
        if self.in_channels != self.out_channels or self.in_channels == 256:  # 如果不进行更新
            # joint
            x = self.gcn_j(x)
            # x = torch.squeeze(x, dim=0)
            y_j = self.relu(self.tcn_j(x) + self.residual(joint_data))
            # bone
            x = self.gcn_b(self.data)
            # x = torch.squeeze(x, dim=0)
            y_b = self.relu(self.tcn_b(x) + self.residual(bone_data))

            # global
            # TODO

        else:  # 经行GN更新
            # 先更新bone
            x = self.gcn_b(x)
            # x = torch.squeeze(x, dim=0)
            y_b = self.relu(self.tcn_b(x) + self.residual(bone_data))
            x = torch.stack((joint_data, y_b), dim=1)
            # 后更新joint
            x = self.gcn_j(x)
            # x = torch.squeeze(x, dim=0)
            y_j = self.relu(self.tcn_j(x) + self.residual(joint_data))
            # global
            # TODO

        return torch.stack((y_j, y_b), dim=1)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, attention=True, alpha=0.6):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        v1 = 25
        v2 = int(v1 * alpha)
        v3 = int(v2 * alpha)
        clip_len = 300
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.rgn_networks = nn.ModuleList((
            TCN_GCN_unit(3, 64, A, residual=False,
                         attention=attention, clip_len=300, adj_len=v1),
            TCN_GCN_unit(
                64, 64, A, attention=attention, clip_len=clip_len, adj_len=v1),
            TCN_GCN_unit(
                64, 64, A, attention=attention, clip_len=clip_len, adj_len=v1),
            TCN_GCN_unit(
                64, 64, A, attention=attention, clip_len=clip_len, adj_len=v1),
            TCN_GCN_unit(64, 128, A, stride=2,
                         attention=attention, clip_len=clip_len, adj_len=v1),
            JointProject(128, in_joints=v1, out_joints=v2),
            TCN_GCN_unit(
                128, 128, A, attention=attention, clip_len=clip_len // 2, adj_len=v2),
            TCN_GCN_unit(
                128, 128, A, attention=attention, clip_len=clip_len // 2, adj_len=v2),
            TCN_GCN_unit(128, 256, A, stride=2,
                         attention=attention, clip_len=clip_len // 2, adj_len=v2),
            JointProject(256, in_joints=v2, out_joints=v3),
            TCN_GCN_unit(
                256, 256, A, attention=attention, clip_len=clip_len // 4, adj_len=v3),
            TCN_GCN_unit(
                256, 256, A, attention=attention, clip_len=clip_len // 4, adj_len=v3)
        ))

        self.fc = nn.Linear(256 * 2, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        joint_data, bone_data = torch.split(x, 1, dim=1)
        joint_data = torch.squeeze(joint_data, dim=1)
        bone_data = torch.squeeze(bone_data, dim=1)
        # print(joint_data.shape, bone_data.shape)
        # both same as size[16, 3, 300, 25, 2]
        N, C, T, V, M = joint_data.size()
        # print(joint_data.shape, bone_data.shape)
        # joint
        x_j = joint_data.permute(
            0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x_j = self.data_bn(x_j)
        x_j = x_j.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # bone
        x_b = bone_data.permute(
            0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x_b = self.data_bn(x_b)
        x_b = x_b.view(N, M, V, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = torch.stack((x_j, x_b), dim=1)

        for i, rgn in enumerate(self.rgn_networks):
            x = rgn(x)

        # N*M,C,T,V
        joint_data, bone_data = torch.split(x, 1, dim=1)
        joint_data = torch.squeeze(joint_data, dim=1)
        bone_data = torch.squeeze(bone_data, dim=1)
        c_new = joint_data.size(1)
        joint_data = self.drop_out(
            joint_data.view(N, M, c_new, -1).mean(3).mean(1))
        bone_data = self.drop_out(bone_data.view(
            N, M, c_new, -1).mean(3).mean(1))
        out = torch.cat((joint_data, bone_data), dim=-1)

        return self.fc(out)


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    model = Model(graph='graph.directed_ntu_rgb_d.Graph')

    # for name, param in model.named_parameters():
    #     print('name is:', name)
    #     print('type(name):', type(name))
    #     print('param:', type(param))
    #     print()

    print('Model total # params:', sum(p.numel()
          for p in model.parameters() if p.requires_grad))
