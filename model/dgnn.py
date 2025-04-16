
from cuda.shift import Shift
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
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


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            # Conv along the temporal dimension only
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BiTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        # NOTE: assuming that temporal convs are shared between node/edge features
        self.tempconv = TemporalConv(
            in_channels, out_channels, kernel_size, stride)

    def forward(self, fv, fe):
        return self.tempconv(fv), self.tempconv(fe)


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

        # shrink
        A = torch.mean(A, dim=0)
        return A


class STCAttention(nn.Module):

    def __init__(self, out_channels, num_joints):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.out_channels = out_channels

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
        out_channels = self.out_channels//2
        # shrink
        fvp = y[:, :out_channels, :, :]
        fep = y[:, out_channels:, :, :]
        return fvp, fep


class JointProject(nn.Module):

    def __init__(self, in_channels, in_joints=25, out_joints=15):
        super().__init__()
        self.in_joints = in_joints
        self.out_joints = out_joints

        self.proj_mat_j = nn.Parameter(torch.empty(in_joints, out_joints))
        self.proj_mat_b = nn.Parameter(torch.empty(in_joints, out_joints))

        self.bn = nn.BatchNorm2d(in_channels)

        nn.init.kaiming_normal_(self.proj_mat_j)
        nn.init.kaiming_normal_(self.proj_mat_b)

        bn_init(self.bn, 1)

    def forward(self, joint_data, bone_data):

        n, c, t, v = joint_data.size()
        # joint
        x = joint_data.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat_j)
        y = y.view(n, c, t, -1)
        y_j = self.bn(y)
        # bone
        x = bone_data.view(n, c * t, v)
        y = torch.matmul(x, self.proj_mat_b)
        y = y.view(n, c, t, -1)
        y_b = self.bn(y)

        return y_j, y_b

    def extra_repr(self):
        return 'in_joints={}, out_joints={}'.format(self.in_joints,
                                                    self.out_joints)


class DGNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        # Adaptive block with learnable graphs; shapes (V_node, V_edge)
        self.source_M = nn.Parameter(
            torch.from_numpy(source_M.astype('float32')))
        self.target_M = nn.Parameter(
            torch.from_numpy(target_M.astype('float32')))

        # Updating functions
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)

        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)

        # residual
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.down_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
            self.down_1 = lambda x: x

        self.relu = nn.ReLU(inplace=True)

    def forward(self, fv, fe):
        # `fv` (node features) has shape (N, C, T, V_node)
        # `fe` (edge features) has shape (N, C, T, V_edge)
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape
        v0 = fv
        e0 = fe

        # Reshape for matmul, shape: (N, CT, V)
        fv = fv.reshape(N, -1, V_node)
        fe = fe.reshape(N, -1, V_edge)

        # Compute features for node/edge updates
        # Aggregate incoming edge features
        fe_in_agg = torch.einsum(
            'nce,ev->ncv', fe, self.source_M.transpose(0, 1))
        # Aggregate outgoing edge features
        fe_out_agg = torch.einsum(
            'nce,ev->ncv', fe, self.target_M.transpose(0, 1))
        # Stack node features and aggregated edge features
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)
        # Reshape and permute for linear layer
        fvp = fvp.view(N, 3 * C, T, V_node).contiguous().permute(0, 2, 3, 1)
        # Apply linear layer and permute back
        fvp = self.H_v(fvp).permute(0, 3, 1, 2)
        fvp = self.bn_v(fvp) + self.down(v0)  # Apply batch normalization
        fvp = self.relu(fvp)  # Apply ReLU activation

        # Aggregate incoming node features
        fv_in_agg = torch.einsum('ncv,ve->nce', fv, self.source_M)
        # Aggregate outgoing node features
        fv_out_agg = torch.einsum('ncv,ve->nce', fv, self.target_M)
        # Stack edge features and aggregated node features
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)
        # Reshape and permute for linear layer
        fep = fep.view(N, 3 * C, T, V_edge).contiguous().permute(0, 2, 3, 1)
        # Apply linear layer and permute back
        fep = self.H_e(fep).permute(0, 3, 1, 2)
        fep = self.bn_e(fep) + self.down_1(e0)  # Apply batch normalization
        fep = self.relu(fep)  # Apply ReLU activation
        return fvp, fep  # Return updated node and edge features


class DGNBlock_R(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, num_joints=25, clip_len=300, attention=False):
        super().__init__()
        self.num_nodes, self.num_edges = source_M.shape
        # Adaptive block with learnable graphs; shapes (V_node, V_edge)
        # self.source_M = nn.Parameter(
        #     torch.from_numpy(source_M.astype('float32')))
        # self.target_M = nn.Parameter(
        #     torch.from_numpy(target_M.astype('float32')))
        # Adaptive MATRIX
        self.num_jpts = num_joints
        self.clip_len = clip_len
        self.source_M = nn.Parameter(
            torch.from_numpy(source_M.astype('float32')), requires_grad=False)
        self.target_M = nn.Parameter(
            torch.from_numpy(target_M.astype('float32')), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.alpha_1 = nn.Parameter(torch.zeros(1))
        self.conv_c = CeN(in_channels*2, num_joints=self.num_jpts,
                          clip_len=self.clip_len)
        # Global attention
        if attention:
            # self.attention_M = STCAttention(out_channels*2, self.num_jpts)

            # shift
            self.out_channels = out_channels*2
            index_array = np.empty(25*self.out_channels).astype(int)
            for i in range(25):
                for j in range(self.out_channels):
                    index_array[i*self.out_channels + j] = (
                        i*self.out_channels + j - j*self.out_channels) % (self.out_channels*25)
            self.shift_out = nn.Parameter(
                torch.from_numpy(index_array), requires_grad=False)
            self.bn = nn.BatchNorm1d(25*self.out_channels)
            bn_init(self.bn, 1)

        self.attention = attention

        # Updating functions
        self.H_v = nn.Linear(3 * in_channels, out_channels)
        self.H_e = nn.Linear(3 * in_channels, out_channels)

        self.bn_v = nn.BatchNorm2d(out_channels)
        self.bn_e = nn.BatchNorm2d(out_channels)
        bn_init(self.bn_v, 1)
        bn_init(self.bn_e, 1)

        # residual
        # residual
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
            self.down_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
            self.down_1 = lambda x: x
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fv, fe):
        # `fv` (node features) has shape (N, C, T, V_node)
        # `fe` (edge features) has shape (N, C, T, V_edge)
        N, C, T, V_node = fv.shape
        _, _, _, V_edge = fe.shape
        v0 = fv
        e0 = fe

        # adptive matrix
        data_concat = torch.cat((fv, fe), dim=1)
        cen = self.conv_c(data_concat)  # V V
        source_M = self.source_M + self.alpha * cen
        target_M = self.target_M + self.alpha_1 * cen

        # # Visualize the target matrix as per the paper's standard
        # topology = self.source_M + self.target_M + self.alpha * cen

        #  # Visualization
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # import datetime
        # def visualize_matrix(matrix, title, filename):
        #     plt.figure(figsize=(10, 8))
        #     sns.heatmap(matrix.detach().cpu().numpy(), annot=False, cmap='viridis')
        #     plt.title(title)
        #     filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + filename + ".png"
        #     plt.savefig(filename)
        #     plt.close()

        # visualize_matrix(cen, 'CEN Matrix', 'cen_matrix.png')
        # visualize_matrix(topology, 'Topology Matrix', 'topology_matrix.png')

        # Reshape for matmul, shape: (N, CT, V)
        fv = fv.reshape(N, -1, V_node)
        fe = fe.reshape(N, -1, V_edge)

        # Aggregate incoming node features
        fv_in_agg = torch.einsum('ncv,ve->nce', fv, source_M)
        # Aggregate outgoing node features
        fv_out_agg = torch.einsum('ncv,ve->nce', fv, target_M)
        # Stack edge features and aggregated node features
        fep = torch.stack((fe, fv_in_agg, fv_out_agg), dim=1)
        # Reshape and permute for linear layer
        fep = fep.view(N, 3 * C, T, V_edge).contiguous().permute(0, 2, 3, 1)
        # Apply linear layer and permute back
        fep = self.H_e(fep).permute(0, 3, 1, 2)
        fep = self.bn_e(fep)  # Apply batch normalization
        fep = self.relu(fep)  # Apply ReLU activation

        fep_temp = fep.reshape(N, -1, V_edge)
        # Compute features for node/edge updates
        # Aggregate incoming edge features
        fe_in_agg = torch.einsum(
            'nce,ev->ncv', fep_temp, source_M.transpose(0, 1))
        # Aggregate outgoing edge features
        fe_out_agg = torch.einsum(
            'nce,ev->ncv', fep_temp, target_M.transpose(0, 1))
        # Stack node features and aggregated edge features
        fvp = torch.stack((fv, fe_in_agg, fe_out_agg), dim=1)
        # Reshape and permute for linear layer
        fvp = fvp.view(N, 3 * C, T, V_node).contiguous().permute(0, 2, 3, 1)
        # Apply linear layer and permute back
        fvp = self.H_v(fvp).permute(0, 3, 1, 2)
        fvp = self.bn_v(fvp)  # Apply batch normalization
        fvp = self.relu(fvp)  # Apply ReLU activation

        # Global attention
        if self.attention:
            data_concat = torch.cat((fvp, fep), dim=1)
            # fvp, fep = self.attention_M(data_concat)

            data_concat = data_concat.permute(
                0, 2, 3, 1).contiguous()  # n,t,v,c
            data_concat = data_concat.view(N*T, -1)
            data_concat = torch.index_select(data_concat, 1, self.shift_out)
            data_concat = self.bn(data_concat)
            data_concat = data_concat.view(
                N, T, V_node, self.out_channels).permute(0, 3, 1, 2)  # n,c,t,v
            out_channels = self.out_channels//2
            # shrink
            fvp = data_concat[:, :out_channels, :, :]
            fep = data_concat[:, out_channels:, :, :]
            fep = fep + self.down_1(e0)
            fvp = fvp + self.down(v0)
            fvp = self.relu(fvp)
            fep = self.relu(fep)
        else:
            fep = fep + self.down_1(e0)
            fvp = fvp + self.down(v0)

        return fvp, fep  # Return updated node and edge features


class GraphTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True):
        super(GraphTemporalConv, self).__init__()
        self.dgn = DGNBlock(in_channels, out_channels, source_M, target_M)
        self.tcn = BiTemporalConv(
            out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = BiTemporalConv(
                in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv_res, fe_res = self.residual(fv, fe)
        fv, fe = self.dgn(fv, fe)
        fv, fe = self.tcn(fv, fe)
        fv += fv_res
        fe += fe_res
        return self.relu(fv), self.relu(fe)


class GraphTemporalConv_R(nn.Module):
    def __init__(self, in_channels, out_channels, source_M, target_M, temp_kernel_size=9, stride=1, residual=True, clip_len=300):
        super(GraphTemporalConv_R, self).__init__()
        self.dgn = DGNBlock_R(in_channels, out_channels,
                              source_M, target_M, clip_len=clip_len)
        self.tcn = BiTemporalConv(
            out_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda fv, fe: (0, 0)
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda fv, fe: (fv, fe)
        else:
            self.residual = BiTemporalConv(
                in_channels, out_channels, kernel_size=temp_kernel_size, stride=stride)

    def forward(self, fv, fe):
        fv_res, fe_res = self.residual(fv, fe)
        fv, fe = self.dgn(fv, fe)
        fv, fe = self.tcn(fv, fe)
        fv += fv_res
        fe += fe_res
        return self.relu(fv), self.relu(fe)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            # KV config pairs should be supplied with the config file
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        source_M, target_M = self.graph.source_M, self.graph.target_M
        clip_len = 300
        self.data_bn_v = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn_e = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = GraphTemporalConv(3, 64, source_M, target_M, residual=False)
        self.l2 = GraphTemporalConv_R(
            64, 64, source_M, target_M, clip_len=clip_len)
        self.l3 = GraphTemporalConv_R(
            64, 64, source_M, target_M, clip_len=clip_len)
        self.l4 = GraphTemporalConv_R(
            64, 64, source_M, target_M, clip_len=clip_len)
        self.l5 = GraphTemporalConv(64, 128, source_M, target_M, stride=2)
        self.l6 = GraphTemporalConv_R(
            128, 128, source_M, target_M, clip_len=clip_len // 2)
        self.l7 = GraphTemporalConv_R(
            128, 128, source_M, target_M, clip_len=clip_len // 2)
        self.l8 = GraphTemporalConv(128, 256, source_M, target_M, stride=2)
        self.l9 = GraphTemporalConv_R(
            256, 256, source_M, target_M, clip_len=clip_len // 4)
        self.l10 = GraphTemporalConv_R(
            256, 256, source_M, target_M, clip_len=clip_len // 4)

        self.fc = nn.Linear(256 * 2, num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn_v, 1)
        bn_init(self.data_bn_e, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        def count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        print('Model total number of params:', count_params(self))

    def forward(self, x):
        fv, fe = torch.split(x, 1, dim=1)
        fv = torch.squeeze(fv, dim=1)
        fe = torch.squeeze(fe, dim=1)

        # fv: torch.Size([32, 3, 300, 25, 2])
        # fe: torch.Size([32, 3, 300, 25, 2])
        N, C, T, V_node, M = fv.shape
        _, _, _, V_edge, _ = fe.shape

        # Preprocessing
        fv = fv.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_node * C, T)
        fv = self.data_bn_v(fv)
        fv = fv.view(N, M, V_node, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_node)

        fe = fe.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V_edge * C, T)
        fe = self.data_bn_e(fe)
        fe = fe.view(N, M, V_edge, C, T).permute(
            0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V_edge)

        fv, fe = self.l1(fv, fe)
        fv, fe = self.l2(fv, fe)
        fv, fe = self.l3(fv, fe)
        fv, fe = self.l4(fv, fe)
        fv, fe = self.l5(fv, fe)
        fv, fe = self.l6(fv, fe)
        fv, fe = self.l7(fv, fe)
        fv, fe = self.l8(fv, fe)
        fv, fe = self.l9(fv, fe)
        fv, fe = self.l10(fv, fe)

        # Shape: (N*M,C,T,V), C is same for fv/fe
        out_channels = fv.size(1)

        # Performs pooling over both nodes and frames, and over number of persons
        fv = fv.view(N, M, out_channels, -1).mean(3).mean(1)
        fe = fe.view(N, M, out_channels, -1).mean(3).mean(1)

        # Concat node and edge features
        out = torch.cat((fv, fe), dim=-1)
        out = self.drop_out(out)
        out = self.fc(out)

        return out


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
