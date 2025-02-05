import torch, numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as tg_softmax
from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GCNConv,
    DiffGroupNorm
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter


class GATGNN_GIM1_globalATTENTION(torch.nn.Module):
    """
    全局注意力模块，用于计算节点的全局注意力权重。
    """
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2):
        super(GATGNN_GIM1_globalATTENTION, self).__init__()

        self.act = act  # 激活函数名称（字符串）
        self.fc_layers = fc_layers  # 全连接层的数量
        self.batch_track_stats = True if batch_track_stats == "True" else False  # 是否跟踪批标准化的统计量

        self.batch_norm = batch_norm  # 是否使用批标准化
        self.dropout_rate = dropout_rate  # Dropout率

        self.global_mlp = torch.nn.ModuleList()  # 用于存储多层感知器（MLP）的ModuleList
        self.bn_list = torch.nn.ModuleList()  # 用于存储批标准化层的ModuleList

        assert fc_layers > 1, "需要至少2个全连接层"  # 确保全连接层的数量大于1

        # 初始化全连接层
        for i in range(self.fc_layers + 1):
            if i == 0:
                # 第一层的输入维度是dim + 108（假设节点特征和全局特征拼接后的维度）
                lin = torch.nn.Linear(dim + 108, dim)
                self.global_mlp.append(lin)  # 添加到MLP列表
            else:
                if i != self.fc_layers:
                    # 中间层，输入和输出维度都是dim
                    lin = torch.nn.Linear(dim, dim)
                else:
                    # 最后一层，输出维度是1（用于计算注意力权重）
                    lin = torch.nn.Linear(dim, 1)
                self.global_mlp.append(lin)  # 添加到MLP列表

            # 添加批标准化层（如果启用）
            if self.batch_norm == "True":
                #bn = BatchNorm1d(dim, track_running_stats=self.batch_track_stats)
                bn = DiffGroupNorm(dim, 10, track_running_stats=self.batch_track_stats)  # 使用差分分组批标准化
                self.bn_list.append(bn)  # 添加到批标准化列表

    def forward(self, x, batch, glbl_x):
        """
        前向传播函数。

        参数：
        - x: 节点特征张量，形状为 [num_nodes, dim]
        - batch: 批索引张量，形状为 [num_nodes]
        - glbl_x: 全局特征张量，形状为 [num_nodes, 108]
        """
        out = torch.cat([x, glbl_x], dim=-1)  # 拼接节点特征和全局特征，形状变为 [num_nodes, dim + 108]

        # 通过全连接层和激活函数处理输入
        for i in range(0, len(self.global_mlp)):
            if i != len(self.global_mlp) - 1:
                out = self.global_mlp[i](out)  # 通过全连接层
                out = getattr(F, self.act)(out)  # 应用激活函数
            else:
                out = self.global_mlp[i](out)  # 最后一层不使用激活函数
                out = tg_softmax(out, batch)  # 计算注意力权重
        return out  # 返回注意力权重


class GATGNN_AGAT_LAYER(MessagePassing):
    """
    图注意力层（GAT），继承自 `MessagePassing`。
    """
    def __init__(self, dim, act, batch_norm, batch_track_stats, dropout_rate, fc_layers=2, **kwargs):
        super(GATGNN_AGAT_LAYER, self).__init__(aggr='add', flow='target_to_source', **kwargs)

        self.act = act  # 激活函数名称（字符串）
        self.fc_layers = fc_layers  # 全连接层的数量
        self.batch_track_stats = True if batch_track_stats == "False" else False  # 是否跟踪批标准化的统计量

        self.batch_norm = batch_norm  # 是否使用批标准化
        self.dropout_rate = dropout_rate  # Dropout率

        # 固定参数
        self.heads = 4  # 注意力头的数量
        self.add_bias = True  # 是否添加偏置
        self.neg_slope = 0.2  # LeakyReLU的负斜率

        self.bn1 = nn.BatchNorm1d(self.heads)  # 批标准化层
        self.W = Parameter(torch.Tensor(dim * 2, self.heads * dim))  # 权重矩阵
        self.att = Parameter(torch.Tensor(1, self.heads, 2 * dim))  # 注意力参数
        self.dim = dim  # 特征维度

        if self.add_bias:
            self.bias = Parameter(torch.Tensor(dim))  # 偏置向量
        else:
            self.register_parameter('bias', None)  # 不使用偏置

        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):
        glorot(self.W)  # 初始化权重矩阵
        glorot(self.att)  # 初始化注意力参数
        zeros(self.bias)  # 初始化偏置向量

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播函数。

        参数：
        - x: 节点特征张量，形状为 [num_nodes, dim]
        - edge_index: 边的索引张量，形状为 [2, num_edges]
        - edge_attr: 边的特征张量，形状为 [num_edges, dim]
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)  # 传播消息

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        """
        消息传播函数。

        参数：
        - edge_index_i: 边的目标节点索引
        - x_i: 目标节点特征
        - x_j: 源节点特征
        - size_i: 目标节点的数量
        - edge_attr: 边特征
        """
        out_i = torch.cat([x_i, edge_attr], dim=-1)  # 拼接目标节点特征和边特征
        out_j = torch.cat([x_j, edge_attr], dim=-1)  # 拼接源节点特征和边特征

        out_i = getattr(F, self.act)(torch.matmul(out_i, self.W))  # 通过全连接层和激活函数处理目标节点特征
        out_j = getattr(F, self.act)(torch.matmul(out_j, self.W))  # 通过全连接层和激活函数处理源节点特征

        out_i = out_i.view(-1, self.heads, self.dim)  # 调整形状为 [num_edges, heads, dim]
        out_j = out_j.view(-1, self.heads, self.dim)  # 调整形状为 [num_edges, heads, dim]

        alpha = getattr(F, self.act)((torch.cat([out_i, out_j], dim=-1) * self.att).sum(dim=-1))  # 计算注意力分数
        alpha = getattr(F, self.act)(self.bn1(alpha))  # 应用批标准化
        alpha = tg_softmax(alpha, edge_index_i)  # 应用softmax函数计算注意力权重

        alpha = F.dropout(alpha, p=self.dropout_rate, training=self.training)  # 应用Dropout
        out_j = (out_j * alpha.view(-1, self.heads, 1)).transpose(0, 1)  # 聚合源节点特征和注意力权重
        return out_j

    def update(self, aggr_out):
        """
        更新函数，聚合消息并输出。

        参数：
        - aggr_out: 聚合后的消息
        """
        out = aggr_out.mean(dim=0)  # 按头维度取平均
        if self.bias is not None:
            out = out + self.bias  # 添加偏置
        return out  # 返回更新后的节点特征


# CGCNN
class DEEP_GATGNN(torch.nn.Module):
    """
    深度GAT模型，结合全局注意力机制。

    DEEP_GATGNN 模型的初始化函数。

    参数:
    - self: 类的实例，表示当前对象。
    - data: 输入数据，包含节点特征、边索引和边特征等信息。
    - dim1: 第一层全连接层的维度，默认值为 64。
    - dim2: 第二层全连接层的维度，默认值为 64。
    - pre_fc_count: 预处理全连接层的数量，默认值为 1。
    - gc_count: 图注意力层（GATGNN_AGAT_LAYER）的数量，也就是图卷积的层数，默认值为 5。
    - post_fc_count: 后处理全连接层的数量，默认值为 1。
    - pool: 池化方法，默认值为 "global_add_pool"。
    - pool_order: 池化顺序，默认值为 "early"。
    - batch_norm: 是否使用批标准化，默认值为 "True"。
    - batch_track_stats: 是否跟踪批标准化的统计量，默认值为 "True"。
    - act: 激活函数名称（字符串），默认值为 "softplus"。
    - dropout_rate: Dropout率，默认值为 0.0。
    - **kwargs: 其他可选参数，用于扩展功能。
    """

    def __init__(
        self,
        data,
        dim1=64,
        dim2=64,
        pre_fc_count=1,
        gc_count=5, # 表示图注意力层（GATGNN_AGAT_LAYER）的数量，也就是图卷积的层数
        post_fc_count=1,
        pool="global_add_pool",
        pool_order="early",
        batch_norm="True",
        batch_track_stats="True",
        act="softplus",
        dropout_rate=0.0,
        **kwargs
    ):
        super(DEEP_GATGNN, self).__init__()

        # 参数初始化
        self.batch_track_stats = True if batch_track_stats == "False" else False  # 是否跟踪批标准化的统计量
        self.batch_norm = batch_norm  # 是否使用批标准化
        self.pool = pool  # 池化方法
        self.act = act  # 激活函数名称（字符串）
        self.pool_order = pool_order  # 池化顺序
        self.dropout_rate = dropout_rate  # Dropout率

        # 全局注意力模块初始化
        self.heads = 4  # 注意力头的数量
        self.global_att_LAYER = GATGNN_GIM1_globalATTENTION(dim1, act, batch_norm, batch_track_stats, dropout_rate)

        # 确定图卷积层的维度
        assert gc_count > 0, "需要至少1个图卷积层"
        gc_dim = dim1 if pre_fc_count == 0 else dim1  # 图卷积层的输入维度
        post_fc_dim = data.num_features if pre_fc_count == 0 else dim1  # 后处理层的输入维度

        # 确定输出维度
        output_dim = 1 if data[0].y.ndim == 0 else len(data[0].y[0])  # 输出维度

        # 预处理层（全连接层）
        self.pre_lin_list_E = torch.nn.ModuleList()  # 边特征的全连接层
        self.pre_lin_list_N = torch.nn.ModuleList()  # 节点特征的全连接层
        if pre_fc_count > 0:
            for i in range(pre_fc_count):
                if i == 0:
                    # 初始化第一层
                    lin_N = torch.nn.Linear(data.num_features, dim1)
                    lin_E = torch.nn.Linear(data.num_edge_features, dim1)
                else:
                    # 初始化后续层
                    lin_N = torch.nn.Linear(dim1, dim1)
                    lin_E = torch.nn.Linear(dim1, dim1)
                self.pre_lin_list_N.append(lin_N)  # 添加到节点特征全连接层列表
                self.pre_lin_list_E.append(lin_E)  # 添加到边特征全连接层列表

        # 图卷积层
        self.conv_list = torch.nn.ModuleList()  # 图卷积层列表
        self.bn_list = torch.nn.ModuleList()  # 批标准化层列表
        for i in range(gc_count):
            conv = GATGNN_AGAT_LAYER(dim1, act, batch_norm, batch_track_stats, dropout_rate)  # 创建图卷积层
            self.conv_list.append(conv)  # 添加到图卷积层列表
            if self.batch_norm == "True":
                bn = DiffGroupNorm(gc_dim, 10, track_running_stats=self.batch_track_stats)  # 创建批标准化层
                self.bn_list.append(bn)  # 添加到批标准化层列表

        # 后处理层（全连接层）
        self.post_lin_list = torch.nn.ModuleList()  # 后处理层列表
        if post_fc_count > 0:
            for i in range(post_fc_count):
                if i == 0:
                    # 初始化第一层
                    lin = torch.nn.Linear(post_fc_dim * 2, dim2) if pool == "set2set" and pool_order == "early" else torch.nn.Linear(post_fc_dim, dim2)
                else:
                    # 初始化后续层
                    lin = torch.nn.Linear(dim2, dim2)
                self.post_lin_list.append(lin)  # 添加到后处理层列表
            self.lin_out = torch.nn.Linear(dim2, output_dim)  # 输出层
        else:
            self.lin_out = torch.nn.Linear(post_fc_dim, output_dim)  # 输出层

        # Set2Set池化（如果启用）
        if self.pool_order == "early" and self.pool == "set2set":
            self.set2set = Set2Set(post_fc_dim, processing_steps=3)  # Set2Set池化层
        elif self.pool_order == "late" and self.pool == "set2set":
            self.set2set = Set2Set(output_dim, processing_steps=3, num_layers=1)  # Set2Set池化层
            self.lin_out_2 = torch.nn.Linear(output_dim * 2, output_dim)  # 输出层

    def forward(self, data):
        """
        前向传播函数。

        参数：
        - data: 输入数据，包含节点特征、边索引和边特征等。
        """
        # 预处理层
        for i in range(len(self.pre_lin_list_N)):
            if i == 0:
                out_x = self.pre_lin_list_N[i](data.x)  # 节点特征通过全连接层
                out_x = getattr(F, 'leaky_relu')(out_x, 0.2)  # 应用LeakyReLU激活函数
                out_e = self.pre_lin_list_E[i](data.edge_attr)  # 边特征通过全连接层
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)  # 应用LeakyReLU激活函数
            else:
                out_x = self.pre_lin_list_N[i](out_x)  # 节点特征通过全连接层
                out_x = getattr(F, self.act)(out_x)  # 应用激活函数
                out_e = self.pre_lin_list_E[i](out_e)  # 边特征通过全连接层
                out_e = getattr(F, 'leaky_relu')(out_e, 0.2)  # 应用LeakyReLU激活函数

        prev_out_x = out_x  # 保存前一层的输出

        # 图卷积层
        for i in range(len(self.conv_list)):
            if len(self.pre_lin_list_N) == 0 and i == 0:
                out_x = self.conv_list[i](data.x, data.edge_index, data.edge_attr)  # 图卷积层前向传播
                if self.batch_norm == "True":
                    out_x = self.bn_list[i](out_x)  # 批标准化
            else:
                out_x = self.conv_list[i](out_x, data.edge_index, out_e)  # 图卷积层前向传播
                if self.batch_norm == "True":
                    out_x = self.bn_list[i](out_x)  # 批标准化
            out_x = torch.add(out_x, prev_out_x)  # 残差连接
            out_x = F.dropout(out_x, p=self.dropout_rate, training=self.training)  # Dropout
            prev_out_x = out_x  # 保存当前层的输出

        # 全局注意力层
        out_a = self.global_att_LAYER(out_x, data.batch, data.glob_feat)  # 计算注意力权重
        out_x = out_x * out_a  # 应用注意力权重

        # 后处理层
        if self.pool_order == "early":
            if self.pool == "set2set":  # 使用Set2Set池化
                out_x = self.set2set(out_x, data.batch)
            else:  # 使用其他池化方法
                out_x = getattr(torch_geometric.nn, self.pool)(out_x, data.batch)
            for i in range(len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)  # 通过全连接层
                out_x = getattr(F, self.act)(out_x)  # 应用激活函数
            out = self.lin_out(out_x)  # 输出层

        elif self.pool_order == "late":
            for i in range(len(self.post_lin_list)):
                out_x = self.post_lin_list[i](out_x)  # 通过全连接层
                out_x = getattr(F, self.act)(out_x)  # 应用激活函数
            out = self.lin_out(out_x)  # 输出层
            if self.pool == "set2set":  # 使用Set2Set池化
                out = self.set2set(out, data.batch)
                out = self.lin_out_2(out)  # 输出层
            else:  # 使用其他池化方法
                out = getattr(torch_geometric.nn, self.pool)(out, data.batch)

        return out.view(-1) if out.shape[1] == 1 else out  # 返回输出结果
