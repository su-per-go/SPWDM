import math
import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F
from itertools import repeat


class DetectionModel(nn.Module):
    def __init__(self, best_pre_len, best_suf_len, max_pool_output_size, tcn_param, mask_atte_param, sc_atte_param,
                 feat_fusion_param):
        super(DetectionModel, self).__init__()
        self.multi_tcn_pre = MutilTCNBlock(**tcn_param["pre_params"])
        self.multi_tcn_suf = MutilTCNBlock(**tcn_param["suf_params"])
        self.pre_sc_atte = SCAtte(**sc_atte_param)
        self.suf_sc_atte = SCAtte(**sc_atte_param)

        self.max_pool_output_size = max_pool_output_size
        small_len, big_len = sorted([best_pre_len, best_suf_len])
        lu_shu = 2
        self.adjustment_layer = None if best_pre_len == best_suf_len else nn.Linear(lu_shu * small_len,
                                                                                    big_len * lu_shu)
        self.conv_channel_reduction_1 = nn.Conv1d(tcn_param["pre_params"]["output_size"],
                                                  tcn_param["pre_params"]["output_size"] // 2,
                                                  kernel_size=1)
        self.mask_atte = MaskAttentionBlock(**mask_atte_param)

        self.g_atte = FeatureFusion(**feat_fusion_param)
        self.flatten = nn.Flatten()
        self.relu_1 = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm1d(mask_atte_param["hidden_dim"])
        self.fc1 = nn.Linear(self.max_pool_output_size * mask_atte_param["hidden_dim"] // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pre, suf, features, feature_mask):
        # 提取前缀局部特征
        pre_out = self.pre_sc_atte(self.multi_tcn_pre(pre))
        suf_out = self.suf_sc_atte(self.multi_tcn_suf(suf))
        # 维度调整
        if self.adjustment_layer is not None:
            if suf_out.shape[2] < pre_out.shape[2]:
                suf_out = self.adjustment_layer(suf_out)
            else:
                pre_out = self.adjustment_layer(pre_out)
        features_out = self.mask_atte(features, feature_mask)
        # 特征拼接
        pre_suf_features = torch.cat([pre_out, suf_out, features_out], dim=2)
        pre_suf_features = self.relu_1(pre_suf_features)
        pre_suf_features = F.layer_norm(pre_suf_features,
                                        normalized_shape=(pre_suf_features.shape[1], pre_suf_features.shape[2]))

        pre_suf_features = self.g_atte(pre_suf_features)
        output = F.adaptive_max_pool1d(pre_suf_features, output_size=self.max_pool_output_size)
        output = self.flatten(output)
        output = self.fc1(output)
        final_output = self.sigmoid(output)
        return final_output


class MutilTCNBlock(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, **kwargs):
        super(MutilTCNBlock, self).__init__()
        self.embedding = nn.Embedding(101, 16)
        self.spatia_dropout = SpatialDropout(dropout)
        self.TCN_1 = TCN(input_size, output_size, num_channels, kernel_size[0], dropout)
        self.TCN_2 = TCN(input_size, output_size, num_channels, kernel_size[1], dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.spatia_dropout(x)
        x1 = self.TCN_1(x)
        x2 = self.TCN_2(x)
        return torch.cat([x1, x2], dim=2)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
        self.conv1 = nn.Conv1d(input_size, output_size, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        residual = self.conv1(x)
        out += residual
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv_1 = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp_1 = Chomp1d(padding)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = SpatialDropout(dropout)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.chomp_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        else:
            return x.contiguous()


class MaskAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MaskAttentionBlock, self).__init__()
        self.linear_1 = nn.Linear(1, hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)

    def forward(self, x, mask):
        x = x.unsqueeze(-1)  # x shape(batch,feature_len)
        # 使用自注意力机制并传入key_padding_mask
        x = self.linear_1(x)
        x = x.permute(1, 0, 2)  # 调整形状为 (sequence_length, batch_size, hidden_dim)
        x, _ = self.self_attention(x, x, x, key_padding_mask=mask)
        x = x.permute(1, 2, 0)  # 恢复形状为 (batch_size, sequence_length, hidden_dim)
        return x


class SCAtte(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5, global_weight=0.5):
        super(SCAtte, self).__init__()

        self.local_size = local_size
        self.local_weight = local_weight
        self.global_weight = global_weight

        t = int(abs(math.log(in_size, 2) + b) / gamma)
        k = t if t % 2 else t + 1
        self.global_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.local_conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_arv_pool = nn.AdaptiveAvgPool1d(local_size)
        self.global_arv_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # 本地和全局平均池化
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, _ = x.shape
        # 调整维度以匹配卷积层
        local_arv = local_arv.reshape(b, 1, -1)
        global_arv = global_arv.view(b, 1, c)
        # 卷积操作
        y_local = self.local_conv(local_arv).view(b, c, self.local_size)
        y_global = self.global_conv(global_arv).view(b, c, 1)
        # 激活和池化
        att_local = y_local.sigmoid()
        att_global = F.adaptive_avg_pool1d(y_global.sigmoid(), self.local_size)
        # 融合注意力
        att_all = F.adaptive_avg_pool1d(att_global * self.global_weight + att_local * self.local_weight,
                                        x.shape[2])
        # 应用注意力到输入
        x = x * att_all
        return x


class FeatureFusion(nn.Module):
    def __init__(self, input_dim, embed_dim, pool_dim):
        super(FeatureFusion, self).__init__()
        self.conv_q = nn.Conv1d(input_dim, embed_dim, 1)
        self.conv_k = nn.Conv1d(input_dim, embed_dim, 1)
        self.conv_v = nn.Conv1d(input_dim, embed_dim, 1)
        self.fc_residual = nn.Conv1d(input_dim, embed_dim, 1)  # 用于调整残差连接的维度
        self.pool_dim = pool_dim
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.conv_q(x)
        K = self.conv_k(x)
        V = self.conv_v(x)
        K_pool = F.adaptive_avg_pool1d(K.transpose(1, 2), self.pool_dim).transpose(1, 2)
        V_pool = F.adaptive_avg_pool1d(V.transpose(1, 2), self.pool_dim).transpose(1, 2)

        # 缩放注意力分数
        attention_scores = torch.matmul(Q, K_pool.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        the_output = torch.matmul(attention_weights, V_pool)

        # 调整残差连接的维度
        residual = self.fc_residual(x)

        # 添加残差连接和 Layer Normalization
        the_output = the_output + residual
        the_output = F.layer_norm(the_output, the_output.shape[2:])

        return the_output


class SpatialDropout(nn.Module):

    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop

    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim() - 2), inputs.shape[-1])  # 默认沿着中间所有的shape
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)
            outputs.mul_(noises)
            return outputs

    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


if __name__ == '__main__':
    tcn_params = {
        "pre_params": {
            "input_size": 16,
            "output_size": 128,
            "num_channels": [64, 128],
            "kernel_size": [3, 5],
            "dropout": 0.1
        },
        "suf_params": {
            "input_size": 16,
            "output_size": 128,
            "num_channels": [64, 128],
            "kernel_size": [3, 5],
            "dropout": 0.1
        }
    }
    mask_attention_params = {
        "hidden_dim": 128,
        "num_heads": 16
    }
    sc_attention_params = {
        "in_size": 32,
        "local_size": 8,
        "local_weight": 0.5,
        "global_weight": 0.5,
    }
    feature_fusion_params = {
        "input_dim": 128,
        "embed_dim": 64,
        "pool_dim": 2
    }
    the_max_pool_output_size = 16
    frag_feature_model = DetectionModel(50, 50,
                                        the_max_pool_output_size, tcn_params, mask_attention_params,
                                        sc_attention_params, feature_fusion_params)
    X = torch.randint(1, 10, (32, 50))  # pre
    Y = torch.randint(1, 10, (32, 50))  # suf
    Z = torch.randint(1, 10, (32, 50)).to(torch.float)  # features
    MASK = torch.randint(2, (32, 50)).bool()
    summary(frag_feature_model, input_data=(X, Y, Z, MASK))
