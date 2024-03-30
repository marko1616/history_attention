import torch
import torch.nn as nn
import torch.nn.functional as F

class HistoryAttention(nn.Module):
    def __init__(self, len_embed, dim):
        super(HistoryAttention, self).__init__()
        #线性层，将当前输入映射为Query
        self.query_map = nn.Linear(len_embed, dim)
        #线性层，将历史输入映射为Key
        self.key_map = nn.Linear(len_embed, dim)

        self.len_embed = len_embed
        self.d_q = dim
        self.d_k = dim

        # 添加两个可学习的权重w
        self.weight_cosine_sim = nn.Parameter(torch.Tensor([1.0]))
        self.weight_attention = nn.Parameter(torch.Tensor([0.0]))
        
    def forward(self, current_input, historical_inputs):
        #current_input的shape：(batch_size, len_embed)
        #historical_inputs的shape：(batch_size, his_num, len_embed)
        #将当前输入映射为Query
        query = self.query_map(current_input)#shape: (batch_size, d_q)
        #将历史输入映射为Key
        key = self.key_map(historical_inputs)#shape: (batch_size, his_num, d_k)

        #计算余弦相似度
        cosine_similarity = F.cosine_similarity(current_input.unsqueeze(1), historical_inputs, dim=-1)

        #计算注意力分数
        #需要在query上增加一个维度，使其shape变为(batch_size, 1, d_q)
        #以便与key进行矩阵乘法
        attention_scores = torch.matmul(query.unsqueeze(1), key.transpose(-2, -1))#shape: (batch_size, 1, his_num)
        attention_scores /= torch.sqrt(torch.tensor(self.d_k, dtype=query.dtype))#防止softmax梯度消失
        #加权余弦相似度
        attention_scores = self.weight_attention * attention_scores + self.weight_cosine_sim * cosine_similarity.unsqueeze(1)

        #attention_scores = F.softmax(attention_scores, dim=-1)#shape: (batch_size, 1, his_num)
        attention_output = attention_scores.squeeze(1)#shape: (batch_size, his_num)

        return attention_output