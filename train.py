import torch, json

from rich import print

from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from model import HistoryAttention
from sentence_transformers import SentenceTransformer

class TextScoreDataset(Dataset):
    def __init__(self, data, transformer_model_name='uer/sbert-base-chinese-nli'):
        """
        Args:
            data (list of dicts): 数据列表，每个字典包含 'instruction', 'input', 'output' 和 'history'。
            transformer_model_name (str): 要使用的句子变换器模型名称。
        """
        self.model = SentenceTransformer(transformer_model_name)
        self.model = self.model.to("cuda:1")
        self.data = data
        for index, item in enumerate(data):
            print(f"bert编码中:{index}")
            item["instruction"] = self.model.encode(item['instruction'])
            for history_index, history in enumerate(item["history"]):
                item["history"][history_index] = (self.model.encode(history[0]),history[2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回数据集中的单个项。
        """
        item = self.data[idx]

        # 获取句子嵌入
        instruction_embedding = item["instruction"]
        history_embeddings = [embed for embed, _ in item["history"]]

        scores = [score for _, score in item['history']]

        # 将嵌入和分数转换为torch张量
        instruction_embedding = torch.tensor(instruction_embedding).to("cuda:1")
        history_embeddings = torch.tensor(history_embeddings).to("cuda:1")
        scores = torch.tensor(scores).to("cuda:1")

        return instruction_embedding, history_embeddings, scores

# 定义超参数
dim_embed = 768
dim_inner = 768

model = HistoryAttention(dim_embed,dim_inner)
model = model.to("cuda:1")
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
criterion = torch.nn.MSELoss()

# 训练模型
num_epochs = 50

with open("./data/hisw_mix.json",'r') as file:
    dataset = TextScoreDataset(json.loads(file.read()))
data_loader = DataLoader(dataset=dataset,batch_size=1,shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for step, data in enumerate(data_loader):
        instruction_embedding, history_embedding, scores = data
        # 清空梯度
        optimizer.zero_grad()

        # 合并指令和历史嵌入作为模型输入，这里简单地将它们相加
        # 在实际应用中，你可能需要更复杂的操作来合并这些嵌入
        inputs = instruction_embedding, history_embedding

        # 预测
        predictions = model(*inputs)

        # 计算损失
        loss = criterion(predictions.squeeze(), scores.squeeze(0))

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()
        if step % 300 == 0:
            print(f'Step {step}, Loss: {loss.item()}, weight_cosine:{model.weight_cosine_sim.tolist()}, weight_attn:{model.weight_attention.tolist()}')
    scheduler.step()
    print(f'[green]Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}, weight_cosine:{model.weight_cosine_sim.tolist()}, weight_attn:{model.weight_attention.tolist()}')
torch.save(model.state_dict(),'history_attn.pth')