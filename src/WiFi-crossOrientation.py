import os
import scipy.io as scio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import Dataset
from lib.gesture_mapping import gesture_descriptions
from lib.gesture_mapping import number_to_gesture20181109

# 数据加载与预处理
def normalize_data(data_1):
    data_1_max = np.concatenate((data_1.max(axis=0), data_1.max(axis=1)), axis=0).max(axis=0)
    data_1_min = np.concatenate((data_1.min(axis=0), data_1.min(axis=1)), axis=0).min(axis=0)
    if (len(np.where((data_1_max - data_1_min) == 0)[0]) > 0):
        return data_1
    data_1_max_rep = np.tile(data_1_max, (data_1.shape[0], data_1.shape[1], 1))
    data_1_min_rep = np.tile(data_1_min, (data_1.shape[0], data_1.shape[1], 1))
    data_1_norm = (data_1 - data_1_min_rep) / (data_1_max_rep - data_1_min_rep)
    return data_1_norm

def zero_padding(data, T_MAX):
    data_pad = []
    for i in range(len(data)):
        t = np.array(data[i]).shape[2]
        data_pad.append(np.pad(data[i], ((0, 0), (0, 0), (T_MAX - t, 0)), 'constant', constant_values=0).tolist())
    return np.array(data_pad)

def load_data(path_to_data, motion_sel, test_orientation = 1):
    global T_MAX
    #data = []
    #label = []
    train_data, train_labels = [], []
    test_data, test_labels = [], []
    for data_root, data_dirs, data_files in os.walk(path_to_data):
        if "Room1" or "Room2" in data_root:
            continue
        for data_file_name in data_files:
            file_path = os.path.join(data_root, data_file_name)
            try:
                data_1 = scio.loadmat(file_path)['velocity_spectrum_ro']
                label_1 = int(data_file_name.split('-')[1])
                torso_location = int(data_file_name.split('-')[2])  # 解析 torso location
                face_orientation = int(data_file_name.split('-')[3])  # 解析 face_orientation
                if label_1 not in motion_sel:
                    continue
                data_normed_1 = normalize_data(data_1)
                if T_MAX < np.array(data_1).shape[2]:
                    T_MAX = np.array(data_1).shape[2]
            except Exception:
                continue

            # 选择一个 torso location 作为测试，其他 torso location 作为训练
            if face_orientation == test_orientation:
                test_data.append(data_normed_1.tolist())
                test_labels.append(label_1)
            else:
                train_data.append(data_normed_1.tolist())
                train_labels.append(label_1)

    # 处理形状
    train_data = zero_padding(train_data, T_MAX)
    test_data = zero_padding(test_data, T_MAX)

    train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 2, 3)
    test_data = np.swapaxes(np.swapaxes(test_data, 1, 3), 2, 3)

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    return train_data, test_data, train_labels, test_labels

# ================== 1. 自定义 Dataset ==================
class WiFiTextDataset(Dataset):
    def __init__(self, wifi_data, labels, tokenizer, max_len=128):
        self.wifi_data = torch.tensor(wifi_data, dtype=torch.float32).permute(0, 4, 1, 2, 3)  # 调整维度顺序
        self.labels = torch.tensor(labels, dtype=torch.long)  # 确保 label 是张量
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.wifi_data)

    def __getitem__(self, idx):
        wifi_sample = self.wifi_data[idx]
        label = int(self.labels[idx])
        gesture_name_1 = number_to_gesture20181109.get(label, "unknown")
        gesture_name = gesture_descriptions.get(label, "This is an unknown hand motion.")
        gesture_name_2 =  "This is a motion of" + gesture_name_1 + "," + gesture_name
        encoded_text = self.tokenizer(
            gesture_name_2,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded_text["input_ids"].squeeze(0)
        attention_mask = encoded_text["attention_mask"].squeeze(0)

        return wifi_sample, input_ids, attention_mask, label

# ================== 2. WiFi 编码器 ==================
class WiFiEncoder(nn.Module):
    def __init__(self, input_shape, feature_dim=512, dropout_ratio=0.5, gru_hidden_units=256):
        super(WiFiEncoder, self).__init__()

        C, T_MAX, H, W = input_shape  # (C=1, T_MAX=时间步长, H=20, W=20)

        # 3D CNN 提取时空特征
        self.conv1 = nn.Conv3d(in_channels=C, out_channels=16, kernel_size=(3, 5, 5), padding=(1, 2, 2))
        self.pool = nn.MaxPool3d((1, 2, 2))  # 仅对 H, W 维度池化
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)

        # 计算 Flatten 后的维度
        self.flatten_dim = 32 * (H // 2) * (W // 2)
        self.fc1 = nn.Linear(self.flatten_dim, 128)

        # GRU 时间序列处理
        self.gru = nn.GRU(128, gru_hidden_units, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(dropout_ratio)

        # 输出 feature_dim 维的特征
        self.fc_out = nn.Linear(gru_hidden_units, feature_dim)

    def forward(self, x):
        batch_size, C, T_MAX, H, W = x.shape  # (batch, C, T, H, W)

        # 3D CNN 提取时空特征
        x = F.relu(self.conv1(x))  # (batch, 16, T, H, W)
        x = self.pool(x)  # (batch, 16, T, H/2, W/2)
        x = F.relu(self.conv2(x))  # (batch, 32, T, H/2, W/2)

        # 展平
        x = x.view(batch_size, T_MAX, -1)  # (batch, T, feature_dim)
        x = F.relu(self.fc1(x))  # (batch, T, 128)

        # 通过 GRU 处理时间信息
        _, h_n = self.gru(x)  # h_n 形状为 (1, batch, gru_hidden_units)

        # 取最后时间步的 hidden state
        x = h_n[-1]  # (batch, gru_hidden_units)

        # 变换到 feature_dim 维度
        x = self.fc_out(x)  # (batch, feature_dim)
        return x

class CNN_RNN_Transformer_Encoder(nn.Module):
    def __init__(self, input_shape, feature_dim=512, cnn_filters=64, rnn_units=128, transformer_dim=128,
                 num_heads=4, num_layers=2, dropout_rate=0.5, use_mean_pooling=False):
        super(CNN_RNN_Transformer_Encoder, self).__init__()

        C, T_MAX, H, W = input_shape  # (C=1, T_MAX=时间步长, H=20, W=20)

        self.use_mean_pooling = use_mean_pooling  # 是否使用平均池化
        self.feature_dim = feature_dim  # 编码输出的维度

        # CNN部分：提取空间特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 计算 RNN 输入维度
        self.rnn_input_dim = cnn_filters * (20 // 2) * (20 // 2)
        self.lstm = nn.LSTM(self.rnn_input_dim, rnn_units, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Transformer 部分
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads,
                                                                    batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # 线性变换到 feature_dim
        self.fc_out = nn.Linear(transformer_dim, feature_dim)

    def forward(self, x):
        # x: 输入形状 [batch_size, T_MAX, 20, 20, 1]
        batch_size, C, T_MAX, H, W = x.shape  # (batch, C, T, H, W)
        x = x.view(batch_size * T_MAX, 1, H, W)  # [batch_size * T_MAX, 1, 20, 20]

        # CNN 部分：处理每个时间步的空间特征
        x = self.conv1(x)  # [batch_size * T_MAX, 64, 20, 20]
        x = self.pool(x)  # [batch_size * T_MAX, 64, 10, 10]
        x = self.dropout1(x)
        x = x.view(batch_size, T_MAX, -1)  # [batch_size, T_MAX, 6400]

        # RNN 部分：处理时序信息
        rnn_out, (hn, cn) = self.lstm(x)  # [batch_size, T_MAX, rnn_units]
        rnn_out = self.dropout2(rnn_out)

        # Transformer 部分：建模时序依赖
        transformer_out = self.transformer(rnn_out)  # [batch_size, T_MAX, transformer_dim]

        # 提取时序特征
        if self.use_mean_pooling:
            cls_output = torch.mean(transformer_out, dim=1)  # 平均池化
        else:
            cls_output = transformer_out[:, -1, :]  # 取最后一个时间步

        # 线性映射到 feature_dim
        output = self.fc_out(cls_output)  # [batch_size, feature_dim]

        return output

# ================== 3. 文本编码器 ==================
class TextEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("D:/Anaconda3/envs/clip_wifi/bert_base_uncased")

        for param in self.bert.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(768, feature_dim)  # 降维

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量
        x = self.fc(cls_embedding)
        return x

class CLIPWiFiTextModel(torch.nn.Module):
    def __init__(self, wifi_encoder, text_encoder):
        super().__init__()
        self.wifi_encoder = wifi_encoder
        self.text_encoder = text_encoder
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))  # 可学习参数

    def forward(self, wifi_data, text_input_ids, text_attention_mask):
        wifi_features = self.wifi_encoder(wifi_data)
        text_features = self.text_encoder(text_input_ids, text_attention_mask)

        # L2 归一化
        wifi_features = F.normalize(wifi_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # 计算可学习的 logit_scale
        logit_scale = self.logit_scale.exp()

        # 计算相似度 logits
        logits_per_image = logit_scale * (wifi_features @ text_features.t())  # WiFi → 文本
        logits_per_text = logits_per_image.t()  # 文本 → WiFi

        return logits_per_image, logits_per_text


def info_nce_loss(logits_per_wifi, logits_per_text):
    batch_size = logits_per_wifi.shape[0]
    labels = torch.arange(batch_size, device=logits_per_wifi.device).long()

    loss_wifi_to_text = F.cross_entropy(logits_per_wifi, labels)
    loss_text_to_wifi = F.cross_entropy(logits_per_text, labels)

    return (loss_wifi_to_text + loss_text_to_wifi) / 2

def train(model, dataloader, optimizer, device, epochs=100, scheduler=None):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            wifi_data, text_input_ids, text_attention_mask, _ = batch
            wifi_data, text_input_ids, text_attention_mask = wifi_data.to(device), text_input_ids.to(device), text_attention_mask.to(device)

            optimizer.zero_grad()

            logits_per_image, logits_per_text = model(wifi_data, text_input_ids, text_attention_mask)
            loss = info_nce_loss(logits_per_image, logits_per_text)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        if scheduler:
            scheduler.step()

def test(clip_model, dataloader, device, class_names, tokenizer):
    clip_model.eval()

    # 生成所有类别的文本特征
    class_labels = sorted(ALL_MOTION)
    prompts = [f"This is a motion of {number_to_gesture20181109[label]}." for label in class_labels]

    # 使用 tokenizer 处理类别文本
    encoded_inputs = tokenizer(prompts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            wifi_data, _, _, labels = batch  # 忽略文本输入
            wifi_data = wifi_data.to(device)
            labels = labels.to(device)

            # 获取 WiFi 特征
            logits_per_wifi, _ = clip_model(wifi_data, input_ids, attention_mask)  # 只计算 WiFi 特征

            # 对相似度矩阵进行 softmax 归一化
            probs = F.softmax(logits_per_wifi, dim=1)  # 按行进行 softmax

            # 预测类别
            preds = logits_per_wifi.argmax(dim=1)  # 预测类别索引
            pred_labels = torch.tensor([class_labels[p] for p in preds.cpu().numpy()]).to(device)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=class_labels)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=class_names, labels=class_labels)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", class_report)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# ================== 7. 训练初始化 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定是否使用已有模型
use_pretrained_model = False  # True 代表加载已有模型，False 代表训练新模型
# 加载 WiFi 数据
data_dir = 'D:/Codefield_python/CLIP_WIFI/Data/gesture1_6'  # 数据存储的目录。
ALL_MOTION = [1, 2, 3, 4, 5, 6]  # 所有动作的类别列表。
#fraction_for_test = 0.1  # 测试集所占比例。
num_classes = len(ALL_MOTION)  # 假设有6个类别
# 示例输入数据
batch_size = 64
T_MAX = 0  # 时间步数
C = 1 # WiFi 数据的通道数
H, W = 20, 20  # 每个时间步的空间维度
input_dim = 1  # 每个时间步的输入通道数
dropout_rate = 0.5  # Dropout 层的比例。

#label:1-6
train_data, test_data, train_labels, test_labels = load_data(data_dir, ALL_MOTION)
print("data finish")

# 设定 tokenizer
tokenizer = BertTokenizer.from_pretrained("D:/Anaconda3/envs/clip_wifi/bert_base_uncased")
print("tokenizer finish")

# 创建 Dataset 和 DataLoader
train_dataset = WiFiTextDataset(train_data, train_labels, tokenizer)
test_dataset = WiFiTextDataset(test_data, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("data_loader finish")

# 初始化编码器和优化器
wifi_encoder = CNN_RNN_Transformer_Encoder(input_shape=(C, T_MAX, H, W)).to(device)
text_encoder = TextEncoder().to(device)
print("encoder finish")

clip_model = CLIPWiFiTextModel(wifi_encoder, text_encoder)

optimizer = torch.optim.Adam(list(wifi_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-4)
print("optimizer finish")

# 加载或训练模型y

if use_pretrained_model:
    print("加载已有模型...")
    wifi_encoder.load_state_dict(torch.load("clip_wifi_encoder.pth", weights_only=True))
    text_encoder.load_state_dict(torch.load("clip_text_encoder.pth", weights_only=True))
    #optimizer.load_state_dict(torch.load("optimizer.pth"))
else:
    print("训练新模型...")

# 训练
train(clip_model, train_loader, optimizer, device)
print("train finish")

class_names = [f"Motion type {i}" for i in ALL_MOTION]  # 生成类别名称
test(clip_model, test_loader, device,class_names, tokenizer)
print("test finish")

torch.save(wifi_encoder.state_dict(), "clip_wifi_encoder.pth")
torch.save(text_encoder.state_dict(), "clip_text_encoder.pth")
#torch.save(optimizer.state_dict(), "optimizer1_6.pth")