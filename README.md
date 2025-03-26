# WifiCLIP

---
当前，我们提供深度学习模型代码和预处理后的数据集。

## 介绍
这是一个 WiFi-Text CLIP 模型，旨在对齐 WiFi 信号中的 **BVP（Body-coordinate Velocity Profile）** 数据
与文本描述，以实现 **zero-shot 学习**。该模型利用 **CNN+RNN+Transformer 结构作为 WiFi 数据编码器**，
并采用 **BERT 作为文本编码器**，从而在 WiFi 传感数据与自然语言之间建立高效的跨模态表示映射。

## 安装要求

推荐使用conda安装。

    # 创建环境（Python 3.10）
    conda create -n my_env python=3.10
    conda activate my_env
    
    # 安装科学计算基础库
    conda install numpy scipy pandas matplotlib scikit-learn seaborn -c conda-forge
    
    # 安装PyTorch（GPU版，CUDA 12.1）
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    # 安装HuggingFace库
    pip install transformers==4.48.0

## 数据集下载

于Data文件夹中存在对应手势的BVP数据，具体原始CSI数据请从以下链接下载：

<https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset>

## 使用数据进行预训练

### 使用当前数据集与训练模型，请运行

> python clip.py





