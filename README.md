# WifiCLIP

---
Currently, we provide deep learning modeling code and preprocessed datasets.

## instruction
这是一个 WiFi-Text CLIP 模型，旨在对齐 WiFi 信号中的 **BVP（Body-coordinate Velocity Profile）** 数据
与文本描述，以实现 **zero-shot 学习**。该模型利用 **CNN+RNN+Transformer 结构作为 WiFi 数据编码器**，
并采用 **BERT 作为文本编码器**，从而在 WiFi 传感数据与自然语言之间建立高效的跨模态表示映射。  

This is a **WiFi-Text CLIP model** designed to align **BVP (Body-coordinate Velocity Profile)** data in WiFi signals
with text descriptions to enable **zero-shot learning**. The model utilizes the **CNN+RNN+Transformer structure as the WiFi data encoder** and
adopts **BERT as the text encoder** to establish an efficient cross-modal representation mapping between WiFi sensing data and natural language.

## Installation Requirements

Recommended to use **conda** installation.

    # Creating the Environment（Python 3.10）
    conda create -n my_env python=3.10
    conda activate my_env
    
    # Installation of the Scientific Computing Foundation Library
    conda install numpy scipy pandas matplotlib scikit-learn seaborn -c conda-forge
    
    # Installing PyTorch（GPU，CUDA 12.1）
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    
    # Installing the HuggingFace library
    pip install transformers==4.48.0

## Dataset Download

The BVP data for the corresponding gesture exists in the Data folder, please download the original CSI data from the following link:

<https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset>

## Pre-training with data

### To use the current dataset with the training model, run

> python clip.py





