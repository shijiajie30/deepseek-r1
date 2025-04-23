# DeepSeek-R1 Project

## 项目简介
DeepSeek-R1 是一个用于文本生成的项目，利用微调后的语言模型进行推理和生成。

## 安装步骤
1. 克隆项目到本地：
   ```bash
   git clone <repository-url>
   ```
2. 安装依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 使用说明
1. 运行 `main.py` 以进行模型训练：
   ```bash
   python main.py
   ```
2. 使用 `inference.py` 进行文本生成：
   ```bash
   python inference.py
   ```

## 项目结构
- `main.py`: 负责模型的训练和保存。
- `inference.py`: 用于加载微调后的模型并进行推理。
- `data_prepare.py`: 数据准备模块。
- `datasets.jsonl`: 训练数据集。
- `saved_model/`: 保存微调后的模型。

## 贡献指南
欢迎贡献代码和提出建议，请提交 Pull Request 或 Issue。

## 许可证信息
本项目采用 MIT 许可证。