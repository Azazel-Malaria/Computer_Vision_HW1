# Fashion-MNIST 三层 MLP 作业模板（NumPy + 手写自动微分）

这个项目对应的作业要求包括：

- 不使用 PyTorch / TensorFlow / JAX 等自动微分框架；
- 代码至少包含：数据加载与预处理、模型定义、训练循环、测试评估、超参数查找 五个模块；
- 支持 SGD、学习率衰减、交叉熵、L2 正则化；
- 根据验证集准确率自动保存最优权重；
- 支持导入最优权重在测试集上评估，并输出混淆矩阵；
- 额外提供第一层权重可视化、错例可视化，方便完成实验报告。

## 目录结构

```text
fashion_mnist_mlp_hw1/
├── README.md
├── requirements.txt
├── run_train.py
├── run_search.py
├── run_test.py
├── run_visualize.py
└── mlp
    ├── __init__.py
    ├── data.py
    ├── evaluate.py
    ├── layers.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    ├── optim.py
    ├── tensor.py
    ├── trainer.py
    └── utils.py
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 1. 训练模型

```bash
python run_train.py \
  --data_dir ./data \
  --output_dir ./outputs/run1 \
  --epochs 30 \
  --batch_size 128 \
  --learning_rate 0.08 \
  --lr_decay_gamma 0.95 \
  --weight_decay 1e-4 \
  --hidden_dims 256 128 \
  --activation relu
```

训练结束后会在 `output_dir` 下得到：

- `best_model.npz`：验证集最优权重
- `config.json`：本次训练配置
- `history.json`：训练过程记录
- `loss_curve.png`：训练/验证 Loss 曲线
- `val_accuracy_curve.png`：验证集 Accuracy 曲线
- `summary.json`：最佳验证结果

## 2. 超参数搜索

```bash
python run_search.py \
  --data_dir ./data \
  --search_dir ./outputs/grid_search \
  --epochs 20
```

该脚本会遍历多组学习率、隐藏层大小、正则化强度和激活函数组合，并保存每次试验结果，最终生成：

- `search_results.json`

## 3. 测试集评估

```bash
python run_test.py \
  --data_dir ./data \
  --checkpoint ./outputs/run1/best_model.npz \
  --config ./outputs/run1/config.json \
  --output_dir ./outputs/test_eval
```

输出内容：

- 测试集准确率
- 混淆矩阵（控制台打印，同时保存到 `confusion_matrix.txt`）

## 4. 权重可视化与错例分析

```bash
python run_visualize.py \
  --data_dir ./data \
  --checkpoint ./outputs/run1/best_model.npz \
  --config ./outputs/run1/config.json \
  --output_dir ./outputs/visuals
```

输出内容：

- `first_layer_weights.png`：第一层隐藏层权重恢复成 28×28 图像后的可视化
- `misclassified_examples.png`：测试集中的若干错分样本

## 自动微分实现说明

本项目在 `mlp/tensor.py` 中手写了一个简化版 `Tensor` 类，实现了：

- 计算图记录
- `backward()` 反向传播
- 加减乘除、矩阵乘法、sum、mean
- ReLU / Sigmoid / Tanh

交叉熵损失在 `mlp/losses.py` 中实现为一个自定义的可反传节点。

## 你写实验报告时可以直接使用的结果

本代码已经覆盖了报告里最常见的素材输出：

- 训练集 / 验证集 Loss 曲线
- 验证集 Accuracy 曲线
- 第一层权重可视化
- 错误分类样本图
- 测试集混淆矩阵

## 建议你在 GitHub README 里额外补充

- 你最终采用的最佳超参数
- 测试集最终准确率
- 模型权重下载链接（如 Google Drive）
- 实验现象与误差分析
