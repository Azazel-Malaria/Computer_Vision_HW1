# Fashion-MNIST 三层 MLP

## 目录结构

```text
hw1/
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
其中手动实现自动微分、激活函数、SGD等方法需要的函数/类均在./mlp文件夹中
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
最终生成：

- `search_results.json`

## 3. 测试集评估

```bash
python run_test.py \
  --data_dir ./data \
  --checkpoint ./outputs/run1/best_model.npz \
  --config ./outputs/run1/config.json \
  --output_dir ./outputs/test_eval
```

输出测试集准确率、混淆矩阵（保存到 `confusion_matrix.txt`）

## 4. 权重可视化与错例分析

```bash
python run_visualize.py \
  --data_dir ./data \
  --checkpoint ./outputs/run1/best_model.npz \
  --config ./outputs/run1/config.json \
  --output_dir ./outputs/visuals
```

输出：

- `first_layer_weights.png`：第一层隐藏层权重恢复成 28×28 图像后的可视化
- `misclassified_examples.png`：测试集中的若干错分样本
