# TensorBoard 可视化指南

本指南介绍如何使用TensorBoard查看模型训练过程的可视化结果。

## 安装依赖

已在项目的`requirements.txt`文件中添加了TensorBoard依赖，请确保安装：

```bash
pip install -r requirements.txt
```

## 启动TensorBoard

在训练模型后，可以使用以下命令启动TensorBoard服务器：

```bash
tensorboard --logdir=runs
```

这将启动TensorBoard服务器，默认情况下可以通过浏览器访问 http://localhost:6006 查看可视化结果。

## 可视化内容

### Transformer模型

Transformer模型的训练过程中，TensorBoard记录了以下指标：

1. **训练指标**
   - Training/Loss：每个epoch的平均训练损失
   - Training/LearningRate：学习率变化

2. **评估指标**
   - Evaluation/Precision：精确率
   - Evaluation/Recall：召回率
   - Evaluation/F1：F1分数

3. **混淆矩阵**
   - Initial/ConfusionMatrix：初始模型的混淆矩阵
   - Validation/ConfusionMatrix：每个epoch验证集的混淆矩阵
   - Test/ConfusionMatrix：最终测试集的混淆矩阵

4. **概率分布**
   - Initial/Probabilities：初始模型的预测概率分布
   - Validation/Probabilities：验证集的预测概率分布
   - Test/Probabilities：测试集的预测概率分布

### Diffusion模型

Diffusion模型的训练过程中，TensorBoard记录了与Transformer模型类似的指标，包括训练损失、评估指标、混淆矩阵和概率分布。

## 使用技巧

1. **比较不同模型**：可以同时查看Transformer和Diffusion模型的训练结果，比较它们的性能。

2. **查看特定指标**：在TensorBoard界面左侧选择要查看的指标类别（如SCALARS、IMAGES等）。

3. **平滑曲线**：使用TensorBoard界面中的平滑滑块调整曲线的平滑程度，使趋势更容易观察。

4. **导出数据**：TensorBoard允许导出数据为CSV格式，方便进一步分析。

## 注意事项

- 训练过程中会在`runs`目录下生成日志文件，这些文件可能会占用一定的磁盘空间。
- 如果需要清除历史记录，可以删除`runs`目录下的相应文件夹。