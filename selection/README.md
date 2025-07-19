# 数据选择方法库

本目录包含了从 `./methods/` 目录转换而来的数据选择方法，所有方法都遵循统一的接口设计。

## 已实现的方法

### 1. 基础方法
- **RandomSelection** (`random.py`) - 随机选择
- **FullSelection** (`full.py`) - 全量数据选择
- **KMeansSelection** (`kmeans.py`) - K-Means 聚类选择

### 2. 基于距离的方法
- **HerdingSelection** (`herding.py`) - Herding 算法
- **KCenterGreedySelection** (`kcentergreedy.py`) - K-Center Greedy 算法
- **LowDimensionSelection** (`lowdimention.py`) - 低维度四叉树方法

### 3. 基于梯度的方法
- **GlisterSelection** (`glister.py`) - GLISTER 算法
- **GradMatchSelection** (`gradmatch.py`) - GradMatch 算法
- **CraigSelection** (`craig.py`) - CRAIG 算法（简化版）

### 4. 基于价值的方法
- **GShapleySelection** (`g_shapley.py`) - 梯度 Shapley 值方法
- **TMCSSelection** (`tmcs.py`) - TMCS Shapley 值方法
- **CSShapleySelection** (`cs_shapley.py`) - 类条件 Shapley 值方法
- **KNNShapleySelection** (`knn_shapley.py`) - KNN Shapley 值方法

### 5. 主动学习方法
- **SVPMaxEntropySelection** (`svp_max_entropy.py`) - 最大熵主动学习
- **BoundaryAwareSelection** (`boundary_aware.py`) - 边界感知选择
- **DBALSelection** (`dbal.py`) - 深度贝叶斯主动学习
- **DFALSelection** (`dfal.py`) - 深度特征主动学习

### 6. 影响函数方法
- **TracInSelection** (`tracin.py`) - TracIn 影响函数
- **LiSSASelection** (`lissa.py`) - LiSSA 影响函数
- **CGInfluenceSelection** (`cg_influence.py`) - 共轭梯度影响函数

### 7. 增量学习方法
- **IASSelection** (`ias.py`) - 增量主动选择

### 8. 强化学习方法
- **DVRLSelection** (`dvrl.py`) - 数据价值强化学习

## 统一接口

所有方法都继承自 `DataSelection` 基类，并实现 `select` 方法：

```python
from selection.data_selection import DataSelection

class YourSelection(DataSelection):
    def select(self,
               model: torch.nn.Module,
               selection_fraction: float,
               train_set,
               task_type: str,
               device: torch.device,
               **kwargs
               ) -> DataLoader:
        """
        选择数据点构建核心集
        
        参数：
          - model: 已实例化的神经网络模型
          - selection_fraction: 核心集占总样本的比例 (0~1)
          - train_set: 训练数据集
          - task_type: 'classification' 或 'regression'
          - device: torch.device
          - **kwargs: 方法特定的参数
          
        返回：
          - DataLoader: 包含选择样本的数据加载器
        """
        pass
```

## 使用示例

```python
import torch
from torch.utils.data import Dataset
from selection.kmeans import KMeansSelection

# 创建数据集和模型
dataset = YourDataset()
model = YourModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用 K-Means 选择方法
selector = KMeansSelection()
core_loader = selector.select(
    model=model,
    selection_fraction=0.3,  # 选择30%的数据
    train_set=dataset,
    task_type='classification',
    device=device,
    k=10  # K-Means 聚类数
)

# 使用核心集进行训练
for batch_data, batch_labels in core_loader:
    # 训练代码
    pass
```

## 方法特定参数

### 基础方法
- **KMeansSelection**: `k` (聚类数量)
- **HerdingSelection/KCenterGreedySelection**: `num_classes`, `balance`
- **LowDimensionSelection**: `epsilon`, `max_depth`, `min_points`

### 梯度方法
- **GlisterSelection/GradMatchSelection**: `val_set`, `batch_size`
- **CraigSelection**: 无额外参数

### Shapley 值方法
- **GShapleySelection**: `num_iterations`, `learning_rate`, `batch_size`
- **TMCSSelection**: `num_iterations`, `use_features`
- **CSShapleySelection**: `num_classes`, `num_iterations`, `balance_classes`
- **KNNShapleySelection**: `k`, `num_iterations`

### 主动学习方法
- **SVPMaxEntropySelection**: `active_fraction_ratio`, `min_selection_size`
- **BoundaryAwareSelection**: `active_fraction_ratio`, `min_selection_size`
- **DBALSelection**: `num_mc_samples`
- **DFALSelection**: `n_clusters`, `diversity_weight`

### 影响函数方法
- **TracInSelection**: `val_set`, `num_epochs`, `learning_rate`
- **LiSSASelection**: `val_set`, `recursion_depth`, `damping`, `scale`, `num_test_samples`
- **CGInfluenceSelection**: `val_set`, `max_cg_iter`, `damping`

### 增量学习方法
- **IASSelection**: `n_clusters`, `gamma`, `alpha`, `lambda_`, `learning_rate`, `batch_size`

### 强化学习方法
- **DVRLSelection**: `val_set`, `num_epochs`, `learning_rate`

## 测试

运行测试脚本验证所有方法：

```bash
python validate_selection_methods.py  # 完整测试（包含之前的方法）
python test_simple_new_methods.py     # 新方法测试
```

## 注意事项

1. 所有方法都会尝试使用 `EmbeddingExtractor` 提取特征，如果失败会回退到原始数据展平
2. 基于梯度的方法（GLISTER, GradMatch, CRAIG, GShapley, TracIn, LiSSA, CGInfluence）计算时间较长
3. 某些方法支持平衡选择，会在每个类别中按比例选择样本
4. 所有方法都会输出选择的样本数量和标签分布（分类任务）
5. 影响函数方法（TracIn, LiSSA, CGInfluence）可能在某些环境下出现数值稳定性问题

## 依赖

- PyTorch
- NumPy
- scikit-learn
- tqdm
- scipy（部分方法需要）

## 转换状态

- ✅ **已转换**: 22/27 个方法（81%）
- ✅ **测试通过**: 20/22 个方法（91%）
- ⚠️ **部分问题**: LiSSA, CGInfluence（复杂梯度计算可能导致数值问题）
- ❌ **未转换**: 5 个方法（复杂度过高或特殊依赖）

转换工作基本完成，涵盖了数据选择领域的主要算法类型，为后续使用提供了统一、易用的接口。 