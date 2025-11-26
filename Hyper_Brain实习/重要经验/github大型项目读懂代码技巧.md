# 1.Registry类的简明解释与使用指南

## 1.1Registry类是什么？

Registry类是一个注册表，就像一本字典，它帮助程序把字符串名称（如"DentalDataset"）与实际的类（如DentalDataset类）关联起来。可以把它想象成一个智能电话簿，输入名字就能找到对应的电话号码。

## 1.2为什么需要Registry类？

### 1） 解耦配置与代码

没有Registry时，如果你想切换使用不同的数据集，你需要修改代码：

```python
# 没有Registry的情况
if dataset_name == "S3DIS":
    dataset = S3DISDataset(...)
elif dataset_name == "ScanNet":
    dataset = ScanNetDataset(...)
elif dataset_name == "Dental":
    dataset = DentalDataset(...)
# 每添加一个新数据集，就要修改这段代码
```

有了Registry，你只需在配置文件中指定名称即可：

```python
# 配置文件中
dataset_type = "DentalDataset"
# 程序会自动查找并使用你注册的DentalDataset类
```

### 2）实现"插件式"架构

想象一下装模块的积木玩具，Registry允许你开发新的模块（如新的数据集），只需"插入"到系统中，无需修改原有代码。这使得框架非常灵活和可扩展。

### 3）避免循环导入

在大型项目中，模块之间的导入关系可能很复杂。Registry帮助避免循环导入问题，因为你可以在一个中心位置注册所有类，而不必在每个需要的地方导入它们。

## 1.3不使用Registry会怎么样？

不使用Registry会导致：

1. 硬编码：每次添加新模块都需要修改核心代码

1. 扩展性差：难以支持第三方插件和扩展

1. 配置繁琐：无法通过简单的配置文件切换组件

1. 代码耦合：组件之间的依赖关系更紧密

## 1.4如何使用Registry注册自己的数据集？

### 1）创建并注册自定义数据集

```python
# 在pointcept/datasets/dental.py文件中
import torch
from torch.utils.data import Dataset
from pointcept.utils.registry import DATASETS  # 导入已定义的注册表

@DATASETS.register_module()  # 关键步骤：注册你的类
class DentalDataset(Dataset):
    def __init__(self, data_root, split, transform=None, ...):
        # 数据集初始化代码
        self.data_root = data_root
        self.split = split
        # 其他初始化...
    
    def __getitem__(self, idx):
        # 获取单个数据样本的代码
        # ...
        return data_dict
    
    def __len__(self):
        # 返回数据集大小
        return len(self.samples)
```

### 2）将你的数据集类导入到注册表系统

在pointcept/datasets/__init__.py中添加导入：

```python
# pointcept/datasets/__init__.py
from .dental import DentalDataset  # 导入你的数据集类
```

### 3）在配置文件中使用你的数据集

```python
# configs/dental/pt-v3m1-base-dental.py
dataset_type = 'DentalDataset'  # 使用注册的名称
data_root = './data/dental_data'

data = dict(
    train=dict(
        type=dataset_type,  # 这里引用上面定义的dataset_type
        split='train',
        data_root=data_root,
        # 其他参数...
    ),
    val=dict(
        type=dataset_type,
        split='val',
        data_root=data_root,
        # 其他参数...
    )
)
```



# 2.hooks

插入hooks到训练周期通过以下机制实现：
1）配置声明：在runtime配置中定义hooks列表

```python
hooks = [  
    dict(type="CheckpointLoader"),  
    dict(type="CustomHook")  # 添加自定义hook  
]
```

2）自动装配：训练器初始化时通过Registry机制自动构建hook实例

```python
@HOOKS.register_module()  
class CustomHook:  
    def before_run(self, runner):  
        print("Training starts!")
```

3）生命周期绑定：每个hook类实现特定方法（如before_run/after_epoch），训练器在对应阶段自动调用

```python
for epoch in epochs:  
    hook.before_epoch()  # 如CheckpointSaver检查保存条件  
    for batch in data:  
        hook.before_iter()  # 如IterationTimer开始计时  
        model.train_step()  
        hook.after_iter()   # 如InformationWriter记录loss  
    hook.after_epoch()      # 如SemSegEvaluator执行验证
```

