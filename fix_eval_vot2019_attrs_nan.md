# 开启属性值绘制雷达图 & 修复`VOT2019`数据集在测试时遮挡/关照等属性值出现`NaN`的问题

## 1.开启 `attrs` 属性值显示以便后续绘制雷达图
找到`EAOBenchmark`的调用处`lib/eval_toolkit/bin/eval.py`, 增加`attrs`列表, 并使用`json`输出结果.

```python

# benchmark = EAOBenchmark(dataset)
# 修改后
benchmark = EAOBenchmark(dataset, tags=['all', 'occlusion', 'motion_change', 'size_change',
                                                        'illum_change', 'camera_motion', 'empty'])

eao_result = {}
with Pool(processes=args.num) as pool:
    for ret in tqdm(pool.imap_unordered(benchmark.eval,
        trackers), desc='eval eao', total=len(trackers), ncols=100):
        eao_result.update(ret)

# 输出结果
import json
print(json.dumps(eao_result, indent=2))
```

最终效果:

```shell
{
  "Oceancheckpoint_eOcean": {
    "all": 0.32700964729718074,
    "occlusion": 0.3408137072877186,
    "motion_change": 0.1690339541471586,
    "size_change": 0.5073183149826236,
    "illum_change": 0.5095810628518825,
    "camera_motion": 0.4606175519586579,
    "empty": 0.05665934655239912
  }
}
------------------------------------------------------------------------
|      Tracker Name      | Accuracy | Robustness | Lost Number |  EAO  |
------------------------------------------------------------------------
| Oceancheckpoint_eOcean |  0.591   |   0.376    |    75.0     | 0.327 |
------------------------------------------------------------------------
```

## 2.属性值出现`NaN`的原因
### (1).`VOT2019.json`文件错误
由于`pysot-toolkit`中提供的`VOT2019.json`文件中涉及`attrs`的值都为`0`, 因此得到的属性计算结果必然是错误的.

### (2).`VOT2019`数据集文件中`graduate`序列相关`.tag`标注文件错误
由于`motion_change.tag`,`occlusion.tag`, `size_change.tag`三个文件只有`768`行, 与该序列其他`.tag`文件行数`864`不一致, 导致`numpy`构建数组时因为维度不一致而出错.

以下为错误堆栈信息:

```shell
loading VOT2019:  37%|█████████████▏                      | 22/60 [00:01<00:02, 16.30it/s, graduate]Traceback (most recent call last):
  File "/data/trackit_nonlocal_light_cascade_mask_second/lib/eval_toolkit/bin/eval.py", line 135, in <module>
    dataset = VOTDataset(args.dataset, root)
  File "/data/TracKit/lib/eval_toolkit/pysot/datasets/vot.py", line 125, in __init__
    load_img=load_img)
  File "/data/TracKit/lib/eval_toolkit/pysot/datasets/vot.py", line 51, in __init__
    self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
TypeError: unsupported operand type(s) for -: 'int' and 'list'
```

## 2.解决方法
### (1).从文件中重新读取`tag`文件, 并生成`VOT2019.json`
参考代码为:
```python
import json
import os

# 数据集路径
path = '/data/TracKit/dataset/'

# 读取
with open(os.path.join(path, "VOT2019.json"), 'r') as f:
    data = json.load(f)

# 相关属性
attrs = ['camera_motion', 'illum_change', 'motion_change', 'size_change', 'occlusion']

# 遍历所有序列
for key in data.keys():
    for attr in attrs:
        tag_file = os.path.join(path, key, attr + '.tag')
        # 不存在该属性文件则跳过
        if not os.path.isfile(tag_file):
            continue

        list = []
        with open(tag_file, 'r') as f:
            for line in f.readlines():
                list.append(int(line))

        data[key][attr] = list

# 写入结果
# 注意: 会覆盖原文件!!!
with open(os.path.join(path, "VOT2019.json"), 'w') as f:
    f.write(json.dumps(data))

print("finished!")
```

已处理好的 `VOT2019.json` 文件下载见 [fix-VOT2019.json](fix_vot2019_json_file/VOT2019.json)

### (2).修改文件 `lib/eval_toolkit/pysot/datasets/vot.py`
在约32行处找到`VOTVideo`的`__init__`函数, 并找到如下代码:

```python
# empty tag
all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
```

在其之前加入补齐每一个 `tags` 长度的代码, 如下:

```python
# 让self.tags的每一个值长度都对齐为标签的长度
for v in self.tags.values():
    if len(v) != len(gt_rect):
        v += [0] * (len(gt_rect) - len(v))

# empty tag
all_tag = [v for k, v in self.tags.items() if len(v) > 0 ]
self.tags['empty'] = np.all(1 - np.array(all_tag), axis=1).astype(np.int32).tolist()
```
