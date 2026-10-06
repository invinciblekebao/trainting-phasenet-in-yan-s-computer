# 新训练数据的格式要求

以下要求来自本包实际使用的 `eqnet/data/seismic_trace.py`，不代表所有 NPZ/HDF5 文件都能直接读取。NPZ 与 HDF5 的标签属性需要与这些约定一致。

## NPZ

| 字段 | 要求 |
|---|---|
| `data` | 二维、三分量波形，`(Nt,3)` 或 `(3,Nt)`，顺序 E/N/Z；读取后转换为 float32 |
| `itp` | 一个整数采样点索引，P 到时，以原始波形起点为 0 |
| `its` | 一个整数采样点索引，S 到时，满足 `0 <= itp < its < Nt` |
| `channels` | 可省略；如提供，使用 `E_N_Z`、`ENZ` 或 `E,N,Z` 的标量字符串 |

示例（波形和标签由你的数据提供）：

```python
np.savez(output_path, data=waveform.astype(np.float32),
         itp=np.int64(p_sample), its=np.int64(s_sample), channels="E_N_Z")
```

不接受 NaN/Inf；三个分量不能全部为常量。`itp/its` 是采样点，不是秒。100 Hz 时 10 秒对应索引 1000。

清单是无表头 CSV/TXT，每行第一列为一个路径；相对路径相对于清单所在目录。训练、验证、测试清单分别制作，最好按事件划分，避免同一地震事件跨集合。

读取器先逐分量对完整波形去均值、除标准差；训练时随机选择同时包含 P/S 的 4096 点窗，验证用确定性裁窗。较短波形右侧补零。P/S 间距必须允许两者同时落入窗口，读取器不会猜测或修改错误标签。

## HDF5

保持现有 DiTing 适配后的分层结构：

```text
train.hdf5
  <event_id>/                       HDF5 Group
    <station_dataset>/              Dataset，(Nt,3) 或 (3,Nt)，E/N/Z
```

TXT 清单每行是 `<event_id>/<station_dataset>`，无表头。例如：

```text
372/372_819
402/402_682
```

**Dataset 属性：**

| 属性 | 内容 |
|---|---|
| `phase_index` | 震相到时的整数采样点索引数组，不能超出当前波形 |
| `phase_type` | 与索引同长度的类型数组，按读取器字符串约定写 `P`、`S` |
| `event_id` | 与每个震相相对应的事件标识数组，能够匹配文件内事件组 |
| `dt_s` | 采样间隔；本包默认 100 Hz，对应 0.01 秒 |
| `latitude`、`longitude` | 台站坐标 |
| `elevation_m` | 台站海拔，米 |
| `phase_polarity` | 可选极性数组，读取器接受 `U` / `D`；P/S 单任务不要求极性监督 |

**事件 Group 属性：**

| 属性 | 内容 |
|---|---|
| `event_time` | 可由 `datetime.fromisoformat()` 解析的事件时间字符串 |
| `event_time_index` | 事件起时相对于波形的采样点索引 |
| `latitude`、`longitude` | 事件坐标 |
| `depth_km` | 事件深度，千米 |

现有读取器即使训练 P/S，也会构造事件及位置相关字段，所以上述元数据需要保留。仅把 `data/itp/its` 放进任意 HDF5 文件，并不等于现有读取器支持该结构；如果你的新数据只有这三项，使用 NPZ 最直接。

使用明确的 trace 清单启动，并关闭 SNR 过滤，沿用近期实验行为。输入波形已经是目标采样率；这里的训练流程不会自动把任意采样率转换成 100 Hz。

## 模型实际收到的输入

- `data`：`(B,3,4096,1)`、float32，三分量 E/N/Z。
- `phase_pick`：`(B,3,4096,1)`、float32，通道为 Noise/P/S。
- P/S 使用宽度 40 点的高斯软标签，sigma 为 `40/6` 点；Noise 由背景概率构造。
- 输出同尺寸三类 logits；损失沿用逐采样点软标签交叉熵，先对 Noise/P/S 求和，再取均值。

NPZ 读取和 HDF5 的采样/裁窗方式仍各自沿用原实现：NPZ 每轮遍历清单，HDF5 按清单随机有放回抽样并裁窗，二者不是完全相同的样本访问策略。
