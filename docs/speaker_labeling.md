# Real-time Speaker Labeling

FunASR 支持实时说话人标注功能，可以在语音识别过程中动态标注说话人身份。本文档介绍如何使用此功能。

## 功能特性

- 实时说话人特征提取
- 动态说话人档案管理
- 支持更新说话人名称
- 性能优化（每个特征提取耗时 <10ms）
- 支持 CPU 和 CUDA 设备
- 与现有说话人分类功能集成

## 使用方法

### 基本用法

```python
from funasr.auto.auto_model import AutoModel

# 初始化模型
model = AutoModel(
    model="SherpaEmbedding",  # 或使用 CAMPPlus
    model_path="path/to/model",
    speaker_similarity_threshold=0.75,  # 可选，默认值
    device="cuda"  # 或 "cpu"
)

# 识别结果会自动包含说话人标识
results = model.generate(audio)
# results 中会包含 "speaker" 字段，格式如 "user1", "user2" 等

# 可以随时更新说话人名称
model.update_speaker_name("user1", "张三")
```

### 高级配置

```python
# 使用 CAMPPlus 模型
model = AutoModel(
    model="CAMPPlus",
    model_path="path/to/model",
    spk_mode="punc_segment",  # 支持 "default", "vad_segment", "punc_segment"
    device="cuda"
)

# 配置说话人管理器
model.speaker_manager.similarity_threshold = 0.8  # 调整相似度阈值
```

## 性能优化

1. 缓存优化
   - 使用 LRU 缓存最近的说话人特征
   - 可配置缓存大小
   - 自动管理缓存生命周期

2. CUDA 加速
   - 支持 GPU 加速特征提取
   - 批处理相似度计算
   - 优化内存布局

3. 集成优化
   - 与 VAD 和时间戳功能集成
   - 支持分段处理
   - 平滑处理重叠区域

## 注意事项

1. 设备选择
   - CPU 模式适合低资源环境
   - CUDA 模式提供更好性能
   - 自动处理设备间数据传输

2. 内存管理
   - 自动管理缓存大小
   - 及时释放不需要的资源
   - 支持批量处理优化

3. 集成注意事项
   - 与现有说话人分类功能兼容
   - 支持平滑过渡
   - 保持一致的说话人标识

## API 参考

### AutoModel

主要配置参数：
- `model`: 选择模型类型 ("SherpaEmbedding" 或 "CAMPPlus")
- `model_path`: 模型路径
- `speaker_similarity_threshold`: 说话人相似度阈值
- `device`: 运行设备 ("cpu" 或 "cuda")
- `spk_mode`: 说话人分段模式

主要方法：
- `generate()`: 生成识别结果，包含说话人标识
- `update_speaker_name()`: 更新说话人名称

### SpeakerManager

主要方法：
- `get_speaker_id()`: 获取说话人标识
- `update_speaker_name()`: 更新说话人名称
- `process_segments()`: 处理说话人分段

## 示例

### 实时识别示例

```python
import torch
from funasr.auto.auto_model import AutoModel

# 初始化模型
model = AutoModel(
    model="SherpaEmbedding",
    model_path="path/to/model",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 处理音频
audio = "path/to/audio.wav"
results = model.generate(audio)

# 处理结果
for result in results:
    print(f"说话人: {result['speaker']}")
    print(f"文本: {result['text']}")
```

### 批量处理示例

```python
# 批量处理多个音频
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
for audio in audio_files:
    results = model.generate(audio)
    for result in results:
        print(f"文件: {audio}")
        print(f"说话人: {result['speaker']}")
        print(f"文本: {result['text']}")
```
