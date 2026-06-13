"""音频特征 —— Whisper 转写 + 音频 embedding。

# TODO(Phase 3:多模态扩展)
当前状态:占位,未实现。
触发条件:Phase 3 接入音频数据(语音 / 配乐)时启用。

实现要点:
  - openai-whisper 转写 → 文本走文本侧 embedding(复用 bge-m3)
  - (可选)wav2vec2 提音频 embedding,做"按声音特征"检索
  - 大多数场景"转写后走文本"就够,纯音频特征仅在音乐 / 环境音识别时需要
"""
