"""视频特征 —— 抽帧 + CLIP,或专用视频模型。

# TODO(Phase 3:多模态扩展)
当前状态:占位,未实现。
触发条件:Phase 3 接入视频数据时启用。

实现要点:
  - opencv-python 抽关键帧(ffmpeg 关键帧检测 / 固定间隔)
  - 关键帧走 image_encoder.py 出向量
  - 多帧聚合策略:平均池化 / 取 top-K 代表帧 / 用专用视频模型(如 VideoCLIP)
  - 配音轨走 audio_encoder.py
"""
