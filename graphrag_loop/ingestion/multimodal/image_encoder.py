"""图片特征提取 —— CLIP / ViT → 视觉 embedding。

# TODO(Phase 3:多模态扩展)
当前状态:占位,未实现。
触发条件:Phase 3 接入图片数据时启用。

实现要点:
  - open-clip-torch 加载 ViT-B/32 或更大模型
  - encode_image(path) → 512/768 维向量(与 bge-m3 文本侧统一空间)
  - 入 Qdrant 时 payload 加 {type: "image", source_uri: "minio://..."}
  - 跨模态检索:文搜图 = 文本向量在图片向量集合里找最相似
"""
