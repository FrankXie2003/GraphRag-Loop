"""对象存储封装(MinIO / S3 兼容)—— 原始图片 / 视频 / 音频大文件存取。

# TODO(Phase 3:多模态扩展)
当前状态:占位,未实现。
触发条件:Phase 3 引入多模态数据(图片/音频/视频)时启用。文本场景不需要。

实现要点(等 Phase 3 真做时):
  - minio-py 客户端(.env 已预留 MINIO_* 配置位)
  - put_object(stream) → URI;get_presigned_url(uri) 给前端访问
  - Neo4j Event 节点的 attachments 字段存 URI 列表(原文件不进图,只存指针)
  - 详见主 README §4 多模态存储架构表
"""
