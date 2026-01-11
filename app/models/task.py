"""
数据库模型定义 - Tortoise ORM
"""
from tortoise import fields
from tortoise.models import Model


class ECGTask(Model):
    """ECG 分析任务模型"""
    
    id = fields.IntField(pk=True)
    filename = fields.CharField(max_length=255)
    file_path = fields.CharField(max_length=512)
    status = fields.CharField(max_length=50, default="pending")  # pending, processing, completed, failed
    result = fields.JSONField(null=True)
    error_message = fields.TextField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)
    
    # 用户关联
    user = fields.ForeignKeyField("models.User", related_name="ecg_tasks", null=True)
    
    class Meta:
        table = "ecg_tasks"
    
    def __str__(self):
        return f"ECGTask(id={self.id}, filename={self.filename}, status={self.status})"
