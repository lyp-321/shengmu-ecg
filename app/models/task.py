"""
ECG 任务数据库模型
"""
from tortoise import fields
from tortoise.models import Model


class ECGTask(Model):
    id = fields.IntField(pk=True)
    filename = fields.CharField(max_length=255)
    file_path = fields.CharField(max_length=500)
    status = fields.CharField(max_length=20, default="pending")  # pending/completed/failed
    result = fields.JSONField(null=True)
    report_path = fields.CharField(max_length=500, null=True)
    error_message = fields.TextField(null=True)
    user = fields.ForeignKeyField("models.User", related_name="tasks")
    created_at = fields.DatetimeField(auto_now_add=True)
    completed_at = fields.DatetimeField(null=True)

    class Meta:
        table = "ecg_tasks"
