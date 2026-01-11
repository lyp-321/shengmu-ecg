"""
用户数据模型 - Tortoise ORM
"""
from tortoise import fields
from tortoise.models import Model


class User(Model):
    """用户模型"""
    
    id = fields.IntField(pk=True)
    username = fields.CharField(max_length=50, unique=True, index=True)
    email = fields.CharField(max_length=100, unique=True, index=True)
    hashed_password = fields.CharField(max_length=255)
    full_name = fields.CharField(max_length=100, null=True)
    role = fields.CharField(max_length=20, default="user")  # admin 或 user
    is_active = fields.BooleanField(default=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta:
        table = "users"
    
    def __str__(self):
        return f"User(id={self.id}, username={self.username}, role={self.role})"
