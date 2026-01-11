"""
用户管理 API 路由（管理员）
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List

from app.schemas.user import UserResponse
from app.models.user import User
from app.core.deps import require_admin

router = APIRouter()


@router.get("", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    admin: User = Depends(require_admin)
):
    """
    获取用户列表（管理员）
    """
    users = await User.all().offset(skip).limit(limit)
    return users


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    admin: User = Depends(require_admin)
):
    """
    获取用户详情（管理员）
    """
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    return user


@router.put("/{user_id}/toggle", response_model=UserResponse)
async def toggle_user_status(
    user_id: int,
    admin: User = Depends(require_admin)
):
    """
    启用/禁用用户（管理员）
    """
    user = await User.get_or_none(id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 不能禁用自己
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能禁用自己"
        )
    
    # 切换状态
    user.is_active = not user.is_active
    await user.save()
    
    return user
