"""
ECG 相关 API 路由
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from typing import List

from app.services.ecg_service import ECGService
from app.schemas.ecg import ECGTaskResponse, ECGResultResponse
from app.models.user import User
from app.core.deps import get_current_active_user, require_admin

router = APIRouter()
ecg_service = ECGService()


@router.post("/upload", response_model=ECGTaskResponse)
async def upload_ecg_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    algo_mode: str = "fusion"  # 新增：算法模式参数
):
    """
    上传 ECG 文件并创建分析任务（需要登录）
    
    Args:
        file: ECG文件
        algo_mode: 算法模式 (fusion/dual/ml_ensemble/dl_advanced/graph)
    """
    try:
        task = await ecg_service.create_task(file, current_user.id, background_tasks, algo_mode)
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=ECGResultResponse)
async def get_task_result(
    task_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    获取任务分析结果（需要登录）
    """
    result = await ecg_service.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 检查权限：只能查看自己的任务，管理员可以查看所有任务
    if result.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问此任务")
    
    return result


@router.get("/tasks", response_model=List[ECGTaskResponse])
async def list_tasks(
    skip: int = 0,
    limit: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """
    获取任务列表（需要登录）
    普通用户只能看到自己的任务，管理员可以看到所有任务
    """
    tasks = await ecg_service.list_tasks(skip, limit, current_user)
    return tasks
@router.get("/tasks/{task_id}/report")
async def download_report(
    task_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    下载 PDF 诊断报告（需要登录）
    """
    from fastapi.responses import FileResponse
    import os
    
    result = await ecg_service.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 检查权限
    if result.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问此报告")
    
    if not result.report_path or not os.path.exists(result.report_path):
        raise HTTPException(status_code=404, detail="报告尚未生成或文件丢失")
    
    return FileResponse(
        path=result.report_path,
        filename=f"ECG_Report_{task_id}.pdf",
        media_type="application/pdf"
    )

@router.get("/tasks/{task_id}/signal")
async def get_task_signal(
    task_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    获取任务原始信号数据（用于历史回溯波形回放）
    """
    # 权限检查逻辑与获取结果一致
    result = await ecg_service.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if result.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权访问此数据")
    
    signal_data = await ecg_service.get_signal(task_id)
    if not signal_data:
        raise HTTPException(status_code=404, detail="原始信号文件丢失")
    
    return signal_data
