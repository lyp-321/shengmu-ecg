"""
交互式模型训练工具 - 带终端可视化界面
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from rich import box
from rich.text import Text
import subprocess


console = Console()


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        🫀 ECG 多模态模型训练系统 v2.0 🫀                  ║
    ║                                                           ║
    ║     基于多模态深度学习与联邦学习的智能心电图分析          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def show_main_menu():
    """显示主菜单"""
    console.print("\n")
    
    table = Table(title="🎯 训练选项", box=box.ROUNDED, show_header=True, 
                  header_style="bold magenta")
    table.add_column("选项", style="cyan", width=8)
    table.add_column("名称", style="green", width=30)
    table.add_column("说明", style="yellow")
    
    table.add_row("1", "🤖 传统机器学习", "XGBoost, LightGBM, CatBoost, RF, SVM")
    table.add_row("2", "🧠 深度学习模型", "ResNet, Transformer, BiLSTM, TCN等")
    table.add_row("3", "🚀 全部训练", "一键训练所有模型（推荐）")
    table.add_row("4", "📊 查看已训练模型", "查看已保存的模型列表")
    table.add_row("5", "🧪 测试模型", "运行模型测试")
    table.add_row("0", "❌ 退出", "退出训练系统")
    
    console.print(table)
    console.print("\n")


def check_environment():
    """检查环境配置"""
    console.print("\n[bold cyan]🔍 检查环境配置...[/bold cyan]\n")
    
    checks = []
    
    # 检查Python版本
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(("Python版本", python_version, "✓" if sys.version_info >= (3, 7) else "✗"))
    
    # 检查PyTorch
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = "✓ CUDA可用" if torch.cuda.is_available() else "✓ CPU模式"
        checks.append(("PyTorch", torch_version, cuda_available))
    except ImportError as e:
        checks.append(("PyTorch", f"导入失败: {e}", "✗"))
    except Exception as e:
        checks.append(("PyTorch", f"错误: {e}", "✗"))
    
    # 检查其他依赖
    packages = [
        ("NumPy", "numpy"),
        ("Scikit-learn", "sklearn"),
        ("XGBoost", "xgboost"),
        ("LightGBM", "lightgbm"),
        ("WFDB", "wfdb"),
        ("Rich", "rich")
    ]
    
    for name, module in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', '已安装')
            checks.append((name, version, "✓"))
        except ImportError:
            checks.append((name, "未安装", "✗"))
    
    # 显示检查结果
    table = Table(title="环境检查结果", box=box.SIMPLE)
    table.add_column("组件", style="cyan")
    table.add_column("版本", style="yellow")
    table.add_column("状态", style="green")
    
    for name, version, status in checks:
        style = "green" if "✓" in status else "red"
        table.add_row(name, version, f"[{style}]{status}[/{style}]")
    
    console.print(table)
    console.print("\n")
    
    # 检查是否有缺失的包（排除PyTorch的CPU警告）
    missing = []
    for name, version, status in checks:
        if "✗" in status and "未安装" in version:
            missing.append(name)
    
    if missing:
        console.print(f"[bold red]⚠️  缺少以下依赖: {', '.join(missing)}[/bold red]")
        console.print("[yellow]请运行: pip install torch xgboost lightgbm wfdb rich[/yellow]\n")
        if not Confirm.ask("是否继续？（某些功能可能不可用）"):
            return False
    
    return True


def check_data():
    """检查数据集"""
    console.print("[bold cyan]📁 检查数据集...[/bold cyan]\n")
    
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        console.print(f"[yellow]数据目录不存在: {data_dir}[/yellow]")
        if Confirm.ask("是否创建数据目录？"):
            os.makedirs(data_dir, exist_ok=True)
            console.print("[green]✓ 数据目录已创建[/green]")
    
    # 检查MIT-BIH数据
    dat_files = [f for f in os.listdir(data_dir) if f.endswith('.dat')] if os.path.exists(data_dir) else []
    
    if len(dat_files) > 0:
        console.print(f"[green]✓ 找到 {len(dat_files)} 个MIT-BIH数据文件[/green]")
        return True
    else:
        console.print("[yellow]⚠️  未找到MIT-BIH数据集[/yellow]")
        console.print("[dim]训练将使用模拟数据（准确率可能较低）[/dim]")
        
        if Confirm.ask("是否下载MIT-BIH数据集？（需要网络连接）"):
            try:
                console.print("[cyan]正在下载MIT-BIH数据集...[/cyan]")
                import wfdb
                wfdb.dl_database('mitdb', data_dir)
                console.print("[green]✓ 数据集下载完成[/green]")
                return True
            except Exception as e:
                console.print(f"[red]✗ 下载失败: {e}[/red]")
                console.print("[yellow]将使用模拟数据继续训练[/yellow]")
                return False
        
        return False


def train_traditional_ml():
    """训练传统机器学习模型"""
    console.print("\n")
    panel = Panel.fit(
        "[bold cyan]🤖 训练传统机器学习模型[/bold cyan]\n\n"
        "包含模型:\n"
        "  • Random Forest\n"
        "  • XGBoost\n"
        "  • LightGBM\n"
        "  • CatBoost\n"
        "  • SVM",
        border_style="cyan"
    )
    console.print(panel)
    
    if not Confirm.ask("\n开始训练？"):
        return
    
    console.print("\n[bold green]🚀 开始训练...[/bold green]\n")
    
    # 运行训练脚本
    try:
        result = subprocess.run(
            ["python", "scripts/train_traditional_ml.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            console.print("\n[bold green]✓ 训练完成！[/bold green]")
        else:
            console.print("\n[bold red]✗ 训练失败[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 训练出错: {e}[/bold red]")
    
    console.input("\n按Enter键继续...")


def train_deep_learning():
    """训练深度学习模型"""
    console.print("\n")
    panel = Panel.fit(
        "[bold cyan]🧠 训练深度学习模型[/bold cyan]\n\n"
        "包含模型:\n"
        "  • ResNet-1D (18层残差网络)\n"
        "  • SE-ResNet-1D (注意力机制)\n"
        "  • Transformer (多头自注意力)\n"
        "  • BiLSTM (双向LSTM)\n"
        "  • TCN (时间卷积网络)\n"
        "  • Inception-1D (多尺度卷积)",
        border_style="cyan"
    )
    console.print(panel)
    
    # 选择训练参数
    console.print("\n[bold]训练参数配置:[/bold]")
    
    epochs = Prompt.ask("训练轮数", default="30")
    batch_size = Prompt.ask("批次大小", default="32")
    
    if not Confirm.ask("\n开始训练？"):
        return
    
    console.print("\n[bold green]🚀 开始训练...[/bold green]\n")
    
    # 运行训练脚本
    try:
        result = subprocess.run(
            ["python", "scripts/train_multimodal_models.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            console.print("\n[bold green]✓ 训练完成！[/bold green]")
            console.print("[cyan]训练曲线已保存到: experiments/results/training_curves.png[/cyan]")
        else:
            console.print("\n[bold red]✗ 训练失败[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 训练出错: {e}[/bold red]")
    
    console.input("\n按Enter键继续...")


def train_all():
    """训练所有模型"""
    console.print("\n")
    panel = Panel.fit(
        "[bold cyan]🚀 一键训练所有模型[/bold cyan]\n\n"
        "将依次训练:\n"
        "  1. 传统机器学习模型 (5个)\n"
        "  2. 深度学习模型 (6个)\n\n"
        "[yellow]预计时间: 30-60分钟[/yellow]",
        border_style="cyan"
    )
    console.print(panel)
    
    if not Confirm.ask("\n确认开始训练所有模型？"):
        return
    
    # 步骤1: 传统ML
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]步骤 1/2: 训练传统机器学习模型[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    try:
        subprocess.run(["python", "scripts/train_traditional_ml.py"], check=True)
        console.print("\n[bold green]✓ 传统ML模型训练完成[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 传统ML训练失败: {e}[/bold red]")
        if not Confirm.ask("继续训练深度学习模型？"):
            return
    
    # 步骤2: 深度学习
    console.print("\n[bold cyan]═══════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]步骤 2/2: 训练深度学习模型[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════[/bold cyan]\n")
    
    try:
        subprocess.run(["python", "scripts/train_multimodal_models.py"], check=True)
        console.print("\n[bold green]✓ 深度学习模型训练完成[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]✗ 深度学习训练失败: {e}[/bold red]")
    
    # 完成
    console.print("\n[bold green]🎉 所有模型训练完成！[/bold green]")
    console.print("\n[cyan]模型保存位置:[/cyan]")
    console.print("  • app/algorithms/models/")
    console.print("\n[cyan]结果保存位置:[/cyan]")
    console.print("  • experiments/results/")
    
    console.input("\n按Enter键继续...")


def show_trained_models():
    """显示已训练的模型"""
    console.print("\n")
    
    model_dir = "app/algorithms/models"
    
    if not os.path.exists(model_dir):
        console.print("[yellow]模型目录不存在[/yellow]")
        console.input("\n按Enter键继续...")
        return
    
    files = os.listdir(model_dir)
    
    if not files:
        console.print("[yellow]未找到已训练的模型[/yellow]")
        console.input("\n按Enter键继续...")
        return
    
    # 分类显示
    ml_models = [f for f in files if f.endswith('.pkl')]
    dl_models = [f for f in files if f.endswith('.pth')]
    
    table = Table(title="📦 已训练的模型", box=box.ROUNDED)
    table.add_column("类型", style="cyan")
    table.add_column("文件名", style="green")
    table.add_column("大小", style="yellow")
    
    for f in ml_models:
        size = os.path.getsize(os.path.join(model_dir, f))
        size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
        table.add_row("传统ML", f, size_str)
    
    for f in dl_models:
        size = os.path.getsize(os.path.join(model_dir, f))
        size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
        table.add_row("深度学习", f, size_str)
    
    console.print(table)
    console.print(f"\n[cyan]总计: {len(files)} 个模型文件[/cyan]")
    
    console.input("\n按Enter键继续...")


def run_tests():
    """运行模型测试"""
    console.print("\n")
    panel = Panel.fit(
        "[bold cyan]🧪 运行模型测试[/bold cyan]\n\n"
        "将测试:\n"
        "  • 多模态融合引擎\n"
        "  • 深度学习模型\n"
        "  • 图神经网络\n"
        "  • 联邦学习框架\n"
        "  • 推理引擎集成",
        border_style="cyan"
    )
    console.print(panel)
    
    if not Confirm.ask("\n开始测试？"):
        return
    
    console.print("\n[bold green]🚀 开始测试...[/bold green]\n")
    
    try:
        subprocess.run(["python", "tests/test_algorithms.py"], check=True)
    except Exception as e:
        console.print(f"\n[bold red]✗ 测试失败: {e}[/bold red]")
    
    console.input("\n按Enter键继续...")


def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        console.print("\n[bold red]环境检查失败，请先安装缺失的依赖[/bold red]")
        return
    
    # 检查数据
    check_data()
    
    # 主循环
    while True:
        console.clear()
        print_banner()
        show_main_menu()
        
        choice = Prompt.ask("请选择", choices=["0", "1", "2", "3", "4", "5"], default="3")
        
        if choice == "0":
            console.print("\n[bold cyan]👋 再见！[/bold cyan]\n")
            break
        elif choice == "1":
            train_traditional_ml()
        elif choice == "2":
            train_deep_learning()
        elif choice == "3":
            train_all()
        elif choice == "4":
            show_trained_models()
        elif choice == "5":
            run_tests()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]训练已中断[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]发生错误: {e}[/bold red]")
        import traceback
        traceback.print_exc()
