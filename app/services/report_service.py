import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

class ECGReportService:
    """ECG PDF 报告生成服务"""
    
    def __init__(self):
        self.reports_dir = "data/reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        self._register_fonts()

    def _register_fonts(self):
        """注册中文字体"""
        # 尝试注册系统常见的中文路径，如果失败则使用默认
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Ubuntu 常用
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
        ]
        
        self.font_name = "Helvetica" # 默认
        for path in font_paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont("ChineseFont", path))
                    self.font_name = "ChineseFont"
                    break
                except:
                    continue

    def generate_report(self, task_data: dict) -> str:
        """
        生成 PDF 报告 - 深度优化展示逻辑
        """
        result = task_data.get('result', {})
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ECG_Report_{task_data['id']}_{timestamp}.pdf"
        file_path = os.path.join(self.reports_dir, filename)
        
        doc = SimpleDocTemplate(file_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        elements = []
        
        # 字体处理
        cn_font = self.font_name
        en_font = "Helvetica"
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('T', fontName=cn_font, fontSize=24, alignment=1, spaceAfter=20, textColor=colors.HexColor("#4f46e5"))
        label_style = ParagraphStyle('L', fontName=cn_font, fontSize=10, textColor=colors.HexColor("#64748b"))
        val_style = ParagraphStyle('V', fontName=en_font, fontSize=10, textColor=colors.HexColor("#1e293b"))
        normal_style = ParagraphStyle('N', fontName=cn_font, fontSize=10, textColor=colors.HexColor("#1e293b"))

        # 1. 头部
        elements.append(Paragraph("智能健康诊断报告", title_style))
        elements.append(Paragraph(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", label_style))
        elements.append(Spacer(1, 15))

        # 2. 基本信息 (使用简单表格)
        base_info = [
            [Paragraph("任务编号", label_style), Paragraph(str(task_data.get('id', '--')), val_style), 
             Paragraph("原始文件", label_style), Paragraph(task_data.get('filename', '--'), val_style)],
            [Paragraph("分析模型", label_style), Paragraph("端云协同 AI 引擎", val_style), 
             Paragraph("分析状态", label_style), Paragraph("已完成", normal_style)]
        ]
        t1 = Table(base_info, colWidths=[70, 180, 70, 180])
        t1.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f8fafc")),
        ]))
        elements.append(t1)
        elements.append(Spacer(1, 20))

        # 3. 核心结论
        elements.append(Paragraph("核心诊断结论", ParagraphStyle('H', fontName=cn_font, fontSize=14, spaceAfter=10)))
        diag = result.get('diagnosis', '数据不足')
        risk = result.get('risk_level', '未知')
        risk_color = "#10b981" if "低" in risk else ("#f59e0b" if "中" in risk else "#f43f5e")
        
        elements.append(Paragraph(f"诊断结果：<font color='#1e293b'><b>{diag}</b></font>", ParagraphStyle('D', fontName=cn_font, fontSize=12)))
        elements.append(Paragraph(f"风险等级：<font color='{risk_color}'><b>{risk}</b></font>", ParagraphStyle('R', fontName=cn_font, fontSize=12)))
        elements.append(Spacer(1, 20))

        # 4. 详细指标 (关键修复：确保数值列宽度足够且字体兼容)
        elements.append(Paragraph("详细生理指标", ParagraphStyle('H', fontName=cn_font, fontSize=14, spaceAfter=10)))
        
        def get_v(k, suffix=""):
            v = result.get(k)
            if v is None or v == 0: return "--"
            return f"{float(v):.2f}{suffix}"

        header = [Paragraph(x, ParagraphStyle('TH', fontName=cn_font, fontSize=10, textColor=colors.white)) 
                  for x in ["指标名称", "分析数值", "单位", "参考范围", "状态"]]
        
        rows = [header]
        metrics = [
            ("平均心率", 'heart_rate', "BPM", "60-100", "正常" if 60<=result.get('heart_rate',0)<=100 else "异常"),
            ("心率变异性", 'hrv_sdnn', "ms", "> 50", "正常" if result.get('hrv_sdnn',0)>50 else "偏低"),
            ("房颤概率", 'afib_prob', "%", "< 50", "正常" if result.get('afib_prob',0)<0.5 else "高危")
        ]
        
        for name, key, unit, ref, status in metrics:
            val = get_v(key, "%" if key=='afib_prob' else "")
            if key == 'afib_prob': val = get_v(key, "") # 后面手动乘100
            if key == 'afib_prob': val = f"{float(result.get(key,0))*100:.1f}%" if result.get(key) else "--"
            
            rows.append([
                Paragraph(name, normal_style),
                Paragraph(val, val_style),
                Paragraph(unit, val_style),
                Paragraph(ref, val_style),
                Paragraph(status, normal_style)
            ])

        t2 = Table(rows, colWidths=[120, 100, 60, 100, 80])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#6366f1")),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,1), (3,-1), 'CENTER'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        elements.append(t2)
        
        # 5. 免责声明
        elements.append(Spacer(1, 40))
        elements.append(Paragraph("注意：本报告由 AI 系统自动生成，仅供参考。若有不适请及时就医。", 
                                 ParagraphStyle('F', fontName=cn_font, fontSize=9, textColor=colors.grey)))

        doc.build(elements)
        return file_path
