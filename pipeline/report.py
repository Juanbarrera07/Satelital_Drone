"""
TMI Report Generation Module.

Compiles execution metrics (Accuracy, Precision, Uncertainty) and generates 
Executive PDF reports ensuring compliance with IPCC Tier 2/3 standards.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import logging

logger = logging.getLogger(__name__)

class ReportEngine:
    def __init__(self, config: dict):
        """
        Initialize the report generator.
        
        Args:
            config (dict): Pipeline configuration governing reporting standards.
        """
        report_config = config.get('reporting', {})
        self.target_standard = report_config.get('target_standard', 'IPCC_Tier_3')
        self.min_acc = report_config.get('min_accuracy', 0.85)
        self.min_prec = report_config.get('min_precision', 0.80)
        self.max_unc = report_config.get('max_uncertainty', 0.15)

    def evaluate_compliance(self, accuracy: float, precision: float, uncertainty: float) -> bool:
        """
        Validates whether APU metrics meet the minimum requirements for the targeted IPCC Tier.
        """
        if accuracy >= self.min_acc and precision >= self.min_prec and uncertainty <= self.max_unc:
            return True
        return False

    def generate_executive_pdf(self, output_path: str, metrics: dict):
        """
        Generate a PDF report using ReportLab.
        
        Args:
            output_path (str): Destination path for the PDF.
            metrics (dict): Dictionary containing APU metrics.
                e.g., {'accuracy': 0.88, 'precision': 0.82, 'uncertainty': 0.05}
        """
        logger.info(f"Generating Executive Audit Report -> {output_path}")
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        elements.append(Paragraph("TerraForge Mining Intelligence - Executive Audit Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        # Summary
        compliance = self.evaluate_compliance(
            metrics.get('accuracy', 0), 
            metrics.get('precision', 0), 
            metrics.get('uncertainty', 1.0)
        )
        
        status_text = f"<font color='green'>COMPLIANT with {self.target_standard}</font>" if compliance else f"<font color='red'>NON-COMPLIANT with {self.target_standard}</font>"
        
        elements.append(Paragraph(f"<b>Overall Compliance Status:</b> {status_text}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Metrics Table
        data = [
            ['Metric', 'Measured Value', 'Threshold Required', 'Status'],
            ['Overall Accuracy', f"{metrics.get('accuracy', 0):.2%}", f">= {self.min_acc:.2%}", 'PASS' if metrics.get('accuracy', 0) >= self.min_acc else 'FAIL'],
            ['Precision (Critical)', f"{metrics.get('precision', 0):.2%}", f">= {self.min_prec:.2%}", 'PASS' if metrics.get('precision', 0) >= self.min_prec else 'FAIL'],
            ['Uncertainty Margin', f"{metrics.get('uncertainty', 0):.2%}", f"<= {self.max_unc:.2%}", 'PASS' if metrics.get('uncertainty', 0) <= self.max_unc else 'FAIL']
        ]
        
        t = Table(data, colWidths=[120, 100, 120, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        
        elements.append(t)
        
        # Save Doc
        doc.build(elements)
        logger.info("PDF generated successfully.")
