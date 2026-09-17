import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from .. import models
from . import analysis_service as svc
from sqlalchemy.orm import Session

def generate_brief_pdf(db: Session, case: models.Case) -> bytes:
    brief_data = svc.run_brief(db, case)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    Story = []
    
    title_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    normal_style = styles["Normal"]
    
    def add_section(title, content):
        Story.append(Paragraph(title, h2_style))
        if isinstance(content, dict):
            content = json.dumps(content, indent=2)
        elif isinstance(content, list):
            content = "\n".join([str(c) for c in content])
        
        # very basic sanitization for reportlab xml parser
        content_str = str(content).replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        Story.append(Paragraph(content_str, normal_style))
        Story.append(Spacer(1, 12))
    
    Story.append(Paragraph(f"Pre-Hearing Brief: {case.title}", title_style))
    Story.append(Spacer(1, 12))
    
    add_section("Case Overview", brief_data.get("case_overview", "N/A"))
    add_section("Key Facts", brief_data.get("key_facts", "N/A"))
    add_section("Legal Issues", brief_data.get("legal_issues", "N/A"))
    add_section("Relevant Laws", brief_data.get("relevant_laws", "N/A"))
    add_section("Relevant Precedents", brief_data.get("relevant_precedents", "N/A"))
    add_section("Prosecution Arguments", brief_data.get("prosecution_arguments", "N/A"))
    add_section("Defence Arguments", brief_data.get("defence_arguments", "N/A"))
    add_section("Evidence Summary", brief_data.get("evidence_summary", "N/A"))
    add_section("Evidence Gaps", brief_data.get("evidence_gaps", "N/A"))
    add_section("Contradictions", brief_data.get("contradictions", "N/A"))
    add_section("Questions to Examine", brief_data.get("questions_to_examine", "N/A"))
    
    doc.build(Story)
    return buffer.getvalue()
