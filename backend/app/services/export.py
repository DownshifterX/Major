import io, json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from .. import models
from . import analysis_service as svc
from sqlalchemy.orm import Session

def generate_brief_pdf(db: Session, case: models.Case) -> bytes:
    data = svc.run_brief(db, case)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Pre-Hearing Brief: {case.title}", styles["Heading1"]), Spacer(1, 12)]

    sections = [
        ("Case Overview", data.get("case_overview")), ("Key Facts", data.get("key_facts")),
        ("Legal Issues", data.get("legal_issues")), ("Relevant Laws", data.get("relevant_laws")),
        ("Relevant Precedents", data.get("relevant_precedents")), ("Prosecution Arguments", data.get("prosecution_arguments")),
        ("Defence Arguments", data.get("defence_arguments")), ("Evidence Summary", data.get("evidence_summary")),
        ("Evidence Gaps", data.get("evidence_gaps")), ("Contradictions", data.get("contradictions")),
        ("Questions to Examine", data.get("questions_to_examine")),
    ]
    for title, content in sections:
        if content is None: continue
        story.append(Paragraph(title, styles["Heading2"]))
        val = json.dumps(content, indent=2) if isinstance(content, dict) else "\n".join([str(c) for c in content]) if isinstance(content, list) else str(content)
        sanitized = val.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        story.append(Paragraph(sanitized, styles["Normal"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buf.getvalue()
