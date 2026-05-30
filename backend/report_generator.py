from datetime import datetime
from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Daily Report", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_report_pdf(
    report_data: dict,
    session_id: str,
    explanation: str | None = None,
    analysis: dict | None = None,
) -> bytes:
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 60, 110)
    pdf.cell(0, 14, "Daily Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    # Meta info
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 7, f"Date: {datetime.now().strftime('%B %d, %Y')}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 7, f"Report ID: {session_id[:8]}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    if explanation:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, _pdf_safe_text(explanation), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    _render_section(pdf, "1. Activity", report_data.get("activity", {}))
    pdf.ln(6)
    _render_section(pdf, "2. Safety", report_data.get("safety", {}))

    if analysis and (analysis.get("analysis") or analysis.get("recommendations")):
        pdf.ln(6)
        _render_analysis(pdf, analysis)

    return pdf.output()


def _render_section_header(pdf: ReportPDF, title: str):
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 60, 110)
    pdf.cell(0, 10, _pdf_safe_text(title), new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y()
    pdf.set_draw_color(30, 60, 110)
    pdf.line(10, y, 200, y)
    pdf.ln(6)


def _render_section(pdf: ReportPDF, title: str, fields: dict):
    _render_section_header(pdf, title)

    for field_key, field in fields.items():
        label = field.get("label", field_key)
        raw_value = field.get("value")
        value = "Not reported" if raw_value is None else _format_field_value(raw_value)

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 8, _pdf_safe_text(label), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 6, _pdf_safe_text(value), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)


def _render_analysis(pdf: ReportPDF, analysis: dict):
    _render_section_header(pdf, "3. Analysis & Recommendations")

    overall = analysis.get("analysis")
    if overall:
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 6, _pdf_safe_text(overall), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    recommendations = analysis.get("recommendations", [])
    for rec in recommendations:
        category = rec.get("category") or "General"
        text = rec.get("text", "")
        source = rec.get("source", "")

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 60, 110)
        pdf.cell(0, 7, _pdf_safe_text(f"- {category}"), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(70, 70, 70)
        pdf.multi_cell(0, 6, _pdf_safe_text(text), new_x="LMARGIN", new_y="NEXT")

        if source:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            pdf.multi_cell(0, 5, _pdf_safe_text(f"Reference: {source}"), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)


def _format_field_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(_format_field_value(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(
            f"{_humanize_key(key)}: {_format_field_value(val)}"
            for key, val in value.items()
        )
    return str(value)


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").title()


def _pdf_safe_text(text: str) -> str:
    # fpdf core fonts expect latin-1-compatible strings.
    return str(text).encode("latin-1", "replace").decode("latin-1")
