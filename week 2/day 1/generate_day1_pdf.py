"""1-page Week 2 Day 1 write-up: ReAct, tool schemas, failure modes."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
NAVY = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#CBD5E1")
HEAD = colors.HexColor("#1E3A5F")
CARD = colors.HexColor("#F8FAFC")


class Footer(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self.saveState()
            self.setFont("Helvetica", 7.5)
            self.setFillColor(MUTED)
            self.drawString(32, 762, "WEB3GEEKS INTERNSHIP · WEEK 2 DAY 1")
            self.drawRightString(580, 762, "RAW PYTHON AGENT (NO FRAMEWORK)")
            self.setStrokeColor(LINE)
            self.line(32, 756, 580, 756)
            self.line(32, 28, 580, 28)
            self.drawString(32, 18, "ReAct Loop · Tool Calling Protocols · max_iterations Guardrail")
            self.drawRightString(580, 18, f"Page {self._pageNumber} of {n}")
            self.restoreState()
            super().showPage()
        super().save()


def P(name, size, leading, color, bold=False, after=3, before=0):
    styles = getSampleStyleSheet()
    return ParagraphStyle(
        name,
        parent=styles["Normal"],
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=leading,
        textColor=color,
        spaceAfter=after,
        spaceBefore=before,
    )


def build_pdf():
    path = str(HERE / "day1_writeup.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=32, rightMargin=32, topMargin=40, bottomMargin=34)
    title = P("t", 13, 16, NAVY, True, 2)
    h = P("h", 10, 12, HEAD, True, 3, 6)
    body = P("b", 8.3, 10.6, MUTED, False, 3)
    small = P("s", 7.4, 9.4, MUTED, False, 2)

    story = []
    story.append(Paragraph("Week 2 Day 1 — Agent Foundations (Raw Python Tool Calling & ReAct)", title))
    story.append(Paragraph(
        "<b>Mental Model:</b> An agent is an LLM in an execution loop: it reasons, selects tools, observes results, and decides what to do next. "
        "A chatbot answers in a single shot without tools; a workflow executes a hardcoded sequence of steps. Agentic systems require "
        "autonomy, tool use, multi-step planning, and self-correction. <i>Overkill:</i> An agent is unnecessary for deterministic lookups, "
        "static FAQs, or known DAGs where a basic script or prompt is faster, cheaper, and predictable.",
        body,
    ))
    story.append(HRFlowable(width="100%", thickness=0.6, color=LINE, spaceAfter=5))

    story.append(Paragraph("ReAct loop", h))
    story.append(Paragraph(
        "<b>Reason</b> (optional text) → <b>Act</b> (<font face='Courier'>tool_use</font> JSON) → "
        "<b>Observe</b> (<font face='Courier'>tool_result</font>) → repeat until the model emits a final text turn. "
        "The Python <font face='Courier'>while</font> loop in <font face='Courier'>agent.py</font> is that cycle, "
        "with <b>max_iterations=8</b> as a hard stop.",
        body,
    ))
    img = HERE / "react_loop.png"
    if img.is_file():
        story.append(Image(str(img), width=7.2 * inch, height=2.45 * inch))

    story.append(Paragraph("Tool schemas", h))
    story.append(Paragraph(
        "Descriptions are the model's API docs. Vague names cause hallucinated arguments; precise descriptions "
        "tell the model <i>when</i> to call, <i>what</i> type to send, and <i>what</i> the tool cannot do.",
        body,
    ))
    data = [
        ["Tool", "Inputs", "Role"],
        ["calculator", "expression: string", "Safe AST arithmetic (+ − × ÷ **). Not Python eval."],
        ["get_weather", "city: string", "Stub Celsius lookup (Karachi, London, Tokyo, NY, Oslo)."],
        ["read_sandbox_file", "filename: string", "Read sandbox/ only; blocks path traversal."],
        ["broken_lookup", "query: string", "Always raises — used to study error handling."],
    ]
    t = Table(data, colWidths=[108, 118, 324])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), CARD),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Paragraph(
        "Worked example: “weather in Karachi and London — which is warmer?” → two <font face='Courier'>get_weather</font> "
        "calls → optional <font face='Courier'>calculator</font> on 33−12 → final answer (Karachi warmer by 21 C).",
        body,
    ))

    story.append(Paragraph("Failure modes observed & mitigations", h))
    fails = [
        ["Failure", "Mitigation"],
        ["Infinite Reason–Act loop", "max_iterations hard stop; log every step."],
        ["Hallucinated tool name", "Dispatch table; return ERROR for unknown names; is_error=True."],
        ["Wrong / unsafe tool arguments", "JSON schema + required fields; calculator uses AST, not eval."],
        ["Silent tool exceptions", "Catch, return ERROR:…, log [observe]; model must not invent numbers."],
        ["Ambiguous user request", "Ask a clarifying question instead of guessing a city/file."],
        ["Missing tool (e.g. stock price)", "Refuse to fabricate; tell the user the tool does not exist."],
    ]
    t2 = Table(fails, colWidths=[200, 350])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), CARD),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t2)

    story.append(Paragraph("Why LangChain / LangGraph / CrewAI exist", h))
    story.append(Paragraph(
        "The hand-rolled loop already works, but production agents need retries, tracing, structured output parsing, "
        "graph control flow, human-in-the-loop gates, memory stores, and multi-agent routing. Frameworks are conveniences "
        "around this same ReAct core — not a different kind of intelligence. After Day 1, those libraries should read as "
        "wrappers over messages, tools, and a stop condition.",
        body,
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Reproduce: pip install -r requirements.txt · set GEMINI_API_KEY in .env · python agent.py", small))

    doc.build(story, canvasmaker=Footer)
    print("wrote", path)


if __name__ == "__main__":
    build_pdf()
