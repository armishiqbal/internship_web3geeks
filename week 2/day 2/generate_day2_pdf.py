"""1-page Week 2 Day 2 write-up: Raw Python vs LangChain Agent.

Accurately covers all 5 tasks:
- Task 1: Concept mapping & LCEL under-the-hood explanation
- Task 2: Tool schemas, docstring role & external JSON catalog
- Task 3: Annotated Reasoning Trace (Reason -> Act -> Observe -> Final)
- Task 4: 3-turn conversation memory with dependent context
- Task 5: Structured output (Pydantic) & graceful error handling (ToolException)
- Synthesis: Productivity benefits vs abstraction leakiness / 'magic'
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
NAVY = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#334155")
LINE = colors.HexColor("#CBD5E1")
HEAD = colors.HexColor("#1E3A5F")
ACCENT = colors.HexColor("#2563EB")
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
            self.setFont("Helvetica-Bold", 7.2)
            self.setFillColor(MUTED)
            self.drawString(32, 764, "WEB3GEEKS INTERNSHIP · WEEK 2 DAY 2")
            self.drawRightString(580, 764, "LANGCHAIN AGENT FRAMEWORK")
            self.setStrokeColor(LINE)
            self.setLineWidth(0.5)
            self.line(32, 758, 580, 758)
            self.line(32, 26, 580, 26)
            self.setFont("Helvetica", 7.2)
            self.drawString(32, 16, "LCEL · create_tool_calling_agent · AgentExecutor · Memory · Pydantic Output")
            self.drawRightString(580, 16, f"Page {self._pageNumber} of {n}")
            self.restoreState()
            super().showPage()
        super().save()


def P(name, size, leading, color, bold=False, after=2, before=0):
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
    path = str(HERE / "day2_writeup.pdf")
    doc = SimpleDocTemplate(
        path,
        pagesize=letter,
        leftMargin=30,
        rightMargin=30,
        topMargin=38,
        bottomMargin=30,
    )

    t_style = P("t", 12, 14.5, NAVY, True, 2)
    h_style = P("h", 9.2, 11.2, HEAD, True, 2, 4)
    b_style = P("b", 7.6, 9.6, MUTED, False, 2.5)

    story = []
    story.append(Paragraph("Week 2 Day 2 — LangChain: Tools, Chains, Memory & Framework Agents", t_style))
    story.append(Paragraph(
        "<b>Overview:</b> Day 1 implemented a raw ReAct loop with explicit HTTP message payloads. Day 2 refactors this "
        "into LangChain's component ecosystem, adding LCEL pipelines, multi-turn session memory, external catalog querying, "
        "Pydantic structured output, and resilient tool error handling.",
        b_style,
    ))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE, spaceAfter=4, spaceBefore=1))

    # 1. Concept Mapping & LCEL Pipeline
    story.append(Paragraph("1. Concept Mapping & LCEL Mechanics", h_style))
    rows = [
        ["Concept", "Day 1 (Raw Python)", "Day 2 (LangChain Abstraction)"],
        ["LLM Wrapper", "genai.Client + generate_content", "ChatGoogleGenerativeAI (implements Runnable interface)"],
        ["Tool Registry", "TOOL_SCHEMAS dict + Python fns", "@tool / StructuredTool (docstring -> function calling schema)"],
        ["Agent Loop", "while-loop in agent.py (manual dispatch)", "create_tool_calling_agent + AgentExecutor (automated ReAct)"],
        ["Memory", "Explicit messages list you append", "ConversationBufferMemory / RunnableWithMessageHistory"],
    ]
    t = Table(rows, colWidths=[90, 195, 267])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.0),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD]),
    ]))
    story.append(t)
    story.append(Paragraph(
        "<b>Under the hood of LCEL (|):</b> LangChain overloads Python's bitwise OR (<font face='Courier'>__or__</font>) "
        "operator across <font face='Courier'>Runnable</font> classes to build a <font face='Courier'>RunnableSequence</font>. "
        "When executing <font face='Courier'>prompt | llm | StrOutputParser()</font>, the output of each link is piped as input "
        "to the next, automatically orchestrating streaming, async execution, batching, and OpenTelemetry callback tracing.",
        b_style,
    ))

    # 2. Tool Calling & External Data
    story.append(Paragraph("2. Tool Docstrings & External Data Source", h_style))
    story.append(Paragraph(
        "Tool docstrings function directly as prompt instructions: LangChain inspects function type hints and extracts docstrings "
        "into the model's tool schema. Reused Day 1's safe AST <font face='Courier'>calculator</font> and <font face='Courier'>get_weather</font>, "
        "and added <font face='Courier'>lookup_product_price</font> to read a local JSON data source (<font face='Courier'>data/products.json</font>).",
        b_style,
    ))

    # 3. Annotated Reasoning Trace
    story.append(Paragraph("3. Annotated Agent Reasoning Trace (ReAct Verification)", h_style))
    trace = [
        ["Phase", "Agent Execution Step & Event Trace", "Subsystem / State"],
        ["REASON", "Model inspects user query + system prompt + tool descriptions; decides tool calls.", "AgentExecutor Plan"],
        ["ACT", "Invoking: get_weather({'city': 'Karachi'}) -> returns 33 C, hot, humid.", "Tool Invocation"],
        ["ACT", "Invoking: get_weather({'city': 'London'}) -> returns 12 C, overcast, light rain.", "Tool Invocation"],
        ["OBSERVE", "Observations injected into agent_scratchpad as tool_result / FunctionResponse.", "Scratchpad Update"],
        ["ACT", "Invoking: calculator({'expression': '33 - 12'}) -> returns 21.", "Tool Invocation"],
        ["FINAL", "Model synthesizes: 'Karachi is warmer than London by 21 degrees Celsius.'", "Terminal Plain Text"],
    ]
    tt = Table(trace, colWidths=[65, 360, 127])
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.9),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD]),
    ]))
    story.append(tt)

    # 4. Multi-Turn Memory & Structured Output
    story.append(Paragraph("4. Conversation Memory & Pydantic Structured Output", h_style))
    story.append(Paragraph(
        "<b>3-Turn Memory Scenario:</b> Tested multi-turn dependency: (1) <i>Find price of Pro ($49)</i> -> "
        "(2) <i>Compare to Enterprise ($149)</i> -> (3) <i>Which to recommend to a budget client?</i> "
        "<font face='Courier'>ConversationBufferMemory</font> preserves past turns in <font face='Courier'>chat_history</font>, "
        "allowing the model to resolve 'Which one' without re-querying. "
        "<b>Structured Output:</b> Forced the final recommendation into a typed <font face='Courier'>ProductRecommendation</font> "
        "Pydantic schema (<font face='Courier'>recommended_product: str, price_usd: float, rationale: str, budget_friendly: bool</font>) "
        "via <font face='Courier'>llm.with_structured_output()</font> for deterministic downstream consumption.",
        b_style,
    ))

    # 5. Failure Recovery, What LangChain Made Easier & Magic / Leakiness
    story.append(Paragraph("5. Error Handling & Architectural Trade-Offs (Raw vs LangChain)", h_style))
    story.append(Paragraph(
        "<b>Graceful Error Recovery:</b> In <font face='Courier'>broken_lookup</font>, throwing <font face='Courier'>ToolException</font> "
        "configured with <font face='Courier'>handle_tool_error=True</font> intercepts runtime crashes and converts the exception "
        "into a synthetic observation, allowing the model to acknowledge upstream timeouts rather than terminating the process.<br/>"
        "<b>What LangChain Made Easier:</b> Drastically reduced boilerplate: declarative LCEL piping, turnkey memory managers, "
        "automated JSON Schema generation from docstrings/type hints, and built-in structured schema parsing.<br/>"
        "<b>Abstraction Leakiness & 'Magic':</b> (1) <i>Package churn:</i> Legacy AgentExecutor was relegated to "
        "<font face='Courier'>langchain_classic</font>. (2) <i>Opaque state:</i> Prompt scratchpad formatting and Gemini thought "
        "signatures are hidden; subtle token/state mismatches fail as remote 400 Bad Requests rather than readable Python exceptions. "
        "(3) <i>Payload obscurity:</i> <font face='Courier'>verbose=True</font> surfaces tool I/O but masks raw HTTP requests.",
        b_style,
    ))

    doc.build(story, canvasmaker=Footer)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build_pdf()
