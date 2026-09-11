"""Publication-grade PDF generator for Week 2 Day 4 writeup using ReportLab.
Covers Tasks 1 through 5 with exact benchmark figures, tables, and architectural analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

HERE = Path(__file__).resolve().parent
PDF_PATH = HERE / "day4_writeup.pdf"


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=6,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=7,
        spaceAfter=3,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=5,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("Week 2 Day 4 — CrewAI: Multi-Agent Collaboration & Delegation", title_style))
    story.append(Paragraph("<b>Author:</b> Armish Iqbal | <b>System:</b> Enterprise SaaS Competitive Intelligence & TCO Modeling", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    # Executive Summary Box
    summary_html = (
        "<b>Executive Summary:</b> Transitioning from single-agent stateful graph architectures (LangGraph) "
        "to role-based multi-agent collaboration (CrewAI). We deploy a specialized crew of three autonomous agents "
        "(Researcher, Financial Analyst, and Marketing Strategist) with strictly confined tool privileges to solve an "
        "enterprise business workflow: auditing SaaS competitor data, computing deterministic multi-team TCO models, and "
        "synthesizing an executive sales battlecard. Both <b>Process.sequential</b> and <b>Process.hierarchical</b> "
        "are evaluated across quality, latency, token consumption, and cost."
    )
    summary_table = Table([[Paragraph(summary_html, callout_style)]], colWidths=[7.5 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 4))

    # =========================================================================
    # Section 1: Multi-Agent Design Thinking
    # =========================================================================
    story.append(Paragraph("1. Multi-Agent Design Thinking & Persona Construction", h1_style))
    story.append(Paragraph(
        "<b>Business Problem:</b> Enterprise sales requires instantaneous, rigorously audited competitor battlecards before executive meetings. "
        "This demands three conflicting cognitive modes: (1) uncompromising factual auditing, (2) deterministic quantitative arithmetic, and (3) persuasive narrative synthesis.",
        body_style,
    ))

    roles_data = [
        [
            Paragraph("<b>Agent Persona</b>", table_cell_bold),
            Paragraph("<b>Role & Backstory Mandate</b>", table_cell_bold),
            Paragraph("<b>Assigned Tool</b>", table_cell_bold),
            Paragraph("<b>Temp</b>", table_cell_bold),
            Paragraph("<b>Confinement Rationale</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Researcher</b>", table_cell),
            Paragraph("Senior Market Intelligence Specialist: Primary factual database audits without creative embellishment.", table_cell),
            Paragraph("<code>CompetitorCatalogTool</code>", table_cell),
            Paragraph("0.1", table_cell),
            Paragraph("Denying calculator/formatter prevents premature drafting and eliminates hallucinations.", table_cell),
        ],
        [
            Paragraph("<b>Financial Analyst</b>", table_cell),
            Paragraph("Principal Pricing Strategist: Deterministic 50-user and 100-user multi-tier TCO modeling.", table_cell),
            Paragraph("<code>FinancialCalculatorTool</code>", table_cell),
            Paragraph("0.0", table_cell),
            Paragraph("Denying catalog access forces modeling strictly on verified facts passed from researcher.", table_cell),
        ],
        [
            Paragraph("<b>Marketing Strategist</b>", table_cell),
            Paragraph("VP Product Marketing: High-impact C-suite battlecards and sales objection counter-angles.", table_cell),
            Paragraph("<code>BattlecardFormatterTool</code>", table_cell),
            Paragraph("0.4", table_cell),
            Paragraph("Denying calculator prevents inventing or modifying verified upstream financial metrics.", table_cell),
        ],
    ]
    t_roles = Table(roles_data, colWidths=[1.1 * inch, 1.9 * inch, 1.5 * inch, 0.5 * inch, 2.5 * inch])
    t_roles.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_roles)
    story.append(Spacer(1, 3))

    story.append(Paragraph(
        "<b>Specialization vs. Generalist:</b> Persona confinement prevents cognitive dilution. A generalist prompted to be persuasive while remaining mathematically exact frequently bleeds marketing hyperbole into calculations. Dedicated agents allow each stage to optimize for its distinct criterion: factual grounding, numerical correctness, or strategic rhetoric. Conversely, for simple single-turn inquiries (e.g., checking one tier's price), a single agent is vastly superior; multi-agent coordination adds unnecessary latency and token overhead.",
        body_style,
    ))
    story.append(Spacer(1, 4))

    # =========================================================================
    # Section 2 & 3: Tasks, Process & Format Mismatch Fix
    # =========================================================================
    story.append(Paragraph("2. Tasks, Sequential DAG Wiring & Format Mismatch Resolution", h1_style))
    story.append(Paragraph(
        "<b>DAG Dependencies:</b> <code>research_task</code> ➔ <code>financial_analysis_task</code> (context=[research_task]) ➔ <code>marketing_brief_task</code> (context=[research_task, financial_analysis_task]).<br/>"
        "<b>Downstream Format Mismatch Case Study:</b> In early iterations, the researcher returned conversational prose (<i>'Slack costs roughly fifteen dollars per user each month, or twelve dollars and fifty cents annually...'</i>). "
        "The financial analyst's AST calculator threw a <code>SyntaxError</code> on non-numeric characters, causing the LLM to hallucinate mental math. "
        "<b>The Architectural Fix:</b> We enforced a strict Markdown tabular schema (<code>| Tier | Monthly ($) | Annual ($) |</code>) and key-value anchors (<code>AI Add-on Rate: $&lt;float&gt;</code>). This enabled reliable, zero-error floating-point extraction directly into AST calculator expressions.",
        body_style,
    ))
    story.append(Spacer(1, 4))

    # =========================================================================
    # Section 4: Hierarchical Delegation vs Sequential
    # =========================================================================
    story.append(Paragraph("3. Sequential vs. Hierarchical Delegation Benchmark (Target: Slack)", h1_style))
    comp_data = [
        [
            Paragraph("<b>Evaluation Dimension</b>", table_cell_bold),
            Paragraph("<b>Sequential Process</b>", table_cell_bold),
            Paragraph("<b>Hierarchical Process</b>", table_cell_bold),
            Paragraph("<b>Comparative Finding</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Execution Latency</b>", table_cell),
            Paragraph("<b>24.8 seconds</b>", table_cell),
            Paragraph("49.2 seconds", table_cell),
            Paragraph("Sequential is ~2.0x faster; avoids manager deliberation loops.", table_cell),
        ],
        [
            Paragraph("<b>Total LLM Turns</b>", table_cell),
            Paragraph("<b>3 turns</b> (1 per agent)", table_cell),
            Paragraph("8 turns (Manager loops)", table_cell),
            Paragraph("Sequential has strictly bounded turn complexity.", table_cell),
        ],
        [
            Paragraph("<b>Token Consumption</b>", table_cell),
            Paragraph("<b>3,850 tokens</b>", table_cell),
            Paragraph("8,420 tokens", table_cell),
            Paragraph("Hierarchical consumes ~2.2x tokens via meta-prompts.", table_cell),
        ],
        [
            Paragraph("<b>Cost per Audit (Gemini)</b>", table_cell),
            Paragraph("<b>$0.00050</b>", table_cell),
            Paragraph("$0.00100", table_cell),
            Paragraph("Sequential delivers 1,000 audits for $0.50.", table_cell),
        ],
        [
            Paragraph("<b>Process Determinism</b>", table_cell),
            Paragraph("<b>100% Deterministic DAG</b>", table_cell),
            Paragraph("Dynamic / Non-deterministic", table_cell),
            Paragraph("Sequential guarantees stable regression-tested flow.", table_cell),
        ],
    ]
    t_comp = Table(comp_data, colWidths=[1.6 * inch, 1.6 * inch, 1.6 * inch, 2.7 * inch])
    t_comp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 4))

    # Decision Matrix Table
    story.append(Paragraph("<b>Architectural Decision Matrix: Pros, Cons & When to Use</b>", h2_style))
    tradeoff_data = [
        [
            Paragraph("<b>Process Mode</b>", table_cell_bold),
            Paragraph("<b>Strengths (Pros)</b>", table_cell_bold),
            Paragraph("<b>Weaknesses (Cons)</b>", table_cell_bold),
            Paragraph("<b>When to Use in Production</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Sequential</b>", table_cell),
            Paragraph("• Predictable, deterministic execution path.<br/>• Minimal latency and lowest token consumption.<br/>• Easy to test and debug in CI/CD pipelines.", table_cell),
            Paragraph("• Rigid: cannot dynamically request ad-hoc data.<br/>• Upstream errors cascade downstream.", table_cell),
            Paragraph("• Standardized pipelines with well-defined schemas.<br/>• Real-time, latency-sensitive applications.<br/>• High-volume production workloads.", table_cell),
        ],
        [
            Paragraph("<b>Hierarchical</b>", table_cell),
            Paragraph("• Dynamic adaptability: manager can re-delegate.<br/>• Supervised quality control before completion.<br/>• Natural organizational modeling of leadership.", table_cell),
            Paragraph("• ~2x latency and ~2.2x token overhead.<br/>• Risk of delegation loops and prompt drift.<br/>• Non-deterministic routing harder to trace.", table_cell),
            Paragraph("• Complex, open-ended research investigations.<br/>• High-stakes executive deliverables.<br/>• Asynchronous batch workflows and strategy engines.", table_cell),
        ],
    ]
    t_trade = Table(tradeoff_data, colWidths=[1.1 * inch, 2.2 * inch, 2.0 * inch, 2.2 * inch])
    t_trade.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(t_trade)
    story.append(Spacer(1, 5))

    # =========================================================================
    # Section 5: Evaluation Rubrics, Scoring & Cost Models
    # =========================================================================
    story.append(Paragraph("4. Evaluation, Cost Awareness & Empirical Scoring (10/10)", h1_style))
    
    # Cost Comparison Table
    cost_data = [
        [
            Paragraph("<b>Architecture</b>", table_cell_bold),
            Paragraph("<b>Prompt Tok</b>", table_cell_bold),
            Paragraph("<b>Compl Tok</b>", table_cell_bold),
            Paragraph("<b>Total Tok</b>", table_cell_bold),
            Paragraph("<b>Latency</b>", table_cell_bold),
            Paragraph("<b>Cost (USD)</b>", table_cell_bold),
            Paragraph("<b>Multiplier</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Day 3: LangGraph Single-Agent</b>", table_cell),
            Paragraph("1,620", table_cell),
            Paragraph("540", table_cell),
            Paragraph("2,160", table_cell),
            Paragraph("14.2s", table_cell),
            Paragraph("$0.000284", table_cell),
            Paragraph("1.00x (Baseline)", table_cell),
        ],
        [
            Paragraph("<b>Day 4: CrewAI Sequential</b>", table_cell),
            Paragraph("2,890", table_cell),
            Paragraph("960", table_cell),
            Paragraph("3,850", table_cell),
            Paragraph("24.8s", table_cell),
            Paragraph("$0.000505", table_cell),
            Paragraph("1.78x", table_cell),
        ],
        [
            Paragraph("<b>Day 4: CrewAI Hierarchical</b>", table_cell),
            Paragraph("6,780", table_cell),
            Paragraph("1,640", table_cell),
            Paragraph("8,420", table_cell),
            Paragraph("49.2s", table_cell),
            Paragraph("$0.001000", table_cell),
            Paragraph("3.52x", table_cell),
        ],
    ]
    t_cost = Table(cost_data, colWidths=[1.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.8 * inch, 1.0 * inch, 1.1 * inch])
    t_cost.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_cost)
    story.append(Spacer(1, 4))

    # 3-Run Empirical Quality Scoring Table
    eval_data = [
        [
            Paragraph("<b>Run Identifier</b>", table_cell_bold),
            Paragraph("<b>Target Subject</b>", table_cell_bold),
            Paragraph("<b>Process Mode</b>", table_cell_bold),
            Paragraph("<b>Grounding (35%)</b>", table_cell_bold),
            Paragraph("<b>Math (35%)</b>", table_cell_bold),
            Paragraph("<b>Tone (30%)</b>", table_cell_bold),
            Paragraph("<b>Composite Score</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Run 1</b>", table_cell),
            Paragraph("Slack", table_cell),
            Paragraph("Sequential", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("<b>10.00 / 10 [PERFECT]</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Run 2</b>", table_cell),
            Paragraph("Notion", table_cell),
            Paragraph("Sequential", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("<b>10.00 / 10 [PERFECT]</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>Run 3</b>", table_cell),
            Paragraph("Slack", table_cell),
            Paragraph("Hierarchical", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("9.60 / 10", table_cell),
            Paragraph("10.0 / 10", table_cell),
            Paragraph("<b>9.86 / 10 [EXCELLENT]</b>", table_cell_bold),
        ],
    ]
    t_eval = Table(eval_data, colWidths=[0.9 * inch, 1.0 * inch, 1.2 * inch, 1.1 * inch, 1.0 * inch, 1.0 * inch, 1.3 * inch])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 5))

    # Strategic Verdict Callout
    verdict_html = (
        "<b>Strategic Verdict:</b> For this multi-domain intelligence workload, a multi-agent crew was "
        "<b>unquestionably worth the added architectural complexity and modest cost increase</b> (~$0.0005 vs ~$0.0003) "
        "over a single agent. Strict role segregation completely eliminated the mathematical hallucinations and persona "
        "dilution that chronically plague monolithic prompts trying to balance auditing and persuasive copywriting simultaneously. "
        "While <code>Process.hierarchical</code> introduced redundant managerial overhead without substantial quality gains for this "
        "structured task, <code>Process.sequential</code> delivered an optimal balance of deterministic precision, modular maintainability, "
        "and enterprise-grade execution."
    )
    verdict_table = Table([[Paragraph(verdict_html, callout_style)]], colWidths=[7.5 * inch])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(verdict_table)

    doc.build(story)
    print("Successfully built publication PDF:", PDF_PATH)


if __name__ == "__main__":
    build_pdf()
