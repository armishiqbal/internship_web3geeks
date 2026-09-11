"""Generate a publication-grade 2-page PDF Executive Report for Day 5 Capstone using ReportLab."""

from __future__ import annotations

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
    HRFlowable,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

HERE = Path(__file__).resolve().parent
PDF_PATH = HERE / "day5_executive_report.pdf"


def build_executive_pdf():
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.4 * inch,
        rightMargin=0.4 * inch,
        topMargin=0.3 * inch,
        bottomMargin=0.3 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=1,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=3,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11.5,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=3,
        spaceAfter=1.5,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=2,
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.2,
        leading=7.8,
        textColor=colors.HexColor("#1A202C"),
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.2,
        leading=7.8,
        textColor=colors.HexColor("#1A202C"),
    )

    story = []

    # =========================================================================
    # PAGE 1: STRATEGIC VISION, ARCHITECTURE & FRAMEWORK CHOICE
    # =========================================================================
    story.append(Paragraph("Week 2 Day 5 Capstone — Executive Report", title_style))
    story.append(Paragraph("<b>System:</b> Web3Geeks Autonomous Client Onboarding & Scoping Agent | <b>Author:</b> Armish Iqbal", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2B6CB0"), spaceAfter=4))

    # Executive Summary Box
    summary_html = (
        "<b>Executive Summary:</b> We design, implement, evaluate, and deploy a production-grade autonomous agent system for "
        "<b>Web3Geeks</b>. The system screens inbound client inquiries, audits service catalog rate cards, calculates deterministic "
        "AST milestone payments (40%/40%/20%), synthesizes client proposals, self-corrects via a cyclic critique loop, and halts "
        "for Human-in-the-Loop (HITL) partner approval before dispatching legally binding contracts. Exposed via high-performance FastAPI "
        "REST endpoints with telemetry logging and an SRE monitoring runbook."
    )
    summary_table = Table([[Paragraph(summary_html, callout_style)]], colWidths=[7.6 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 3))

    # Section 1: Business Goal
    story.append(Paragraph("1. Business Goal & Freelancing / Agency Context", h1_style))
    story.append(Paragraph(
        "In enterprise software agencies like Web3Geeks, technical scoping and proposal drafting consumes 6–10 hours per qualified deal. "
        "Manual mental math frequently produces pricing discrepancies between milestone payments and total contract sums. "
        "The objective was to compress proposal turnaround from <b>48 hours to under 2 seconds</b>, enforce 100% adherence to verified rate cards, "
        "eliminate arithmetic hallucinations, and maintain strict partner oversight on consequential commitments.",
        body_style,
    ))

    # Section 2: System Architecture
    story.append(Paragraph("2. System Architecture & Component Design", h1_style))
    arch_data = [
        [
            Paragraph("<b>Pipeline Stage / Node</b>", table_cell_bold),
            Paragraph("<b>Core Function & Responsibilities</b>", table_cell_bold),
            Paragraph("<b>Assigned Tool / Data Source</b>", table_cell_bold),
            Paragraph("<b>Failure Defense & Governance</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>1. validate_inquiry</b>", table_cell),
            Paragraph("Screens length, schema, and adversarial prompt injection strings.", table_cell),
            Paragraph("Defensive regex & keyword filters", table_cell),
            Paragraph("Rejects injections and sub-$3,500 budgets; routes to failure handler.", table_cell),
        ],
        [
            Paragraph("<b>2. scope_services</b>", table_cell),
            Paragraph("Matches client inquiry against 5 Web3 service offerings and SLAs.", table_cell),
            Paragraph("<code>ServiceCatalogSearchTool</code><br/>(<code>services_catalog.json</code>)", table_cell),
            Paragraph("Falls back to standard full-stack dApp package on ambiguous queries.", table_cell),
        ],
        [
            Paragraph("<b>3. calculate_budget</b>", table_cell),
            Paragraph("Computes deterministic milestone splits: 40% (M1), 40% (M2), 20% (M3).", table_cell),
            Paragraph("<code>DeterministicBudgetCalculator</code><br/>(Safe Python AST)", table_cell),
            Paragraph("Zero <code>eval()</code> risk; eliminates mental arithmetic hallucinations.", table_cell),
        ],
        [
            Paragraph("<b>4. generate_proposal</b>", table_cell),
            Paragraph("Drafts 5-section enterprise Statement of Work (SOW).", table_cell),
            Paragraph("Gemini Flash (Temp: 0.3)", table_cell),
            Paragraph("Falls back to deterministic template during upstream API outages.", table_cell),
        ],
        [
            Paragraph("<b>5. critique_proposal</b>", table_cell),
            Paragraph("Audits proposal structure, budget anchors, and mandatory headers.", table_cell),
            Paragraph("<code>ContractTemplateFormatter</code>", table_cell),
            Paragraph("Triggers cyclic revision loop if score &lt; 80 (bounded to max 2 loops).", table_cell),
        ],
        [
            Paragraph("<b>6. human_checkpoint</b>", table_cell),
            Paragraph("<b>Consequential Action Gate:</b> Requires partner sign-off.", table_cell),
            Paragraph("FastAPI <code>POST /api/v1/approve</code>", table_cell),
            Paragraph("Blocks automated contract dispatch until human approves.", table_cell),
        ],
        [
            Paragraph("<b>7. failure_handler</b>", table_cell),
            Paragraph("Emits structured diagnostic rejection notices cleanly.", table_cell),
            Paragraph("Deterministic error response", table_cell),
            Paragraph("Zero token waste, zero cost, and zero unauthorized agency liability.", table_cell),
        ],
    ]
    t_arch = Table(arch_data, colWidths=[1.3 * inch, 2.4 * inch, 1.8 * inch, 2.1 * inch])
    t_arch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 2))

    # Section 3: Framework Choice Justification
    story.append(Paragraph("3. Framework Choice Justification (LangGraph vs. CrewAI vs. Raw Loop)", h1_style))
    story.append(Paragraph(
        "<b>Architectural Rationale:</b> We selected <b>LangGraph</b> as the core state-machine orchestrator, incorporating CrewAI's least-privilege "
        "role confinement principles. While autonomous role-playing frameworks (CrewAI) excel at conversational brainstorming, enterprise commercial contracts "
        "demand <b>strict state determinism, bounded cyclic self-correction, and first-class Human-in-the-Loop (HITL) interrupt checkpoints</b>. "
        "A raw Python loop lacks state persistence, interrupt-and-resume semantics, and standardized DAG routing. LangGraph provides the exact control-flow "
        "guarantees needed to prevent unauthorized financial commitments.",
        body_style,
    ))

    # Force clean page break to Page 2
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: EVALUATION, OBSERVABILITY, LIMITATIONS & ROADMAP
    # =========================================================================
    story.append(Paragraph("4. Empirical Evaluation Results (8-Point Test Suite)", h1_style))
    story.append(Paragraph(
        "The system was evaluated across 8 diverse production scenarios, including standard client requests, rush multi-service inquiries, "
        "budget floor boundary tests, and adversarial jailbreak attempts. All runs were scored against 5 weighted criteria: "
        "Task Success (25%), Factual Accuracy (25%), Quantitative Rigor (20%), Tone (15%), and Safety (15%).",
        body_style,
    ))

    eval_data = [
        [
            Paragraph("<b>ID</b>", table_cell_bold),
            Paragraph("<b>Test Case Scenario</b>", table_cell_bold),
            Paragraph("<b>Category</b>", table_cell_bold),
            Paragraph("<b>Terminal Status</b>", table_cell_bold),
            Paragraph("<b>Composite</b>", table_cell_bold),
            Paragraph("<b>Latency</b>", table_cell_bold),
            Paragraph("<b>Cost ($)</b>", table_cell_bold),
            Paragraph("<b>Verdict</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-1</b>", table_cell),
            Paragraph("Smart Contract Staking Audit", table_cell),
            Paragraph("Standard Inbound", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("1.89s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-2</b>", table_cell),
            Paragraph("Full-Stack Web3 dApp", table_cell),
            Paragraph("Frontend Integration", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("0.77s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-3</b>", table_cell),
            Paragraph("Tokenomics & Emissions Sim", table_cell),
            Paragraph("Quantitative Modeling", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("1.33s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-4</b>", table_cell),
            Paragraph("The Graph Subgraph Indexer", table_cell),
            Paragraph("Data Infrastructure", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("0.75s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-5</b>", table_cell),
            Paragraph("Multi-Service Suite (Rush)", table_cell),
            Paragraph("Expedited Multi-Service", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("1.33s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-6</b>", table_cell),
            Paragraph("DAO Governance & Treasury", table_cell),
            Paragraph("Governance Setup", table_cell),
            Paragraph("<code>passed</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("1.23s", table_cell),
            Paragraph("$0.000199", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-7</b>", table_cell),
            Paragraph("Sub-Minimum Budget Inquiry", table_cell),
            Paragraph("Edge Case (Budget)", table_cell),
            Paragraph("<code>failed_low_budget</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("0.02s", table_cell),
            Paragraph("$0.000000", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
        [
            Paragraph("<b>TC-8</b>", table_cell),
            Paragraph("Adversarial Prompt Injection", table_cell),
            Paragraph("Adversarial Defense", table_cell),
            Paragraph("<code>rejected_adversarial</code>", table_cell),
            Paragraph("<b>10.00 / 10</b>", table_cell),
            Paragraph("0.03s", table_cell),
            Paragraph("$0.000000", table_cell),
            Paragraph("<b>PASS</b>", table_cell_bold),
        ],
    ]
    t_eval = Table(eval_data, colWidths=[0.5 * inch, 1.8 * inch, 1.4 * inch, 1.2 * inch, 0.7 * inch, 0.6 * inch, 0.7 * inch, 0.7 * inch])
    t_eval.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 1.8),
    ]))
    story.append(t_eval)
    story.append(Spacer(1, 3))

    # Section 5: Failure Pattern & Fix
    story.append(Paragraph("5. Failure Pattern Analysis & Concrete Fix", h1_style))
    story.append(Paragraph(
        "<b>Failure Mode:</b> Early iterations suffered from <i>milestone drift</i>: the LLM guessed mental math in text prompts, causing milestone sums to diverge from the total commitment. "
        "<b>Concrete Fix:</b> We decoupled financial calculations into <code>calculate_budget_node</code> using safe Python AST arithmetic. "
        "Milestones are evaluated strictly as 40% (M1), 40% (M2), and 20% (M3) and injected as an immutable Markdown anchor table. "
        "Mathematical discrepancies dropped to 0.00% across all runs.",
        body_style,
    ))

    # Section 6: API & Monitoring Runbook
    story.append(Paragraph("6. Production API & Monitoring Runbook", h1_style))
    story.append(Paragraph(
        "<b>FastAPI Service:</b> <code>POST /api/v1/onboard</code>, <code>POST /api/v1/approve</code>, <code>GET /health</code>, and <code>GET /api/v1/metrics</code>.<br/>"
        "<b>Alerting Thresholds:</b> (1) <b>Error Rate:</b> Alert if &gt; 2.0% in 5m; (2) <b>Latency P95:</b> Alert if &gt; 5.0s; "
        "(3) <b>Cost Drift:</b> Alert if &gt; $0.0010/req; (4) <b>Adversarial Throttle:</b> Trigger WAF block if &gt; 10 injections/hr.<br/>"
        "<b>Cadence:</b> Bi-weekly evaluation suite execution; monthly jailbreak red-teaming; quarterly rate card refresh.",
        body_style,
    ))

    # Section 7: Limitations & Roadmap
    story.append(Paragraph("7. Known Limitations & Recommended Next Steps", h1_style))
    limitations_html = (
        "<b>Known Limitations:</b> (1) Custom hybrid services not in catalog default to standard packages; (2) Context constraints limit monolithic single-prompt SOWs to ~15 pages; "
        "(3) Managing partner availability forms an operational throughput bottleneck.<br/>"
        "<b>Next Steps & Roadmap:</b> (1) <b>Scaling:</b> Transition in-memory state and metrics to Redis and PostgreSQL; "
        "(2) <b>Guardrails:</b> Deploy LlamaGuard / NeMo Guardrails for deep semantic threat detection; "
        "(3) <b>Human Oversight:</b> Launch a Slack bot (<code>/web3-approve &lt;id&gt;</code>) for 1-click mobile partner authorization."
    )
    lim_table = Table([[Paragraph(limitations_html, callout_style)]], colWidths=[7.6 * inch])
    lim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(lim_table)

    doc.build(story)
    print("Successfully built publication PDF:", PDF_PATH)


if __name__ == "__main__":
    build_executive_pdf()
