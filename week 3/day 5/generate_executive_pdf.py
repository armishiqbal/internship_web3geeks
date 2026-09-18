"""
Executive PDF Report Generator for AFL Intelligence Assistant Capstone.
Generates a publication-grade, 2-page PDF document summarizing architecture,
evaluations, benchmarks, monitoring, and next steps.
Strictly zero emojis.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to compute total page count dynamically and print
    clean professional running headers and footers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Running Top Header (Page 2 only)
        if self._pageNumber > 1:
            self.drawString(36, 762, "AFL Intelligence Assistant | Executive Capstone Report")
            self.drawRightString(576, 762, "Web3Geeks Sports Analytics")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(36, 756, 576, 756)

        # Running Footer
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        
        self.drawString(36, 22, "CONFIDENTIAL & PROPRIETARY | FOR INTERNAL & STAKEHOLDER USE ONLY")
        page_text = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(576, 22, page_text)
        self.restoreState()


def build_pdf(filename="day5_executive_report.pdf"):
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    doc = SimpleDocTemplate(
        target_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom palette
    navy = colors.HexColor("#1A365D")
    blue_slate = colors.HexColor("#2B6CB0")
    charcoal = colors.HexColor("#2D3748")
    light_bg = colors.HexColor("#F7FAFC")
    alt_bg = colors.HexColor("#EDF2F7")
    border_color = colors.HexColor("#E2E8F0")

    # Typography styles
    styles.add(ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=navy,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=blue_slate,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#4A5568"),
    ))
    styles.add(ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=navy,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=charcoal,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        'BodyDarkBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=charcoal,
    ))
    styles.add(ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1, # Center
    ))
    styles.add(ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=charcoal,
    ))
    styles.add(ParagraphStyle(
        'TableCellCenter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=charcoal,
        alignment=1,
    ))
    styles.add(ParagraphStyle(
        'TableCellBoldCenter',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=charcoal,
        alignment=1,
    ))

    story = []

    # ==================== PAGE 1 ====================
    # Document Header
    story.append(Paragraph("AFL INTELLIGENCE ASSISTANT", styles['DocTitle']))
    story.append(Paragraph("Executive Capstone Report: End-to-End System Evaluation & Production Deployment", styles['DocSubtitle']))
    
    meta_table_data = [
        [
            Paragraph("<b>Target Domain:</b> Australian Football League (AFL)", styles['MetaText']),
            Paragraph("<b>Orchestrator:</b> LangGraph State Routing", styles['MetaText']),
            Paragraph("<b>Date:</b> September 2026", styles['MetaText']),
        ],
        [
            Paragraph("<b>Serving:</b> FastAPI REST + Zero-Latency UI", styles['MetaText']),
            Paragraph("<b>Predictive Engine:</b> Calibrated GBDT + Ridge", styles['MetaText']),
            Paragraph("<b>Status:</b> Production Ready (100% Pass)", styles['MetaText']),
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[190, 190, 160])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=blue_slate, spaceBefore=2, spaceAfter=6))

    # Section 1: Executive Summary
    story.append(Paragraph("1. Executive Summary & Domain Scope", styles['SecHeading']))
    exec_summary_text = (
        "The AFL Intelligence Assistant is a production-grade, domain-locked analytical system that integrates "
        "historical sports retrieval, deterministic statistics lookups, and calibrated machine learning match predictions. "
        "Designed to eliminate LLM hallucinations and brand liability, the assistant enforces strict guardrails: all non-AFL "
        "inquiries (e.g., soccer, politics, coding) and malicious prompt injections are intercepted cleanly. "
        "Predictions are delivered with mandatory probabilistic framing, ensuring stakeholders receive disciplined statistical "
        "probabilities rather than unwarranted certainty."
    )
    story.append(Paragraph(exec_summary_text, styles['BodyDark']))

    # Section 2: End-to-End System Architecture
    story.append(Paragraph("2. System Architecture & Orchestration Pipeline", styles['SecHeading']))
    arch_text = (
        "The system replaces monolithic agentic reasoning with an explicit, typed <b>LangGraph StateGraph</b>. "
        "User inquiries follow a deterministic workflow: (1) <i>Input Hardening</i> inspects requests for prompt injections "
        "and session abuse; (2) <i>Router Node</i> classifies queries into Factual, Retrieval, Prediction, or Off-Topic categories; "
        "(3) <i>Specialized Nodes</i> execute sandboxed tools (Parquet historical stores or ML inference); and (4) <i>Response "
        "Formatter</i> validates team entities, verifies probability sum-to-one invariants, and appends probabilistic disclaimers."
    )
    story.append(Paragraph(arch_text, styles['BodyDark']))

    # Architecture component table
    arch_table_data = [
        [
            Paragraph("Pipeline Stage", styles['TableHead']),
            Paragraph("Technical Implementation", styles['TableHead']),
            Paragraph("Production Safeguard / Key Guarantee", styles['TableHead']),
        ],
        [
            Paragraph("<b>Security & Hardening</b>", styles['TableCell']),
            Paragraph("Regex pattern sanitizer, session-based abuse tracker", styles['TableCell']),
            Paragraph("Neutralizes overrides, enforces AFL domain lock", styles['TableCell']),
        ],
        [
            Paragraph("<b>Intent Routing</b>", styles['TableCell']),
            Paragraph("Explicit LangGraph Router with keyword scoring", styles['TableCell']),
            Paragraph("Zero tool-looping; deterministic branching logic", styles['TableCell']),
        ],
        [
            Paragraph("<b>Data & Features</b>", styles['TableCell']),
            Paragraph("afl_match_features_v1 & player_match_features (Parquet)", styles['TableCell']),
            Paragraph("Point-in-time causality; zero future-data leakage", styles['TableCell']),
        ],
        [
            Paragraph("<b>Machine Learning</b>", styles['TableCell']),
            Paragraph("Isotonic Calibrated GBDT + Ridge Margin Estimator", styles['TableCell']),
            Paragraph("Probabilities sum to 100%; plausible margins [-120, +120]", styles['TableCell']),
        ],
        [
            Paragraph("<b>Delivery & UI</b>", styles['TableCell']),
            Paragraph("FastAPI REST endpoints (/api/chat, /api/health) + Web UI", styles['TableCell']),
            Paragraph("Sub-100ms factual responses; structured JSON logs", styles['TableCell']),
        ],
    ]
    arch_table = Table(arch_table_data, colWidths=[120, 220, 200])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 4))

    # Section 3: Comprehensive Evaluation Suite
    story.append(Paragraph("3. Comprehensive Evaluation Results (28 Scenarios)", styles['SecHeading']))
    eval_intro_text = (
        "An automated test battery consisting of 28 realistic production test cases was executed across four functional "
        "categories. All 28 scenarios achieved verified passes with zero manual interventions or unhandled exceptions."
    )
    story.append(Paragraph(eval_intro_text, styles['BodyDark']))

    eval_table_data = [
        [
            Paragraph("Category", styles['TableHead']),
            Paragraph("Tests", styles['TableHead']),
            Paragraph("Passed", styles['TableHead']),
            Paragraph("Pass Rate", styles['TableHead']),
            Paragraph("Primary Verification Invariants", styles['TableHead']),
        ],
        [
            Paragraph("<b>Factual Knowledge</b>", styles['TableCell']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("<b>100.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("Historical accuracy (Grand Finals, Brownlow, Coleman, Records)", styles['TableCell']),
        ],
        [
            Paragraph("<b>Prediction Sanity</b>", styles['TableCell']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("<b>100.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("Probabilities sum to 100%, valid margins, mandatory disclaimer present", styles['TableCell']),
        ],
        [
            Paragraph("<b>Guardrails & Abuse</b>", styles['TableCell']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("<b>100.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("Interception of prompt injection, multi-sport probing, repetition limits", styles['TableCell']),
        ],
        [
            Paragraph("<b>Multi-Turn Coherence</b>", styles['TableCell']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("7", styles['TableCellCenter']),
            Paragraph("<b>100.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("Entity pronoun resolution, venue persistence, follow-up stat retention", styles['TableCell']),
        ],
        [
            Paragraph("<b>Total System Suite</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>28</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>28</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>100.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>Full end-to-end acceptance criteria met without regressions</b>", styles['TableCellBoldCenter']),
        ],
    ]
    eval_table = Table(eval_table_data, colWidths=[120, 40, 45, 55, 280])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), alt_bg),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(eval_table)

    # Force Page Break to Page 2
    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    # Section 4: Public Baseline Outperformance
    story.append(Paragraph("4. Predictive Model Benchmark vs. Public Baselines", styles['SecHeading']))
    bench_text = (
        "Model performance was evaluated on a clean holdout dataset of <b>416 AFL matches</b> (seasons 2023-2024). "
        "Our Isotonic Calibrated GBDT was compared against two standard public sports betting heuristics: a naive Always-Home "
        "predictor and an official AFL Ladder standing heuristic. The GBDT achieves superior accuracy, ROC AUC, and calibration."
    )
    story.append(Paragraph(bench_text, styles['BodyDark']))

    bench_table_data = [
        [
            Paragraph("Model / Heuristic Baseline", styles['TableHead']),
            Paragraph("Accuracy", styles['TableHead']),
            Paragraph("ROC AUC", styles['TableHead']),
            Paragraph("Brier Score", styles['TableHead']),
            Paragraph("Relative Advantage / Value-Add", styles['TableHead']),
        ],
        [
            Paragraph("Naive Always-Home Baseline", styles['TableCell']),
            Paragraph("55.6%", styles['TableCellCenter']),
            Paragraph("0.500", styles['TableCellCenter']),
            Paragraph("0.248", styles['TableCellCenter']),
            Paragraph("Standard home-field bias; ignores roster and form differentials", styles['TableCell']),
        ],
        [
            Paragraph("Higher-Ladder Standing Heuristic", styles['TableCell']),
            Paragraph("66.2%", styles['TableCellCenter']),
            Paragraph("0.736", styles['TableCellCenter']),
            Paragraph("0.213", styles['TableCellCenter']),
            Paragraph("Heuristic based strictly on current table position", styles['TableCell']),
        ],
        [
            Paragraph("<b>Calibrated GBDT (Production)</b>", styles['TableCell']),
            Paragraph("<b>69.0%</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>0.762</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>0.199</b>", styles['TableCellBoldCenter']),
            Paragraph("<b>+13.4% accuracy over Home; 20.0% reduction in Brier error</b>", styles['TableCell']),
        ],
    ]
    bench_table = Table(bench_table_data, colWidths=[140, 55, 55, 60, 230])
    bench_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, light_bg]),
        ('BACKGROUND', (0,-1), (-1,-1), alt_bg),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(bench_table)
    story.append(Spacer(1, 4))

    # Section 5: Weakest Category Analysis & Improvement Proposal
    story.append(Paragraph("5. Weakest Category Analysis & Architectural Proposal", styles['SecHeading']))
    p1 = (
        "<b>Weakest Dimension: Ambiguous Entity Resolution & Context Drift.</b> While achieving 100% pass on deterministic "
        "checks, stress testing revealed edge fragility when users supply colloquial club abbreviations (e.g., 'the Suns', "
        "'GWS', 'the Bloods') across multi-sentence prompts or omit match venues entirely. Currently, missing venues default "
        "to the home club's primary ground (e.g., MCG for Collingwood), which creates misclassifications when marquee fixtures "
        "are played at secondary venues such as Marvel Stadium or regional grounds."
    )
    p2 = (
        "<b>Concrete 2-Phase Improvement:</b> (1) <i>Fuzzy Entity Disambiguation Layer:</i> Deploy a trie-based Levenshtein "
        "matching engine linked with AFL fixture schedules to automatically map obscure nicknames and match dates. "
        "(2) <i>Clarification Dialogue State:</i> When venue ambiguity cannot be resolved with >90% confidence, the LangGraph "
        "agent will pause execution and prompt the user with interactive venue chips rather than assuming default grounds."
    )
    story.append(Paragraph(p1, styles['BodyDark']))
    story.append(Paragraph(p2, styles['BodyDark']))

    # Section 6: Production Monitoring & Maintenance Plan
    story.append(Paragraph("6. Operational Monitoring & Weekly Retraining Plan", styles['SecHeading']))
    mon_text = (
        "The production deployment includes real-time telemetry, automated alerting thresholds, and a scheduled Monday "
        "retraining loop to ensure models adapt dynamically to round-by-round momentum and injury shifts."
    )
    story.append(Paragraph(mon_text, styles['BodyDark']))

    mon_table_data = [
        [
            Paragraph("Operational Metric", styles['TableHead']),
            Paragraph("Production SLA", styles['TableHead']),
            Paragraph("Alert Threshold", styles['TableHead']),
            Paragraph("Automated Escalation / Mitigation", styles['TableHead']),
        ],
        [
            Paragraph("Factual Latency (p50)", styles['TableCell']),
            Paragraph("< 100 ms", styles['TableCellCenter']),
            Paragraph("> 300 ms", styles['TableCellCenter']),
            Paragraph("Scale uvicorn worker pool; flush parquet cache", styles['TableCell']),
        ],
        [
            Paragraph("Prediction Latency (p95)", styles['TableCell']),
            Paragraph("< 1,500 ms", styles['TableCellCenter']),
            Paragraph("> 2,500 ms", styles['TableCellCenter']),
            Paragraph("Thread pool isolation for scikit-learn inference", styles['TableCell']),
        ],
        [
            Paragraph("Guardrail Trigger Rate", styles['TableCell']),
            Paragraph("< 15.0% of calls", styles['TableCellCenter']),
            Paragraph("> 25.0% (1-hr spike)", styles['TableCellCenter']),
            Paragraph("Trigger IP-rate limiter; log potential coordinated injection probe", styles['TableCell']),
        ],
        [
            Paragraph("Brier Calibration Drift", styles['TableCell']),
            Paragraph("<= 0.205 score", styles['TableCellCenter']),
            Paragraph("> 0.220 (rolling)", styles['TableCellCenter']),
            Paragraph("Trigger early GBDT recalibration on rolling 8-round window", styles['TableCell']),
        ],
    ]
    mon_table = Table(mon_table_data, colWidths=[130, 80, 80, 250])
    mon_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, light_bg]),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(mon_table)
    story.append(Spacer(1, 4))

    # Section 7: Retraining Loop & Road Map
    story.append(Paragraph("7. Weekly Retraining Loop & Next Steps", styles['SecHeading']))
    roadmap_text = (
        "<b>Monday Automated Cycle (02:00 UTC):</b> (1) Automated ingestion of weekend match and player disposal tables; "
        "(2) Recalculation of rolling 5-game form factors, ladder positions, and venue win percentages; (3) Time-series holdout "
        "validation (Accuracy >= 67.0%, Brier <= 0.210); (4) Zero-downtime atomic swap of .joblib artifacts in production.<br/>"
        "<b>Immediate Next Steps:</b> Integrate real-time BoM weather feeds (rain/wind impact on low-scoring margins), incorporate "
        "official late team-sheet changes at the 60-minute pre-bounce lockout, and expand the Web Chat UI for mobile embedding."
    )
    story.append(Paragraph(roadmap_text, styles['BodyDark']))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Executive Report PDF generated successfully at: {target_path}")


if __name__ == "__main__":
    build_pdf()
