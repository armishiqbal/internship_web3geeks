"""
Generate publication-quality, strictly 3-page executive PDF report for Day 5
covering all 9 required sections of Task 6:
1. Problem Definition
2. Data Preparation
3. Model Development
4. Model Tuning
5. Diagnostics
6. Final Results
7. Model Interpretation
8. Production Readiness
9. Limitations & Future Improvements

Calibrated to match pipeline.py, final_metrics.csv, and inference.py.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent

class NumberedCanvas(canvas.Canvas):
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

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(colors.HexColor("#475569"))
        
        # Header (Top)
        self.drawString(36, 762, "DATA SCIENCE INTERNSHIP · WEEK 1 DAY 5 FINAL REPORT")
        self.drawRightString(576, 762, "END-TO-END ML WORKFLOW & PRODUCTION VALIDATION")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.7)
        self.line(36, 756, 576, 756)
        
        # Footer (Bottom)
        self.line(36, 28, 576, 28)
        self.setFont("Helvetica", 7.5)
        self.drawString(36, 18, "Confidential · UCI Adult Census Income · End-to-End ML Pipeline")
        self.drawRightString(576, 18, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf():
    pdf_path = HERE / "day5_final_report.pdf"
    
    # Page setup: Letter = 612 x 792 pt. Margins = 36 pt. Printable width = 540 pt.
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )
    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1E3A5F"),
        spaceBefore=4,
        spaceAfter=3
    )
    h2_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2563EB"),
        spaceBefore=2,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.4,
        leading=10,
        textColor=colors.HexColor("#1E293B")
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=9,
        textColor=colors.HexColor("#0F172A")
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    table_cell_header = ParagraphStyle(
        'TableHeaderCell',
        parent=table_cell,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )
    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.HexColor("#065F46"),
        alignment=1
    )

    story = []

    # =========================================================================
    # PAGE 1: 1. PROBLEM, 2. DATA PREPARATION, 3. DEVELOPMENT, 4. TUNING
    # =========================================================================
    story.append(Paragraph("Week 1 Final Project Report: Production ML Pipeline", title_style))
    story.append(Paragraph("<b>Author:</b> Machine Learning Intern &nbsp;|&nbsp; <b>Dataset:</b> UCI Adult Income (&gt;$50K) &nbsp;|&nbsp; <b>Production Artifact:</b> <code>final_model.joblib</code>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1E3A5F"), spaceBefore=0, spaceAfter=5))
    
    # 1. Problem Definition
    story.append(Paragraph("1. Problem Definition", h1_style))
    p1_text = (
        "The objective of this project is to build an automated, leak-free machine learning classification system that accurately predicts "
        "whether an individual's annual income exceeds $50,000 based on census demographic and employment attributes. Personal income prediction "
        "is a foundational capability in financial underwriting, credit risk assessment, and premium customer acquisition. In targeted outreach, "
        "accurately identifying high-income earners (&gt;$50K) enables cost-effective resource allocation, while misclassifications carry asymmetric "
        "costs: failing to engage an eligible high earner represents substantial lost customer lifetime value compared to the modest cost of a false outreach."
    )
    story.append(Paragraph(p1_text, body_style))
    story.append(Spacer(1, 4))

    # 2. Data Preparation
    story.append(Paragraph("2. Data Preparation & Validation Strategy", h1_style))
    p2_text = (
        "<b>2.1 Dataset & Missing Value Handling:</b> The dataset contains 32,561 records from the 1994 US Census with a 24.08% positive class "
        "prevalence (&gt;$50K). Missing values represented as '?' in <code>workclass</code>, <code>occupation</code>, and <code>native_country</code> "
        "were imputed using most-frequent category replacement, while numeric gaps were imputed using median values.<br/>"
        "<b>2.2 Row-Level Feature Engineering (<code>pipeline.py</code>):</b> To prevent data leakage, domain features were engineered purely within "
        "each row: (1) <code>is_married</code> (flagging civilian/AF married spouses), (2) <code>is_higher_ed</code> (schooling &ge; 13 years), "
        "(3) <code>has_capital_gain</code> and <code>has_capital_loss</code> (binary participation indicators), (4) <code>log_capital_gain</code> "
        "(&plusmn;1 log-scaling), and (5) compounding wage interaction <code>edu_x_hours</code> (<code>education_num &times; hours_per_week</code>).<br/>"
        "<b>2.3 Preprocessing & Splitting:</b> 12 numeric columns were standardized via <code>StandardScaler</code>; 8 categorical columns were "
        "encoded via <code>OneHotEncoder(handle_unknown='ignore')</code>, producing 105 transformed dimensions. The dataset was partitioned via "
        "an 80/20 stratified split (Train: <i>N</i>=26,048, Test: <i>N</i>=6,513, seed=42) with confirmed zero index overlap."
    )
    story.append(Paragraph(p2_text, body_style))
    story.append(Spacer(1, 4))

    # 3. Model Development & Baseline Progression
    story.append(Paragraph("3. Model Development & Baseline Performance", h1_style))
    p3_text = (
        "Model progression evolved across five stages: (a) Day 1 trivial baselines (Majority Class: 0.0% Recall; Education Heuristic: 0.4811 F1), "
        "(b) Day 2 classical baselines (Decision Tree: 0.6194 F1; Logistic Regression: 0.6734 F1), (c) Day 3 gradient-boosted trees establishing "
        "clear superiority in capturing non-linear interactions over linear models, (d) Day 4 multi-model tuning, and (e) Day 5 probability calibration. "
        "The complete progression on identical holdout data is benchmarked below:"
    )
    story.append(Paragraph(p3_text, body_style))
    story.append(Spacer(1, 3))

    # Lifecycle Table
    headers = [
        Paragraph("Model / Lifecycle Phase", table_cell_header),
        Paragraph("Protocol", table_cell_header),
        Paragraph("Accuracy", table_cell_header),
        Paragraph("Precision", table_cell_header),
        Paragraph("Recall", table_cell_header),
        Paragraph("F1-Score", table_cell_header),
        Paragraph("ROC-AUC", table_cell_header),
        Paragraph("Brier", table_cell_header),
        Paragraph("Status", table_cell_header),
    ]
    table_rows = [
        headers,
        [Paragraph("Day 1 Majority (<=50K)", table_cell), Paragraph("Holdout Test", table_cell), Paragraph("75.93%", table_cell), Paragraph("0.0%", table_cell), Paragraph("0.0%", table_cell), Paragraph("0.0000", table_cell), Paragraph("0.5000", table_cell), Paragraph("0.2407", table_cell), Paragraph("Baseline", table_cell)],
        [Paragraph("Day 1 Education Heuristic", table_cell), Paragraph("Holdout Test", table_cell), Paragraph("74.67%", table_cell), Paragraph("47.46%", table_cell), Paragraph("48.79%", table_cell), Paragraph("0.4811", table_cell), Paragraph("0.6583", table_cell), Paragraph("—", table_cell), Paragraph("Baseline", table_cell)],
        [Paragraph("Day 2 Logistic Regression", table_cell), Paragraph("Holdout Test", table_cell), Paragraph("85.58%", table_cell), Paragraph("74.06%", table_cell), Paragraph("61.73%", table_cell), Paragraph("0.6734", table_cell), Paragraph("0.9078", table_cell), Paragraph("—", table_cell), Paragraph("Shortlist", table_cell)],
        [Paragraph("Day 2 Decision Tree", table_cell), Paragraph("Holdout Test", table_cell), Paragraph("81.04%", table_cell), Paragraph("59.93%", table_cell), Paragraph("64.09%", table_cell), Paragraph("0.6194", table_cell), Paragraph("0.7525", table_cell), Paragraph("—", table_cell), Paragraph("Shortlist", table_cell)],
        [Paragraph("Day 4 Tuned Logistic (L1)", table_cell), Paragraph("5-Fold Train CV", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("0.6635", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("Tuned", table_cell)],
        [Paragraph("Day 4 Tuned Random Forest", table_cell), Paragraph("5-Fold Train CV", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("0.6876", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("Tuned", table_cell)],
        [Paragraph("Day 4 Tuned HGB (Winner)", table_cell), Paragraph("5-Fold Train CV", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("0.7150", table_cell), Paragraph("—", table_cell), Paragraph("—", table_cell), Paragraph("Winner", table_cell)],
        [Paragraph("Day 4 Saved Artifact", table_cell), Paragraph("Artifact Record", table_cell), Paragraph("86.75%", table_cell), Paragraph("71.22%", table_cell), Paragraph("75.45%", table_cell), Paragraph("0.7327", table_cell), Paragraph("0.9297", table_cell), Paragraph("0.0872", table_cell), Paragraph("Day 4 Record", table_cell)],
        [Paragraph("<b>FINAL MODEL (Pipeline @ 0.40)</b>", table_cell_bold), Paragraph("Live Holdout", table_cell_bold), Paragraph("<b>86.75%</b>", table_cell_bold), Paragraph("<b>71.32%</b>", table_cell_bold), Paragraph("<b>75.19%</b>", table_cell_bold), Paragraph("<b>0.7321</b>", table_cell_bold), Paragraph("<b>0.9297</b>", table_cell_bold), Paragraph("<b>0.0871</b>", table_cell_bold), Paragraph("<b>SELECTED</b>", badge_style)]
    ]
    t_bench = Table(table_rows, colWidths=[120, 60, 42, 42, 42, 45, 45, 38, 52])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#DCFCE7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#1E3A5F")),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 4))

    # 4. Model Tuning
    story.append(Paragraph("4. Model Tuning & Hyperparameter Optimization", h1_style))
    p4_text = (
        "A controlled 5-fold cross-validated <code>RandomizedSearchCV</code> was conducted across three model families: HistGradientBoosting "
        "(40 iterations), Random Forest (30 iterations), and L1 Logistic Regression (30 iterations). HistGradientBoosting won decisively with "
        "CV F1 = <b>0.7150</b> (+2.74% over RF, +5.15% over LR). The winning configuration identified: <code>learning_rate=0.15</code>, "
        "<code>max_iter=300</code>, <code>max_leaf_nodes=31</code>, <code>min_samples_leaf=50</code>, and <code>l2_regularization=5.0</code>. "
        "The heavy L2 regularization and 50-sample leaf minimum effectively curbed tree overfitting while maintaining sharp split boundaries."
    )
    story.append(Paragraph(p4_text, body_style))

    # Page 1 Break
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: 5. DIAGNOSTICS & 6. FINAL RESULTS & ERROR ANALYSIS
    # =========================================================================
    story.append(Paragraph("5. Model Diagnostics: Learning Curves & Probability Calibration", h1_style))
    p5_text = (
        "<b>5.1 Bias vs. Variance Diagnostics:</b> An 8-point learning curve evaluation revealed that the training-validation F1 gap began at "
        "35.81% (scarce data, <i>N</i>=2,083) and steadily contracted to <b>4.10%</b> at full training scale (<i>N</i>=20,838; Train F1=0.7513, "
        "Val F1=0.7103). The diagnostic status is officially certified as <b>Reasonably Balanced</b>, confirming optimal sample efficiency without "
        "runaway variance.<br/>"
        "<b>5.2 Calibration & Cutoff Selection:</b> Platt Sigmoid scaling (<code>CalibratedClassifierCV</code>, cv=5) improved out-of-fold Brier Score "
        "from 0.0904 to 0.0900. A decision threshold sweep identified <b>&tau; = 0.40</b> as optimal, capturing +8.42% additional recall (+528 high earners) "
        "and reducing false negatives by -24.5% compared to default 0.50 cutoff."
    )
    story.append(Paragraph(p5_text, body_style))
    story.append(Spacer(1, 4))

    diag_img_path = HERE / "calibration_roc_pr.png"
    if diag_img_path.is_file():
        story.append(Paragraph("<b>Figure 1:</b> Diagnostic Panels — Reliability Curve (Platt Calibrated), ROC Curve (AUC=0.9297), and PR Curve (AUC=0.8335)", h2_style))
        story.append(Image(str(diag_img_path), width=530, height=130))
        story.append(Spacer(1, 4))

    # 6. Final Results & Error Analysis
    story.append(Paragraph("6. Final Results, Confusion Matrix & Error Analysis", h1_style))
    p6_intro = (
        "On the untouched holdout test set (<i>N</i>=6,513), the calibrated pipeline achieved <b>86.75% Accuracy</b>, <b>71.32% Precision</b>, "
        "<b>75.19% Recall</b>, <b>0.7321 F1-Score</b>, <b>0.9297 ROC-AUC</b>, and <b>0.0871 Brier Score</b>. The model committed 863 total errors "
        "(13.25% error rate): <b>474 False Positives</b> and <b>389 False Negatives</b>."
    )
    story.append(Paragraph(p6_intro, body_style))
    story.append(Spacer(1, 3))

    cm_img_path = HERE / "confusion_matrix.png"
    if cm_img_path.is_file():
        cm_panel_data = [
            [
                Image(str(cm_img_path), width=210, height=140),
                [
                    Paragraph("<b>Operational Metrics & Asymmetric Cost Model</b>", h2_style),
                    Table([
                        [Paragraph("True Negatives (TN)", table_cell_bold), Paragraph("4,471 (90.41% Specificity)", table_cell)],
                        [Paragraph("False Positives (FP)", table_cell_bold), Paragraph("474 (9.59% Fall-out)", table_cell)],
                        [Paragraph("False Negatives (FN)", table_cell_bold), Paragraph("389 (24.81% Miss Rate)", table_cell)],
                        [Paragraph("True Positives (TP)", table_cell_bold), Paragraph("1,179 (75.19% Sensitivity)", table_cell)],
                        [Paragraph("Operating Cutoff (&tau;)", table_cell_bold), Paragraph("0.40 (Platt Calibrated)", table_cell)],
                        [Paragraph("Net Cost Reduction", table_cell_bold), Paragraph("<b>19.6% Saved vs 0.50</b>", table_cell)]
                    ], colWidths=[110, 115], style=[
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
                    ]),
                    Spacer(1, 3),
                    Paragraph("<b>Error Archetypes:</b><br/>"
                              "• <i>'Paper-Affluent' FP (474 cases):</i> Married individuals in clerical/public sector roles whose combined demographic score (<code>is_married=1</code>, <code>education_num=10</code>) pushes probability above 0.40 despite modest wage scales.<br/>"
                              "• <i>'Uncredentialed Earner' FN (389 cases):</i> Unmarried tradesmen working 50–70 hrs/week who earn &gt;$50K on individual merit but lack the marriage household signal.", body_style)
                ]
            ]
        ]
        t_cm_panel = Table(cm_panel_data, colWidths=[220, 310])
        t_cm_panel.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 1),
            ('RIGHTPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(t_cm_panel)
        story.append(Spacer(1, 3))

    # Subgroup Fairness
    story.append(Paragraph("<b>Demographic Subgroup Fairness Audit:</b>", h2_style))
    sub_headers = [
        Paragraph("Cohort", table_cell_header),
        Paragraph("Sample (N)", table_cell_header),
        Paragraph("Base Rate", table_cell_header),
        Paragraph("Recall (TPR)", table_cell_header),
        Paragraph("Precision (PPV)", table_cell_header),
        Paragraph("FP Share", table_cell_header),
        Paragraph("FN Share", table_cell_header),
    ]
    sub_data = [
        sub_headers,
        [Paragraph("Female", table_cell), Paragraph("2,158", table_cell), Paragraph("11.35%", table_cell), Paragraph("72.65%", table_cell), Paragraph("73.55%", table_cell), Paragraph("2.97%", table_cell), Paragraph("3.10%", table_cell)],
        [Paragraph("Male", table_cell), Paragraph("4,355", table_cell), Paragraph("30.38%", table_cell), Paragraph("75.66%", table_cell), Paragraph("70.94%", table_cell), Paragraph("9.41%", table_cell), Paragraph("7.39%", table_cell)],
        [Paragraph("White", table_cell), Paragraph("5,533", table_cell), Paragraph("25.48%", table_cell), Paragraph("76.10%", table_cell), Paragraph("71.49%", table_cell), Paragraph("7.74%", table_cell), Paragraph("6.09%", table_cell)],
        [Paragraph("Black", table_cell), Paragraph("662", table_cell), Paragraph("14.05%", table_cell), Paragraph("64.52%", table_cell), Paragraph("71.43%", table_cell), Paragraph("3.63%", table_cell), Paragraph("4.98%", table_cell)],
        [Paragraph("Asian-Pac-Islander", table_cell), Paragraph("200", table_cell), Paragraph("26.00%", table_cell), Paragraph("73.08%", table_cell), Paragraph("65.52%", table_cell), Paragraph("10.00%", table_cell), Paragraph("7.00%", table_cell)],
        [Paragraph("Amer-Indian-Eskimo", table_cell), Paragraph("73", table_cell), Paragraph("12.33%", table_cell), Paragraph("44.44%", table_cell), Paragraph("66.67%", table_cell), Paragraph("2.74%", table_cell), Paragraph("6.85%", table_cell)],
    ]
    t_sub = Table(sub_data, colWidths=[100, 65, 65, 75, 75, 65, 65])
    t_sub.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_sub)
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Fairness Summary:</b> Gender recall parity is balanced (75.66% Male vs 72.65% Female, 3.01% delta). Lower recall in indigenous cohorts reflects historical data sparsity (<i>N</i>=73), requiring post-processing threshold adjustments.", body_style))

    # Page 2 Break
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: 7. INTERPRETATION, 8. PRODUCTION READINESS, 9. LIMITATIONS
    # =========================================================================
    story.append(Paragraph("7. Model Interpretation & Behavioral Dynamics", h1_style))
    fi_img_path = HERE / "feature_importance.png"
    fi_data = [
        [
            Image(str(fi_img_path), width=230, height=145) if fi_img_path.is_file() else Paragraph("Feature Importance Plot", body_style),
            [
                Paragraph("<b>Permutation Feature Importance (&Delta; ROC-AUC)</b>", h2_style),
                Table([
                    [Paragraph("1. marital_status", table_cell_bold), Paragraph("<b>0.0779 &plusmn; 0.0021</b>", table_cell), Paragraph("Primary routing signal", table_cell)],
                    [Paragraph("2. capital_gain", table_cell_bold), Paragraph("<b>0.0593 &plusmn; 0.0030</b>", table_cell), Paragraph("Deterministic wealth trigger", table_cell)],
                    [Paragraph("3. age", table_cell_bold), Paragraph("<b>0.0434 &plusmn; 0.0038</b>", table_cell), Paragraph("Career seniority quadratic arc", table_cell)],
                    [Paragraph("4. education_num", table_cell_bold), Paragraph("<b>0.0298 &plusmn; 0.0021</b>", table_cell), Paragraph("Credential sheepskin effect", table_cell)],
                    [Paragraph("5. hours_per_week", table_cell_bold), Paragraph("0.0136 &plusmn; 0.0012", table_cell), Paragraph("Labor supply intensity", table_cell)],
                    [Paragraph("6. capital_loss", table_cell_bold), Paragraph("0.0133 &plusmn; 0.0008", table_cell), Paragraph("Active investor indicator", table_cell)],
                    [Paragraph("7. occupation", table_cell_bold), Paragraph("0.0128 &plusmn; 0.0010", table_cell), Paragraph("Managerial vs service split", table_cell)],
                    [Paragraph("11. fnlwgt (weight)", table_cell_bold), Paragraph("0.0008 &plusmn; 0.0002", table_cell), Paragraph("Census weight (Redundant)", table_cell)],
                    [Paragraph("14. education (raw)", table_cell_bold), Paragraph("-0.0002 &plusmn; 0.0001", table_cell), Paragraph("Redundant with education_num", table_cell)],
                ], colWidths=[90, 85, 110], style=[
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ('TOPPADDING', (0, 0), (-1, -1), 1.2),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
                ]),
                Spacer(1, 2),
                Paragraph("<b>Interpretation Insights:</b> Earning likelihood scales monotonically with education and hours worked, cresting between ages 45–53. Marital status represents 31.7% of importance, acting as an overwhelming dual-earner proxy and driving the 'Marriage Trap'.", body_style)
            ]
        ]
    ]
    t_fi_panel = Table(fi_data, colWidths=[235, 295])
    t_fi_panel.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 1),
        ('RIGHTPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(t_fi_panel)
    story.append(Spacer(1, 4))

    # 8. Production Readiness
    story.append(Paragraph("8. Production Readiness & Serving Specification", h1_style))
    p8_text = (
        "<b>8.1 Serialized Production Artifact:</b> The model is fully bundled inside <code>final_model.joblib</code> (1.62 MB). "
        "It encapsulates the entire inference pipeline: raw feature engineering, imputation, one-hot encoding, gradient boosting, and Platt scaling.<br/>"
        "<b>8.2 Clean Inference API (<code>inference.py</code>):</b> Clients execute <code>score_raw(df)</code> passing raw census DataFrames with zero manual preprocessing.<br/>"
        "<b>Sample Real Holdout Predictions Generated via <code>inference.py</code>:</b>"
    )
    story.append(Paragraph(p8_text, body_style))
    story.append(Spacer(1, 2))

    inf_headers = [
        Paragraph("Age", table_cell_header),
        Paragraph("Education", table_cell_header),
        Paragraph("Marital Status", table_cell_header),
        Paragraph("Occupation", table_cell_header),
        Paragraph("Hours", table_cell_header),
        Paragraph("P(&gt;50K)", table_cell_header),
        Paragraph("Prediction", table_cell_header),
        Paragraph("Actual", table_cell_header),
        Paragraph("Audit", table_cell_header),
    ]
    inf_rows = [
        inf_headers,
        [Paragraph("49", table_cell), Paragraph("Bachelors", table_cell), Paragraph("Married-civ-spouse", table_cell), Paragraph("Exec-managerial", table_cell), Paragraph("40", table_cell), Paragraph("0.8097", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("CORRECT", badge_style)],
        [Paragraph("70", table_cell), Paragraph("Bachelors", table_cell), Paragraph("Married-civ-spouse", table_cell), Paragraph("Handlers-cleaners", table_cell), Paragraph("40", table_cell), Paragraph("0.7904", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("CORRECT", badge_style)],
        [Paragraph("29", table_cell), Paragraph("Bachelors", table_cell), Paragraph("Married-civ-spouse", table_cell), Paragraph("Sales", table_cell), Paragraph("45", table_cell), Paragraph("0.4878", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("&gt;50K", table_cell), Paragraph("CORRECT", badge_style)],
        [Paragraph("50", table_cell), Paragraph("Bachelors", table_cell), Paragraph("Divorced", table_cell), Paragraph("Exec-managerial", table_cell), Paragraph("30", table_cell), Paragraph("0.1172", table_cell), Paragraph("&le;50K", table_cell), Paragraph("&le;50K", table_cell), Paragraph("CORRECT", badge_style)],
        [Paragraph("21", table_cell), Paragraph("HS-grad", table_cell), Paragraph("Never-married", table_cell), Paragraph("Farming-fishing", table_cell), Paragraph("24", table_cell), Paragraph("0.0010", table_cell), Paragraph("&le;50K", table_cell), Paragraph("&le;50K", table_cell), Paragraph("CORRECT", badge_style)],
    ]
    t_inf = Table(inf_rows, colWidths=[30, 65, 95, 95, 35, 55, 55, 55, 55])
    t_inf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_inf)
    story.append(Spacer(1, 4))

    # 9. Limitations & Future Improvements
    story.append(Paragraph("9. Limitations & Four-Dimensional Future Improvements", h1_style))
    p9_text = (
        "• <b>More Data:</b> Ingest modern Census Bureau Current Population Survey (CPS) records. The 1994 data lacks contemporary digital and gig-economy roles, and inflation has shifted the real-wage equivalent of $50K to ~$105K today.<br/>"
        "• <b>Better Features:</b> Integrate metropolitan Cost of Living Indices (COLI) to eliminate regional false positives, and engineer <code>occupation_x_hours</code> to credit overtime earnings in skilled trades.<br/>"
        "• <b>Different Models:</b> Benchmark cost-sensitive XGBoost and LightGBM with asymmetric loss functions (5:1 penalty on FN) directly integrated into gradient tree splits.<br/>"
        "• <b>Additional Validation:</b> Implement automated Population Stability Index (PSI) monitoring to detect data drift, and deploy cohort-specific decision thresholding to eliminate subgroup recall gaps."
    )
    story.append(Paragraph(p9_text, body_style))
    story.append(Spacer(1, 4))

    # Sign-off Table
    sign_table = Table([
        [Paragraph("<b>Lead Data Scientist:</b> Armish Iqbal", table_cell), Paragraph("<b>Artifact ID:</b> final_model.joblib (1.62 MB)", table_cell), Paragraph("<b>Status:</b> PRODUCTION READY & APPROVED", badge_style)]
    ], colWidths=[180, 180, 180], style=[
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ])
    story.append(sign_table)

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {pdf_path} ({pdf_path.stat().st_size / 1e3:.1f} KB)")


if __name__ == "__main__":
    build_pdf()
