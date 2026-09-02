import os
import sys
import pypdf
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

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
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (Top)
        self.drawString(32, 762, "DATA SCIENCE INTERNSHIP · DAY 3 EXECUTIVE SUMMARY REPORT")
        self.drawRightString(580, 762, "FEATURE ENGINEERING & STATISTICAL CV")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(32, 756, 580, 756)
        
        # Footer (Bottom)
        self.line(32, 28, 580, 28)
        self.drawString(32, 18, "Confidential · ML Benchmark Pipeline · Zero-Leakage Preprocessing")
        self.drawRightString(580, 18, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_path = "day3_summary_report.pdf"
    
    # Page setup: Letter is 612 x 792 pt.
    # Margins: left=32, right=32, top=40, bottom=34. Usable width = 548 pt.
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=40,
        bottomMargin=34
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#0F172A")    # Deep Slate
    accent_blue = colors.HexColor("#1E40AF")      # Navy
    accent_teal = colors.HexColor("#0284C7")      # Vibrant Cyan/Blue
    text_dark = colors.HexColor("#1E293B")        # Dark Slate
    muted_text = colors.HexColor("#475569")       # Muted Slate
    card_bg = colors.HexColor("#F8FAFC")          # Slate 50
    border_color = colors.HexColor("#E2E8F0")     # Slate 200
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=17,
        textColor=primary_color,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=muted_text,
        spaceAfter=5
    )
    
    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=11,
        textColor=accent_blue,
        spaceBefore=3,
        spaceAfter=2
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=text_dark
    )
    
    bold_body = ParagraphStyle(
        'BoldBodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=text_dark
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.2,
        textColor=text_dark
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8.2,
        textColor=primary_color
    )
    
    table_cell_code = ParagraphStyle(
        'TableCellCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.2,
        leading=7.8,
        textColor=colors.HexColor("#0F766E")
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.8,
        leading=8.5,
        textColor=colors.white
    )

    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8.0,
        textColor=colors.HexColor("#15803D")
    )

    story = []
    
    # -------------------------------------------------------------
    # PAGE 1: TASK 1, TASK 2, TASK 3
    # -------------------------------------------------------------
    
    # Title & Metadata Banner
    header_table_data = [
        [
            Paragraph("<b>DAY 3: FEATURE ENGINEERING & STATISTICAL MODEL EVALUATION</b>", title_style),
            Paragraph("<b>STATUS:</b> Fully Executed<br/><b>DATASET:</b> Adult Census (N=32,561)<br/><b>TARGET:</b> Income >50K (24.08%)", ParagraphStyle('MetaRight', fontName='Helvetica', fontSize=6.5, leading=8.5, textColor=muted_text, alignment=2))
        ]
    ]
    t_head = Table(header_table_data, colWidths=[380, 168])
    t_head.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_head)
    story.append(Paragraph("A principled approach to expanding feature representation without data leakage, cross-validating competing algorithms, and benchmarking dimensionality trade-offs.", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=border_color, spaceBefore=0, spaceAfter=4))
    
    # --- TASK 1: ENGINEERED FEATURE DICTIONARY ---
    story.append(Paragraph("1. Task 1: Engineered Features Dictionary & Univariate Signals", sec_heading))
    story.append(Paragraph("Eight domain-principled features were constructed strictly at row-level (zero target leakage). Mutual Information (MI) and grouped target rates confirm substantial non-linear predictive signals:", body_style))
    story.append(Spacer(1, 2))
    
    feat_dict_data = [
        [
            Paragraph("Feature Name", table_header),
            Paragraph("Type", table_header),
            Paragraph("Creation Rule (Zero-Leakage)", table_header),
            Paragraph("Domain Justification / Signal Logic", table_header),
            Paragraph("MI", table_header),
            Paragraph("Univariate Signal (P(>50K))", table_header)
        ],
        [
            Paragraph("<b>is_married</b>", table_cell_bold),
            Paragraph("Binary Flag", table_cell),
            Paragraph("marital_status in {Civ-Spouse, AF-Spouse}", table_cell_code),
            Paragraph("Household stability & dual-income earner proxy; strongest adult census predictor.", table_cell),
            Paragraph("<b>0.1105</b>", table_cell_bold),
            Paragraph("Unmarried: <b>6.5%</b><br/>Married: <b>44.7%</b> (6.9x lift)", table_cell)
        ],
        [
            Paragraph("<b>edu_x_hours</b>", table_cell_bold),
            Paragraph("Interaction", table_cell),
            Paragraph("education_num * hours_per_week", table_cell_code),
            Paragraph("Captures compounding marginal wage returns per hour for educated workers.", table_cell),
            Paragraph("<b>0.0840</b>", table_cell_bold),
            Paragraph("Q1: <b>6.2%</b> | Q2: <b>16.4%</b><br/>Q3: <b>30.0%</b> | Q4: <b>52.2%</b>", table_cell)
        ],
        [
            Paragraph("<b>log_capital_gain</b>", table_cell_bold),
            Paragraph("Log Numeric", table_cell),
            Paragraph("np.log1p(capital_gain)", table_cell_code),
            Paragraph("Compresses severe right-skewed tail, linearizing massive positive returns.", table_cell),
            Paragraph("<b>0.0824</b>", table_cell_bold),
            Paragraph("Zero: <b>20.7%</b> | Mid: <b>89.9%</b><br/>High: <b>98.8%</b>", table_cell)
        ],
        [
            Paragraph("<b>age_bucket</b>", table_cell_bold),
            Paragraph("Categorical", table_cell),
            Paragraph("pd.cut(age, [0, 24, 34, 44, 54, 64, 200])", table_cell_code),
            Paragraph("Models non-linear human earning curve peaking in mid-to-late career stages.", table_cell),
            Paragraph("<b>0.0631</b>", table_cell_bold),
            Paragraph("&lt;25: <b>1.1%</b> | 25-34: <b>16.8%</b><br/>45-54: <b>40.1%</b> | 65+: <b>20.7%</b>", table_cell)
        ],
        [
            Paragraph("<b>is_higher_ed</b>", table_cell_bold),
            Paragraph("Binary Flag", table_cell),
            Paragraph("education in {Bachelors, Masters, Prof, Doc}", table_cell_code),
            Paragraph("Isolates step-function wage credential premium over standard secondary schooling.", table_cell),
            Paragraph("<b>0.0494</b>", table_cell_bold),
            Paragraph("Non-Higher: <b>16.1%</b><br/>Higher Ed: <b>48.5%</b> (3.0x lift)", table_cell)
        ],
        [
            Paragraph("<b>hours_bin</b>", table_cell_bold),
            Paragraph("Categorical", table_cell),
            Paragraph("pd.cut(hours, [-1, 20, 35, 40, 50, 200])", table_cell_code),
            Paragraph("Distinguishes non-linear part-time, standard 40h, and extreme overtime pay tiers.", table_cell),
            Paragraph("<b>0.0385</b>", table_cell_bold),
            Paragraph("Part-time: <b>6.7%</b> | Overtime: <b>39.6%</b><br/>Extreme (&gt;50h): <b>41.3%</b>", table_cell)
        ],
        [
            Paragraph("<b>has_capital_gain</b>", table_cell_bold),
            Paragraph("Binary Flag", table_cell),
            Paragraph("(capital_gain > 0).astype(int)", table_cell_code),
            Paragraph("Separates asset holders from non-investors given 92% structural sparsity.", table_cell),
            Paragraph("<b>0.0323</b>", table_cell_bold),
            Paragraph("No Gain: <b>20.7%</b><br/>Has Gain: <b>61.8%</b> (3.0x lift)", table_cell)
        ],
        [
            Paragraph("<b>has_capital_loss</b>", table_cell_bold),
            Paragraph("Binary Flag", table_cell),
            Paragraph("(capital_loss > 0).astype(int)", table_cell_code),
            Paragraph("Indicator of active portfolio management & liquid wealth exposure.", table_cell),
            Paragraph("<b>0.0098</b>", table_cell_bold),
            Paragraph("No Loss: <b>22.8%</b><br/>Has Loss: <b>50.9%</b> (2.2x lift)", table_cell)
        ],
    ]
    
    t_feat = Table(feat_dict_data, colWidths=[68, 52, 120, 148, 30, 130])
    t_feat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), accent_blue),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (4,0), (4,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, card_bg]),
        ('GRID', (0,0), (-1,-1), 0.4, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_feat)
    story.append(Spacer(1, 4))
    
    # --- TASK 2: PIPELINE REBUILD & ZERO-LEAKAGE ARCHITECTURE ---
    story.append(Paragraph("2. Task 2: Rebuilding the Preprocessing Pipeline (Strict Zero-Leakage)", sec_heading))
    p2_text = (
        "<b>Architecture:</b> Engineered features are generated inside a clean scikit-learn transformer (<code>FeatureEngineer</code> / <code>FunctionTransformer</code>) "
        "placed upstream of a <code>ColumnTransformer</code>. Numerical features (including <code>edu_x_hours</code> and <code>log_capital_gain</code>) receive <code>SimpleImputer(strategy='median')</code> "
        "and <code>StandardScaler</code>. Categorical features (including <code>age_bucket</code> and <code>hours_bin</code>) receive <code>SimpleImputer(strategy='most_frequent')</code> "
        "and <code>OneHotEncoder(handle_unknown='ignore', sparse_output=False)</code>. All estimators are fitted strictly on training folds inside cross-validation, guaranteeing zero test leakage."
    )
    story.append(Paragraph(p2_text, body_style))
    story.append(Spacer(1, 4))
    
    # --- TASK 3: CROSS-VALIDATED MODEL COMPARISON ---
    story.append(Paragraph("3. Task 3: 5-Fold Stratified Cross-Validation Benchmark", sec_heading))
    story.append(Paragraph("Three diverse classifiers were evaluated under identical 5-fold stratified CV splits (80/20 train/validation ratio per fold). HistGradientBoosting clearly outperforms competing architectures across all evaluation dimensions:", body_style))
    story.append(Spacer(1, 2))
    
    cv_data = [
        [
            Paragraph("Model Architecture", table_header),
            Paragraph("Accuracy (Mean ± Std)", table_header),
            Paragraph("ROC-AUC (Mean ± Std)", table_header),
            Paragraph("F1-Score (Mean ± Std)", table_header),
            Paragraph("Primary Metric Assessment & Behavior", table_header)
        ],
        [
            Paragraph("<b>HistGradientBoosting</b>", table_cell_bold),
            Paragraph("<b>0.8723 ± 0.0032</b>", table_cell_bold),
            Paragraph("<b>0.9272 ± 0.0016</b>", table_cell_bold),
            Paragraph("<b>0.7120 ± 0.0070</b>", table_cell_bold),
            Paragraph("<font color='#15803D'><b>Top Performer.</b></font> Lowest variance across folds; handles mixed non-linear interactions natively.", table_cell)
        ],
        [
            Paragraph("<b>Random Forest (100 trees)</b>", table_cell),
            Paragraph("0.8557 ± 0.0041", table_cell),
            Paragraph("0.9042 ± 0.0032", table_cell),
            Paragraph("0.6775 ± 0.0090", table_cell),
            Paragraph("Strong recall but lower ROC-AUC precision; higher fold-to-fold variance (±0.0032).", table_cell)
        ],
        [
            Paragraph("<b>Logistic Regression (L2)</b>", table_cell),
            Paragraph("0.8512 ± 0.0028", table_cell),
            Paragraph("0.9055 ± 0.0022", table_cell),
            Paragraph("0.6589 ± 0.0073", table_cell),
            Paragraph("Fast and stable baseline; limited by linear hyperplanes on complex non-linear combinations.", table_cell)
        ]
    ]
    t_cv = Table(cv_data, colWidths=[110, 95, 95, 95, 153])
    t_cv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#F0FDF4"), colors.white, card_bg]),
        ('GRID', (0,0), (-1,-1), 0.4, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_cv)
    story.append(Spacer(1, 3))
    
    # Embedded 5-fold CV Plot
    plot1_path = "task3_5fold_boxplots.png"
    if os.path.exists(plot1_path):
        # Original is 5361 x 1777 (aspect ratio ~ 3.017)
        # Width: 548 pt, Height: 548 / 3.017 = 181 pt. Let's scale to width=548, height=138 pt.
        img1 = Image(plot1_path, width=548, height=132)
        story.append(img1)
        story.append(Paragraph("<b>Figure 1:</b> Stratified 5-Fold CV Score Distributions (Accuracy, ROC-AUC, F1). HistGradientBoosting demonstrates clear dominance with zero fold overlap in ROC-AUC.", ParagraphStyle('FigCap', fontName='Helvetica-Oblique', fontSize=6.2, leading=7.5, textColor=muted_text, alignment=1)))
    
    # -------------------------------------------------------------
    # PAGE BREAK TO PAGE 2
    # -------------------------------------------------------------
    story.append(PageBreak())
    
    # -------------------------------------------------------------
    # PAGE 2: TASK 4, TASK 5, AND FINAL TUNING ROADMAP
    # -------------------------------------------------------------
    story.append(Paragraph("4. Task 4: Statistical Hypothesis Testing & Feature Importance", sec_heading))
    story.append(Paragraph("To confirm whether HistGradientBoosting's superiority is statistically believable and practically meaningful, paired tests were executed across identical 10-fold CV splits against Logistic Regression:", body_style))
    story.append(Spacer(1, 2))
    
    # Two-column layout for Task 4: Left = Stats Table & Plot; Right = Feature Importance Analysis
    stat_table_data = [
        [Paragraph("Hypothesis Test Metric", table_header), Paragraph("Value / Result", table_header), Paragraph("Statistical & Practical Interpretation", table_header)],
        [
            Paragraph("<b>HGB vs LR Mean ROC-AUC</b>", table_cell_bold),
            Paragraph("<b>0.9280 vs 0.9143</b>", table_cell_bold),
            Paragraph("Mean ROC-AUC Lift: <b>+0.0138 (+1.38%)</b>. Consistent lift across every partition.", table_cell)
        ],
        [
            Paragraph("<b>Paired t-Test (Parametric)</b>", table_cell_bold),
            Paragraph("<b>t = 33.45, p = 9.41e-11</b>", table_cell_bold),
            Paragraph("p &lt;&lt; 0.001. Null hypothesis (equal means) definitively rejected.", table_cell)
        ],
        [
            Paragraph("<b>Wilcoxon Signed-Rank</b>", table_cell_bold),
            Paragraph("<b>W = 0.0, p = 0.00195</b>", table_cell_bold),
            Paragraph("Non-parametric test confirms zero rank inversions: <b>10 Wins, 0 Losses, 0 Ties</b>.", table_cell)
        ],
        [
            Paragraph("<b>Practical Significance</b>", table_cell_bold),
            Paragraph("<font color='#15803D'><b>Meaningful Lift</b></font>", badge_style),
            Paragraph("In income classification, a +1.38% AUC and +5.3% F1 lift translates to hundreds of additional high-value earners correctly classified without false-positive inflation.", table_cell)
        ]
    ]
    t_stat = Table(stat_table_data, colWidths=[120, 115, 313])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), accent_blue),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, card_bg]),
        ('GRID', (0,0), (-1,-1), 0.4, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 3))
    
    # Two-column box: Left = Plot 2 (Boxplot), Right = Feature Importance Table / Text
    plot2_path = "task4_10fold_roc_auc_boxplot.png"
    
    # Feature importance table data
    imp_summary_data = [
        [Paragraph("Engineered Feature", table_header), Paragraph("HGB Perm. Imp.", table_header), Paragraph("LR Coeff (Odds Ratio)", table_header), Paragraph("Domain & Mechanical Role", table_header)],
        [
            Paragraph("<b>edu_x_hours</b>", table_cell_bold),
            Paragraph("<b>0.0237</b> (Top Eng)", table_cell_bold),
            Paragraph("-0.283 (OR: 0.75)*", table_cell),
            Paragraph("Highest HGB engineered signal; captures compounding hourly education returns.", table_cell)
        ],
        [
            Paragraph("<b>log_capital_gain</b>", table_cell_bold),
            Paragraph("0.0000 (Subsumed)", table_cell),
            Paragraph("<b>+5.170 (OR: 176.0)</b>", table_cell_bold),
            Paragraph("Dominant linear driver for LR; massive positive odds for capital market investors.", table_cell)
        ],
        [
            Paragraph("<b>has_capital_gain</b>", table_cell_bold),
            Paragraph("0.0000 (Subsumed)", table_cell),
            Paragraph("<b>-5.003 (OR: 0.007)</b>", table_cell_bold),
            Paragraph("Acts as negative intercept calibration for non-investors in logistic equation.", table_cell)
        ],
        [
            Paragraph("<b>age_bucket</b>", table_cell_bold),
            Paragraph("0.0012", table_cell),
            Paragraph("&lt;25: -1.05 | 45-54: +0.55", table_cell),
            Paragraph("Reflects peak earning years (45-54) vs early entry-level suppression (&lt;25).", table_cell)
        ],
        [
            Paragraph("<b>hours_bin</b>", table_cell_bold),
            Paragraph("0.0003", table_cell),
            Paragraph("Reduced: -1.00 | Extr: +0.61", table_cell),
            Paragraph("Strong threshold demarcation between part-time and excessive overtime work.", table_cell)
        ],
        [
            Paragraph("<b>is_married</b>", table_cell_bold),
            Paragraph("0.0001", table_cell),
            Paragraph("+0.437 (OR: 1.55)", table_cell),
            Paragraph("HGB splits directly on raw <code>marital_status</code> (imp: 0.0910); LR relies on binary.", table_cell)
        ]
    ]
    t_imp = Table(imp_summary_data, colWidths=[85, 78, 95, 120])
    t_imp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, card_bg]),
        ('GRID', (0,0), (-1,-1), 0.4, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    
    if os.path.exists(plot2_path):
        # 2361 x 1752 (aspect ratio ~ 1.348)
        img2 = Image(plot2_path, width=160, height=115)
        
        two_col_data = [
            [img2, t_imp]
        ]
        t_twocol = Table(two_col_data, colWidths=[165, 383])
        t_twocol.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t_twocol)
        story.append(Paragraph("<b>Figure 2:</b> 10-Fold Paired ROC-AUC Boxplot (Left). Feature Importance Comparison Table (Right). *Note: LR coefficient for edu_x_hours reflects collinearity adjustment in the presence of base education_num and hours_per_week.", ParagraphStyle('FigCap2', fontName='Helvetica-Oblique', fontSize=6.0, leading=7.2, textColor=muted_text)))
    else:
        story.append(t_imp)
        
    story.append(Spacer(1, 4))
    
    # --- TASK 5: FEATURE SELECTION & COMPUTATIONAL BENCHMARK ---
    story.append(Paragraph("5. Task 5: Feature Selection Benchmark & Dimensionality Check", sec_heading))
    story.append(Paragraph("L1-regularized feature selection (SelectFromModel with LogisticRegression penalty='l1') was benchmarked against the full feature representation across 5-fold CV on HistGradientBoosting:", body_style))
    story.append(Spacer(1, 2))
    
    fs_data = [
        [
            Paragraph("Feature Set Configuration", table_header),
            Paragraph("Orig / Transformed", table_header),
            Paragraph("Retained Features", table_header),
            Paragraph("Accuracy (Mean ± Std)", table_header),
            Paragraph("ROC-AUC (Mean ± Std)", table_header),
            Paragraph("F1-Score (Mean ± Std)", table_header),
            Paragraph("CV Wall Time", table_header)
        ],
        [
            Paragraph("<b>Full Feature Set (All)</b>", table_cell_bold),
            Paragraph("14 / 123", table_cell),
            Paragraph("<b>123 (100%)</b>", table_cell_bold),
            Paragraph("<b>0.8742 ± 0.0036</b>", table_cell_bold),
            Paragraph("<b>0.9283 ± 0.0016</b>", table_cell_bold),
            Paragraph("<b>0.7154 ± 0.0080</b>", table_cell_bold),
            Paragraph("<b>7.92s</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>Moderate L1 Selection (C=0.1)</b>", table_cell),
            Paragraph("14 / 123", table_cell),
            Paragraph("49 (39.8%)", table_cell),
            Paragraph("0.8739 ± 0.0037", table_cell),
            Paragraph("0.9283 ± 0.0015", table_cell),
            Paragraph("0.7143 ± 0.0084", table_cell),
            Paragraph("8.84s (+0.92s)", table_cell)
        ],
        [
            Paragraph("<b>Aggressive L1 Selection (C=0.01)</b>", table_cell),
            Paragraph("14 / 123", table_cell),
            Paragraph("18 (14.6%)", table_cell),
            Paragraph("0.8686 ± 0.0022", table_cell),
            Paragraph("0.9249 ± 0.0023", table_cell),
            Paragraph("0.6983 ± 0.0061", table_cell),
            Paragraph("2.88s (-5.04s)", table_cell)
        ]
    ]
    t_fs = Table(fs_data, colWidths=[125, 65, 65, 85, 85, 80, 43])
    t_fs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#EFF6FF"), colors.white, card_bg]),
        ('GRID', (0,0), (-1,-1), 0.4, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_fs)
    story.append(Spacer(1, 4))
    
    # --- EXECUTIVE DECISION & TUNING ROADMAP ---
    story.append(Paragraph("6. Architectural Decision & Hyperparameter Tuning Roadmap (Day 4 Plan)", sec_heading))
    
    roadmap_box_data = [
        [
            Paragraph(
                "<b>FINAL ARCHITECTURAL RECOMMENDATION:</b><br/>"
                "• <b>Selected Algorithm:</b> <code>HistGradientBoostingClassifier</code>. Highly statistically superior to Logistic Regression (p=9.4e-11, 10/10 fold wins) and Random Forest (+2.3% ROC-AUC). Offers native binning, fast histogram computation, and automatic non-linear split handling.<br/>"
                "• <b>Selected Feature Representation:</b> <code>Full Feature Set (All 123 features)</code>. Preserves peak ROC-AUC (0.9283) and F1 (0.7154). Moderate L1 selection eliminated 74 features with equal ROC-AUC but <i>increased</i> total runtime (8.84s vs 7.92s) due to the selector fitting step. Aggressive L1 sacrificed 0.34% ROC-AUC and 1.71% F1 score. Because tree boosting inherently performs greedy coordinate feature selection at split points, the full representation maximizes interaction discovery with zero speed penalty.",
                ParagraphStyle('DecisionBox', fontName='Helvetica', fontSize=6.8, leading=8.6, textColor=colors.HexColor("#0F172A"))
            ),
            Paragraph(
                "<b>DAY 4 HYPERPARAMETER SEARCH GRID:</b><br/>"
                "• <code>learning_rate</code>: [0.03, 0.05, 0.10, 0.15]<br/>"
                "• <code>max_leaf_nodes</code>: [15, 31, 63, 127]<br/>"
                "• <code>min_samples_leaf</code>: [20, 50, 100]<br/>"
                "• <code>l2_regularization</code>: [0.0, 0.5, 1.0, 5.0]<br/>"
                "• <code>max_iter</code>: Early stopping with tolerance=1e-4<br/>"
                "<b>Objective:</b> Maximize cross-validated ROC-AUC while fine-tuning threshold calibration for F1 optimization.",
                ParagraphStyle('GridBox', fontName='Helvetica', fontSize=6.8, leading=8.6, textColor=colors.HexColor("#1E3A8A"))
            )
        ]
    ]
    t_roadmap = Table(roadmap_box_data, colWidths=[330, 218])
    t_roadmap.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F1F5F9")),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (0,0), 0.6, colors.HexColor("#CBD5E1")),
        ('BOX', (1,0), (1,0), 0.6, colors.HexColor("#93C5FD")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_roadmap)
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Check page count with pypdf
    reader = pypdf.PdfReader(pdf_path)
    page_count = len(reader.pages)
    print(f"Generated PDF: {pdf_path} with {page_count} pages.")
    return page_count

if __name__ == "__main__":
    count = build_pdf()
    if count == 2:
        print("SUCCESS: PDF is exactly 2 pages!")
    else:
        print(f"WARNING: PDF has {count} pages instead of 2. Adjust spacing!")
