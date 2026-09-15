from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.pdfgen import canvas
import pypdf

HERE = Path(__file__).resolve().parent
NAVY = colors.HexColor('#0F172A')
MUTED = colors.HexColor('#334155')
LINE = colors.HexColor('#CBD5E1')
HEAD = colors.HexColor('#1E3A5F')
CARD = colors.HexColor('#F8FAFC')
ACCENT = colors.HexColor('#0284C7')


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
            self.setFont('Helvetica-Bold', 7.5)
            self.setFillColor(HEAD)
            self.drawString(28, 764, 'AFL DATA FOUNDATIONS · WEEK 3 DAY 1')
            self.setFont('Helvetica', 7.5)
            self.setFillColor(MUTED)
            self.drawRightString(584, 764, 'DATA DICTIONARY & PREDICTION CONTRACT')
            self.setStrokeColor(LINE)
            self.line(28, 758, 584, 758)
            self.line(28, 24, 584, 24)
            self.drawString(28, 14, 'Zero-Leakage Rolling Features · Strict Temporal Split · Target Contract for Day 2')
            self.drawRightString(584, 14, f'Page {self._pageNumber} of {n}')
            self.restoreState()
            super().showPage()
        super().save()


def P(name, size, leading, color, bold=False, after=2, before=0):
    styles = getSampleStyleSheet()
    return ParagraphStyle(
        name,
        parent=styles['Normal'],
        fontName='Helvetica-Bold' if bold else 'Helvetica',
        fontSize=size,
        leading=leading,
        textColor=color,
        spaceAfter=after,
        spaceBefore=before,
    )


styles = getSampleStyleSheet()
th_s = ParagraphStyle('th', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=6.2, leading=7.5, textColor=colors.white)
td_s = ParagraphStyle('td', parent=styles['Normal'], fontName='Helvetica', fontSize=5.8, leading=7.0, textColor=NAVY)
td_b = ParagraphStyle('tdb', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=5.8, leading=7.0, textColor=HEAD)
td_code = ParagraphStyle('tdc', parent=styles['Normal'], fontName='Courier-Bold', fontSize=5.6, leading=6.8, textColor=HEAD)


def make_cell(val, is_header=False, is_code=False):
    if is_header:
        return Paragraph(str(val), th_s)
    elif is_code:
        return Paragraph(str(val), td_code)
    else:
        return Paragraph(str(val), td_s)


def build_pdf():
    path = str(HERE / 'data_dictionary_and_targets.pdf')
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=28, rightMargin=28, topMargin=34, bottomMargin=26)
    title = P('t', 11.5, 13.5, NAVY, True, 2)
    h = P('h', 8.2, 10.0, HEAD, True, 2, 3)
    body = P('b', 6.8, 8.5, MUTED, False, 2)
    small = P('s', 6.2, 7.8, MUTED, False, 1)

    story = []
    story.append(Paragraph('AFL Data Foundations — Data Dictionary & Target Specification Contract', title))
    story.append(Paragraph(
        '<b>Data Inventory:</b> 43 AFL seasons (1983–2025), 20 clubs, 3,109 players, and 7,904 unique matches. '
        '<b>Data Quality Audit:</b> Stripped whitespace/tab prefixes; normalized team names (\'W. Bulldogs\' -&gt; \'western bulldogs\'); '
        'imputed corrupt negative disposals as K+HB; resolved sparse zero-encoding for goals/behinds; and purged duplicate biographical entries.',
        body,
    ))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LINE, spaceAfter=3))

    # Relational Tables
    story.append(Paragraph('1. Relational Schema & Table Joining Contract', h))
    t1_raw = [
        ['Table Name', 'Grain', 'Rows', 'Primary Keys', 'Join Path'],
        ['team_matches_home_away_raw', 'Team-Match (2/game)', '15,808', 'id / [date, team]', 'Pairs H to A on [date, year, round, clean(team)==clean(opp)]'],
        ['afl_players_round_by_round_stats_raw', 'Player-Match (Box score)', '274,089', 'id / [player_id, date]', 'Joins to matches on [date, year, round, team, opp]; to info on player_id'],
        ['afl_players_seasonal_stats_raw', 'Player-Season-Phase', '25,491', '[player_id, year, team]', 'Joins to info on clean numeric player_id (regex stripped \'ID_\')'],
        ['afl_players_info_raw', 'Player Biographical', '2,843', 'id', 'Master biographical dimension: name, debut, DOB, height, weight'],
    ]
    t1_data = []
    for r_idx, row in enumerate(t1_raw):
        t1_data.append([make_cell(c, is_header=(r_idx == 0), is_code=(r_idx > 0 and c_idx == 0)) for c_idx, c in enumerate(row)])
    t1 = Table(t1_data, colWidths=[120, 85, 38, 95, 218])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEAD),
        ('BACKGROUND', (0, 1), (-1, -1), CARD),
        ('GRID', (0, 0), (-1, -1), 0.3, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t1)

    # Targets
    story.append(Paragraph('2. Prediction Targets Specification (Day 2 Models Contract)', h))
    t2_raw = [
        ['Target Name', 'Level', 'Type', 'Mathematical Definition / Formula', 'Justification / Downstream Role'],
        ['home_margin (Primary)', 'Match', 'Regression', 'Score(Home) - Score(Away)', 'Retains victory magnitude; services spread/line betting; maps to prob via CDF.'],
        ['home_team_win', 'Match', 'Binary', '1 if Margin &gt; 0, 0 if Margin &lt; 0 (draws = 0.5)', 'Direct match-winner classification target (65 draws = 0.82% excluded/neutral).'],
        ['top_disposal_getter', 'Player', 'Binary', '1 if Disposals_i == max_match(Disposals)', 'Flags leading ball-winner across both squads in match; continuous = disposals.'],
        ['top_goal_kicker', 'Player', 'Binary', '1 if Goals_i == max_match(Goals) and Goals &gt; 0', 'Flags leading goal-scorer in match; continuous = goals kicked.'],
        ['fantasy_points', 'Player', 'Composite', '3K + 2HB + 3M + 4T + 1HO + 6G + 1B + 1FF - 3FA', 'AFL Fantasy official scoring system (unbiased cross-position production metric).'],
        ['player_impact_score', 'Player', 'Composite', 'Disp + 2*CP + 3*Clear + 4*I50 + 6G + 4T + 2*GA - 3*Clang', 'Brownlow/Champion Data composite measuring field-impact &amp; territory gain.'],
    ]
    t2_data = []
    for r_idx, row in enumerate(t2_raw):
        t2_data.append([make_cell(c, is_header=(r_idx == 0), is_code=(r_idx > 0 and c_idx in [0, 3])) for c_idx, c in enumerate(row)])
    t2 = Table(t2_data, colWidths=[90, 32, 45, 185, 204])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEAD),
        ('BACKGROUND', (0, 1), (-1, -1), CARD),
        ('GRID', (0, 0), (-1, -1), 0.3, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t2)

    # Versioned Features
    story.append(Paragraph('3. Versioned Feature Dictionary (Strict Zero-Leakage Pipeline)', h))
    t3_raw = [
        ['Feature Name', 'Window', 'Description', 'Predictive Signal & Findings'],
        ['home/away_roll_win_5', '5-game (shift 1)', 'Rolling 5-match win rate entering game', 'Form differential has r = 0.35 with margin; top quintile wins 80.2% vs bottom 40.6%.'],
        ['roll_margin_5_diff', '5-game (shift 1)', 'Net scoring margin differential (Home - Away)', 'Correlation r = 0.41 with final match margin; strong continuous signal.'],
        ['home/away_win_streak', 'Expanding', 'Consecutive wins entering match (0 if loss/draw)', 'Measures team momentum and psychological winning form.'],
        ['rest_diff', 'Pre-match lag', 'Days rest diff: home_days_rest - away_days_rest', '+1 day rest advantage elevates home win rate from 58.1% to 60.6% (+1.8 pts margin).'],
        ['is_interstate_match', 'Contextual', '1 if visiting team traveled interstate, 0 for derby', 'Home win rate jumps from 55.8% (local derbies) to 63.1% against interstate travelers.'],
        ['ladder_rank_diff', 'Round-entry', 'Pre-match ladder standing advantage (Away - Home)', 'Correlation r = 0.41 with margin after Round 4; 2.4 points margin per rank advantage.'],
        ['player_roll5_disposals', '5-game (shift 1)', 'Player rolling 5-match average disposals', 'Correlation r = 0.67 to actual match disposals; primary predictor for top disposals.'],
        ['pos_archetype', 'Profile', 'Midfielder, Forward, Defender, Ruck, Utility', 'Midfielders avg 20.4 disp / 0.54 goals; Forwards avg 11.7 disp / 1.35 goals.'],
    ]
    t3_data = []
    for r_idx, row in enumerate(t3_raw):
        t3_data.append([make_cell(c, is_header=(r_idx == 0), is_code=(r_idx > 0 and c_idx == 0)) for c_idx, c in enumerate(row)])
    t3 = Table(t3_data, colWidths=[105, 68, 175, 208])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEAD),
        ('BACKGROUND', (0, 1), (-1, -1), CARD),
        ('GRID', (0, 0), (-1, -1), 0.3, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t3)

    # Split & Ceiling
    story.append(Paragraph('4. Reproducible Time-Based Split & Realistic Prediction Ceiling', h))
    story.append(Paragraph(
        '<b>Reproducible Split Protocol (src/features.py):</b> <font face="Courier">get_time_based_split(df, split_year=2024, test_year=2025)</font><br/>'
        '• <b>Train Set:</b> Seasons &lt; 2024 (7,472 match rows, 254,216 player-game rows).<br/>'
        '• <b>Validation Set:</b> Season 2024 (216 match rows, 9,937 player-game rows).<br/>'
        '• <b>Hold-out Test Set:</b> Season 2025 (216 match rows, 9,936 player-game rows).<br/>'
        '<b>Temporal Leakage Rationale:</b> In sports time-series, random cross-validation leaks future tactical adjustments, player breakout trajectories, and late-season momentum into earlier-round predictions. Real-world evaluation must strictly mirror production: training on historical seasons and evaluating forward in time.<br/>'
        '<b>Realistic Prediction Ceiling:</b> AFL is an inherently high-variance contact sport played with an elliptical ball across varying oval dimensions, vulnerable to in-game injuries, microclimatic weather shifts, and umpire adjudications. Top-tier industry models (Squiggle consensus, Matter of Stats, Footy Forecaster) achieve a realistic ceiling of <b>67% to 72% match accuracy</b> (log loss ~0.58–0.62). Any model reporting &gt;80% or \'near-perfect\' accuracy is an undeniable red flag indicating target leakage (e.g., in-match count stats like inside-50s) or forward lookahead bias.',
        body,
    ))

    story.append(Spacer(1, 2))
    story.append(Paragraph('<b>Artifact Version:</b> afl_match_features_v1.parquet (0.66 MB) · afl_player_match_features_v1.parquet (5.44 MB) · Git Branch: main', small))

    doc.build(story, canvasmaker=Footer)
    print('Wrote PDF:', path)


if __name__ == '__main__':
    build_pdf()
