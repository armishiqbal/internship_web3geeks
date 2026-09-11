"""Generate an enterprise-grade workflow architecture diagram for CrewAI Day 4 using Pillow."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
IMG_PATH = HERE / "workflow_architecture.png"


def create_diagram():
    # 2x Retina resolution
    W, H = 1920, 1160
    img = Image.new("RGBA", (W, H), "#F8FAFC")
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        f_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 36)
        f_subtitle = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 18)
        f_sec = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 22)
        f_node_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 18)
        f_node_role = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
        f_node_tool = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 13)
        f_edge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 13)
        f_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 12)
        f_legend = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    except Exception:
        f_title = f_sec = f_node_title = ImageFont.load_default()
        f_subtitle = f_node_role = f_node_tool = f_edge = f_badge = f_legend = ImageFont.load_default()

    # Header
    draw.text((60, 40), "Week 2 Day 4 — CrewAI Multi-Agent Workflow Architecture", fill="#0F172A", font=f_title)
    draw.text((60, 88), "Dual Execution Topologies: Deterministic Sequential Pipeline vs. Dynamic Hierarchical Delegation", fill="#475569", font=f_subtitle)

    # Helper: draw rounded card with border
    def draw_card(box, fill_color, border_color, radius=12, border_width=2):
        draw.rounded_rectangle(box, radius=radius, fill=fill_color, outline=border_color, width=border_width)

    # Helper: draw badge
    def draw_badge(x, y, text, bg_color, text_color="#FFFFFF"):
        bbox = draw.textbbox((x, y), text, font=f_badge)
        pad_x, pad_y = 10, 4
        badge_box = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
        draw.rounded_rectangle(badge_box, radius=6, fill=bg_color)
        draw.text((x, y), text, fill=text_color, font=f_badge)

    # Helper: draw horizontal arrow
    def draw_h_arrow(x1, y1, x2, y2, color="#64748B", text=None, text_above=True, width=3):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        # Arrowhead
        arr_size = 8
        draw.polygon([(x2, y2), (x2 - arr_size * 1.5, y2 - arr_size), (x2 - arr_size * 1.5, y2 + arr_size)], fill=color)
        if text:
            t_box = draw.textbbox((0, 0), text, font=f_edge)
            tw = t_box[2] - t_box[0]
            th = t_box[3] - t_box[1]
            tx = (x1 + x2) / 2 - tw / 2
            ty = y1 - th - 8 if text_above else y1 + 8
            # draw small backdrop
            draw.rectangle((tx - 6, ty - 2, tx + tw + 6, ty + th + 2), fill="#F8FAFC")
            draw.text((tx, ty), text, fill=color, font=f_edge)

    # Helper: draw vertical arrow
    def draw_v_arrow(x, y1, y2, color="#64748B", width=3, text=None, text_right=True):
        draw.line([(x, y1), (x, y2)], fill=color, width=width)
        arr_size = 8
        if y2 > y1:
            draw.polygon([(x, y2), (x - arr_size, y2 - arr_size * 1.5), (x + arr_size, y2 - arr_size * 1.5)], fill=color)
        else:
            draw.polygon([(x, y2), (x - arr_size, y2 + arr_size * 1.5), (x + arr_size, y2 + arr_size * 1.5)], fill=color)
        if text:
            tx = x + 12 if text_right else x - 120
            ty = (y1 + y2) / 2 - 8
            draw.text((tx, ty), text, fill=color, font=f_edge)

    # =========================================================================
    # SECTION 1: TOP PANEL - SEQUENTIAL PIPELINE
    # =========================================================================
    seq_panel = (60, 135, 1860, 520)
    draw_card(seq_panel, "#FFFFFF", "#CBD5E1", radius=16, border_width=2)
    draw.text((85, 155), "1. SEQUENTIAL PIPELINE (Process.sequential) — Deterministic DAG Handoffs", fill="#1E293B", font=f_sec)
    draw_badge(850, 160, "PREDICTABLE LATENCY • 100% DETERMINISTIC • 3,850 TOKENS • $0.00050", "#2563EB")

    # Sequence Nodes:
    # 0. User Request -> 1. Researcher -> 2. Financial Analyst -> 3. Marketing Strategist -> 4. Final Battlecard
    y_center = 330
    node_w, node_h = 290, 160

    # Box 0: Query Trigger
    q_box = (85, y_center - 60, 245, y_center + 60)
    draw_card(q_box, "#F1F5F9", "#94A3B8", radius=12, border_width=2)
    draw.text((105, y_center - 32), "Target Competitor", fill="#0F172A", font=f_node_title)
    draw.text((105, y_center - 5), 'Query: "Slack"', fill="#2563EB", font=f_node_role)
    draw.text((105, y_center + 18), "Multi-team TCO", fill="#64748B", font=f_node_role)

    # Arrow 0 -> 1
    draw_h_arrow(245, y_center, 335, y_center, color="#0284C7", text="Input Query")

    # Node 1: Researcher
    r_box = (335, y_center - 80, 625, y_center + 80)
    draw_card(r_box, "#F0F9FF", "#0284C7", radius=14, border_width=2)
    draw_badge(355, y_center - 65, "ROLE: AUDIT SPECIALIST", "#0284C7")
    draw.text((355, y_center - 35), "Senior Market Researcher", fill="#0F172A", font=f_node_title)
    draw.text((355, y_center - 10), "Extracts tiers, pricing & limits", fill="#475569", font=f_node_role)
    draw.text((355, y_center + 12), "LLM Temp: 0.1 | Zero Hallucination", fill="#64748B", font=f_node_role)
    draw.text((355, y_center + 38), "Tool: competitor_catalog_search", fill="#0369A1", font=f_node_tool)

    # Arrow 1 -> 2
    draw_h_arrow(625, y_center, 735, y_center, color="#D97706", text="Structured Pricing Table")

    # Node 2: Financial Analyst
    f_box = (735, y_center - 80, 1025, y_center + 80)
    draw_card(f_box, "#FFFBEB", "#D97706", radius=14, border_width=2)
    draw_badge(755, y_center - 65, "ROLE: PRICING STRATEGIST", "#D97706")
    draw.text((755, y_center - 35), "Principal Financial Analyst", fill="#0F172A", font=f_node_title)
    draw.text((755, y_center - 10), "Computes 50 & 100 seat TCO", fill="#475569", font=f_node_role)
    draw.text((755, y_center + 12), "LLM Temp: 0.0 | Safe AST Parser", fill="#64748B", font=f_node_role)
    draw.text((755, y_center + 38), "Tool: financial_tco_calculator", fill="#B45309", font=f_node_tool)

    # Arrow 2 -> 3
    draw_h_arrow(1025, y_center, 1135, y_center, color="#059669", text="Exact TCO & Discount Math")

    # Node 3: Marketing Strategist
    m_box = (1135, y_center - 80, 1425, y_center + 80)
    draw_card(m_box, "#ECFDF5", "#059669", radius=14, border_width=2)
    draw_badge(1155, y_center - 65, "ROLE: PRODUCT MARKETING", "#059669")
    draw.text((1155, y_center - 35), "VP Product Marketing", fill="#0F172A", font=f_node_title)
    draw.text((1155, y_center - 10), "Synthesizes sales battlecard", fill="#475569", font=f_node_role)
    draw.text((1155, y_center + 12), "LLM Temp: 0.4 | Persuasive Framing", fill="#64748B", font=f_node_role)
    draw.text((1155, y_center + 38), "Tool: battlecard_formatter", fill="#047857", font=f_node_tool)

    # Arrow 3 -> Final
    draw_h_arrow(1425, y_center, 1535, y_center, color="#6366F1", text="Validated Layout")

    # Box 4: Final Output
    out_box = (1535, y_center - 70, 1835, y_center + 70)
    draw_card(out_box, "#EEF2FF", "#6366F1", radius=14, border_width=2)
    draw_badge(1555, y_center - 55, "FINAL DELIVERABLE", "#6366F1")
    draw.text((1555, y_center - 25), "Executive Battlecard", fill="#0F172A", font=f_node_title)
    draw.text((1555, y_center + 2), "• 1. Exec Intelligence Summary", fill="#4338CA", font=f_node_role)
    draw.text((1555, y_center + 22), "• 2. Quantitative TCO Analysis", fill="#4338CA", font=f_node_role)
    draw.text((1555, y_center + 42), "• 3. Sales Objection Playbook", fill="#4338CA", font=f_node_role)

    # =========================================================================
    # SECTION 2: BOTTOM PANEL - HIERARCHICAL DELEGATION
    # =========================================================================
    hier_panel = (60, 550, 1860, 1100)
    draw_card(hier_panel, "#FFFFFF", "#CBD5E1", radius=16, border_width=2)
    draw.text((85, 570), "2. HIERARCHICAL DELEGATION (Process.hierarchical) — Supervisory Multi-Agent Management", fill="#1E293B", font=f_sec)
    draw_badge(950, 575, "SUPERVISORY AUDITING • DYNAMIC RE-ROUTING • 8,420 TOKENS • $0.00100", "#7C3AED")

    # Manager Node in Top Center
    mgr_w, mgr_h = 560, 120
    mgr_x1, mgr_y1 = 680, 625
    mgr_x2, mgr_y2 = mgr_x1 + mgr_w, mgr_y1 + mgr_h
    mgr_box = (mgr_x1, mgr_y1, mgr_x2, mgr_y2)
    draw_card(mgr_box, "#F5F3FF", "#7C3AED", radius=14, border_width=2)
    draw_badge(mgr_x1 + 20, mgr_y1 + 18, "SUPERVISORY MANAGER AGENT", "#7C3AED")
    draw.text((mgr_x1 + 20, mgr_y1 + 50), "Director of Market Strategy & Research Operations", fill="#0F172A", font=f_node_title)
    draw.text((mgr_x1 + 20, mgr_y1 + 78), "Assesses master objective • Delegates sub-tasks • Reviews outputs before final sign-off", fill="#5B21B6", font=f_node_role)
    draw.text((mgr_x1 + 20, mgr_y1 + 98), "allow_delegation = True • Multi-turn orchestration loops (8 turns)", fill="#6D28D9", font=f_node_tool)

    # 3 Sub-Agent Worker Nodes in Bottom Row
    worker_y1 = 860
    worker_y2 = worker_y1 + 140

    # Sub 1: Researcher Worker
    w1_box = (140, worker_y1, 460, worker_y2)
    draw_card(w1_box, "#F0F9FF", "#0284C7", radius=12, border_width=2)
    draw_badge(160, worker_y1 + 15, "DELEGATED SPECIALIST", "#0284C7")
    draw.text((160, worker_y1 + 45), "Researcher Agent", fill="#0F172A", font=f_node_title)
    draw.text((160, worker_y1 + 72), "Tool: competitor_catalog_search", fill="#0369A1", font=f_node_tool)
    draw.text((160, worker_y1 + 98), "allow_delegation = True", fill="#475569", font=f_node_role)

    # Sub 2: Financial Analyst Worker
    w2_box = (740, worker_y1, 1060, worker_y2)
    draw_card(w2_box, "#FFFBEB", "#D97706", radius=12, border_width=2)
    draw_badge(760, worker_y1 + 15, "DELEGATED SPECIALIST", "#D97706")
    draw.text((760, worker_y1 + 45), "Financial Analyst Agent", fill="#0F172A", font=f_node_title)
    draw.text((760, worker_y1 + 72), "Tool: financial_tco_calculator", fill="#B45309", font=f_node_tool)
    draw.text((760, worker_y1 + 98), "allow_delegation = True", fill="#475569", font=f_node_role)

    # Sub 3: Marketing Strategist Worker
    w3_box = (1340, worker_y1, 1660, worker_y2)
    draw_card(w3_box, "#ECFDF5", "#059669", radius=12, border_width=2)
    draw_badge(1360, worker_y1 + 15, "DELEGATED SPECIALIST", "#059669")
    draw.text((1360, worker_y1 + 45), "Marketer Agent", fill="#0F172A", font=f_node_title)
    draw.text((1360, worker_y1 + 72), "Tool: battlecard_formatter", fill="#047857", font=f_node_tool)
    draw.text((1360, worker_y1 + 98), "allow_delegation = True", fill="#475569", font=f_node_role)

    # Delegation lines from Manager to Workers with 2-way arrows
    def draw_delegation_flow(x_top, y_top, x_bot, y_bot, label_down, label_up, col="#7C3AED"):
        # Downward delegation
        draw.line([(x_top, y_top), (x_bot, y_bot)], fill=col, width=2)
        arr_size = 7
        draw.polygon([(x_bot, y_bot), (x_bot - arr_size, y_bot - arr_size * 1.5), (x_bot + arr_size, y_bot - arr_size * 1.5)], fill=col)
        # Upward return
        draw.line([(x_bot + 18, y_bot), (x_top + 18, y_top)], fill="#64748B", width=2)
        draw.polygon([(x_top + 18, y_top), (x_top + 18 - arr_size, y_top + arr_size * 1.5), (x_top + 18 + arr_size, y_top + arr_size * 1.5)], fill="#64748B")

    # Manager -> Researcher
    draw_delegation_flow(780, mgr_y2, 300, worker_y1, "1. Delegate Audit", "Verified Catalog")
    draw.text((450, 770), "1. Delegate Catalog Audit ➔", fill="#7C3AED", font=f_edge)
    draw.text((450, 795), "⬅ Verified Pricing Data", fill="#64748B", font=f_edge)

    # Manager -> Financial Analyst
    draw_delegation_flow(960, mgr_y2, 900, worker_y1, "2. Delegate Math", "TCO Results")
    draw.text((930, 770), "2. Delegate TCO Modeling ➔", fill="#7C3AED", font=f_edge)
    draw.text((930, 795), "⬅ Verified Calculations", fill="#64748B", font=f_edge)

    # Manager -> Marketer
    draw_delegation_flow(1140, mgr_y2, 1500, worker_y1, "3. Delegate Strategy", "Draft Battlecard")
    draw.text((1240, 770), "3. Delegate Positioning ➔", fill="#7C3AED", font=f_edge)
    draw.text((1240, 795), "⬅ Synthesized Battlecard", fill="#64748B", font=f_edge)

    # Manager Final Sign-off Arrow -> Output
    mgr_out_box = (1480, 640, 1820, 730)
    draw_card(mgr_out_box, "#FAF5FF", "#7C3AED", radius=12, border_width=2)
    draw_badge(1500, 652, "EXECUTIVE SIGN-OFF", "#7C3AED")
    draw.text((1500, 678), "Final Synthesized Brief", fill="#0F172A", font=f_node_title)
    draw.text((1500, 702), "Audited C-suite Intelligence", fill="#5B21B6", font=f_node_role)
    draw_h_arrow(mgr_x2, (mgr_y1 + mgr_y2) / 2, 1480, 685, color="#7C3AED", text="Quality Review & Approval")

    # Save to file
    img.save(IMG_PATH, "PNG", dpi=(300, 300))
    print(f"Successfully generated workflow diagram: {IMG_PATH} ({W}x{H})")


if __name__ == "__main__":
    create_diagram()
