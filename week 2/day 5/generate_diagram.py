"""Generate an enterprise-grade workflow architecture diagram for Day 5 Capstone using Pillow."""

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
        f_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 34)
        f_subtitle = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
        f_sec = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 20)
        f_node_title = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 17)
        f_node_role = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
        f_node_tool = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 12)
        f_edge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 12)
        f_badge = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 11)
    except Exception:
        f_title = f_sec = f_node_title = ImageFont.load_default()
        f_subtitle = f_node_role = f_node_tool = f_edge = f_badge = ImageFont.load_default()

    # Header
    draw.text((60, 35), "Week 2 Day 5 Capstone — Web3Geeks Client Onboarding Agent Architecture", fill="#0F172A", font=f_title)
    draw.text((60, 80), "Stateful Control-Flow (LangGraph) • AST Financial Isolation • Cyclic Self-Correction • HITL Governance • FastAPI", fill="#475569", font=f_subtitle)

    def draw_card(box, fill_color, border_color, radius=12, border_width=2):
        draw.rounded_rectangle(box, radius=radius, fill=fill_color, outline=border_color, width=border_width)

    def draw_badge(x, y, text, bg_color, text_color="#FFFFFF"):
        bbox = draw.textbbox((x, y), text, font=f_badge)
        pad_x, pad_y = 8, 3
        badge_box = (bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y)
        draw.rounded_rectangle(badge_box, radius=5, fill=bg_color)
        draw.text((x, y), text, fill=text_color, font=f_badge)

    def draw_h_arrow(x1, y1, x2, y2, color="#64748B", text=None, text_above=True, width=3):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
        arr_size = 7
        draw.polygon([(x2, y2), (x2 - arr_size * 1.5, y2 - arr_size), (x2 - arr_size * 1.5, y2 + arr_size)], fill=color)
        if text:
            t_box = draw.textbbox((0, 0), text, font=f_edge)
            tw = t_box[2] - t_box[0]
            th = t_box[3] - t_box[1]
            tx = (x1 + x2) / 2 - tw / 2
            ty = y1 - th - 6 if text_above else y1 + 6
            draw.rectangle((tx - 4, ty - 2, tx + tw + 4, ty + th + 2), fill="#F8FAFC")
            draw.text((tx, ty), text, fill=color, font=f_edge)

    # Main Pipeline Container
    panel = (50, 125, 1870, 780)
    draw_card(panel, "#FFFFFF", "#CBD5E1", radius=16, border_width=2)
    draw.text((75, 145), "Core Agent Execution Pipeline (LangGraph StateGraph Engine)", fill="#1E293B", font=f_sec)
    draw_badge(750, 150, "STATEFUL CYCLIC GRAPH • HUMAN-IN-THE-LOOP CHECKPOINT • AST FINANCIALS", "#2563EB")

    # Flow Coordinates
    y_row1 = 300
    y_row2 = 560

    # 1. Inbound Request
    draw_card((75, y_row1 - 65, 275, y_row1 + 65), "#F1F5F9", "#64748B", radius=12)
    draw_badge(95, y_row1 - 50, "INBOUND TRIGGER", "#64748B")
    draw.text((95, y_row1 - 22), "Client Inquiry", fill="#0F172A", font=f_node_title)
    draw.text((95, y_row1 + 5), "Scope, Budget & SLA", fill="#475569", font=f_node_role)
    draw.text((95, y_row1 + 25), "REST POST /api/v1/onboard", fill="#2563EB", font=f_node_tool)

    draw_h_arrow(275, y_row1, 355, y_row1, color="#3B82F6", text="Validate")

    # 2. Node 1: Input Validation
    draw_card((355, y_row1 - 75, 625, y_row1 + 75), "#EFF6FF", "#3B82F6", radius=12)
    draw_badge(375, y_row1 - 60, "STAGE 1: DEFENSE", "#3B82F6")
    draw.text((375, y_row1 - 32), "1. validate_inquiry", fill="#0F172A", font=f_node_title)
    draw.text((375, y_row1 - 5), "• Length & schema validation", fill="#475569", font=f_node_role)
    draw.text((375, y_row1 + 15), "• Adversarial prompt screening", fill="#475569", font=f_node_role)
    draw.text((375, y_row1 + 35), "• Budget floor check (min $3,500)", fill="#475569", font=f_node_role)

    # Conditional Arrow: Failure Branch down
    draw.line([(490, y_row1 + 75), (490, y_row2 - 60)], fill="#EF4444", width=2)
    draw.polygon([(490, y_row2 - 60), (484, y_row2 - 70), (496, y_row2 - 70)], fill="#EF4444")
    draw.text((500, (y_row1 + y_row2) / 2 - 10), "Invalid / Injection", fill="#EF4444", font=f_edge)

    # Failure Handler Node
    draw_card((355, y_row2 - 60, 625, y_row2 + 60), "#FEF2F2", "#EF4444", radius=12)
    draw_badge(375, y_row2 - 45, "ERROR HANDLING", "#EF4444")
    draw.text((375, y_row2 - 18), "7. failure_handler", fill="#991B1B", font=f_node_title)
    draw.text((375, y_row2 + 8), "Graceful refusal, structured diagnostic", fill="#7F1D1D", font=f_node_role)
    draw.text((375, y_row2 + 28), "notice & zero financial dispatch.", fill="#7F1D1D", font=f_node_role)

    draw_h_arrow(625, y_row1, 705, y_row1, color="#0284C7", text="Passed")

    # 3. Node 2: Service Scoping
    draw_card((705, y_row1 - 75, 985, y_row1 + 75), "#F0F9FF", "#0284C7", radius=12)
    draw_badge(725, y_row1 - 60, "STAGE 2: SCOPING", "#0284C7")
    draw.text((725, y_row1 - 32), "2. scope_services", fill="#0F172A", font=f_node_title)
    draw.text((725, y_row1 - 5), "Queries Web3Geeks Service Catalog", fill="#475569", font=f_node_role)
    draw.text((725, y_row1 + 15), "Matches deliverables & tech stack", fill="#475569", font=f_node_role)
    draw.text((725, y_row1 + 40), "Tool: ServiceCatalogSearchTool", fill="#0369A1", font=f_node_tool)

    draw_h_arrow(985, y_row1, 1065, y_row1, color="#D97706", text="Catalog Data")

    # 4. Node 3: AST Budget Calculation
    draw_card((1065, y_row1 - 75, 1345, y_row1 + 75), "#FFFBEB", "#D97706", radius=12)
    draw_badge(1085, y_row1 - 60, "STAGE 3: QUANTITATIVE", "#D97706")
    draw.text((1085, y_row1 - 32), "3. calculate_budget", fill="#0F172A", font=f_node_title)
    draw.text((1085, y_row1 - 5), "Deterministic AST Arithmetic Evaluator", fill="#475569", font=f_node_role)
    draw.text((1085, y_row1 + 15), "Milestones: 40% / 40% / 20% split", fill="#475569", font=f_node_role)
    draw.text((1085, y_row1 + 40), "Tool: DeterministicBudgetCalculator", fill="#B45309", font=f_node_tool)

    draw_h_arrow(1345, y_row1, 1425, y_row1, color="#059669", text="Verified Math")

    # 5. Node 4: Proposal Drafting
    draw_card((1425, y_row1 - 75, 1725, y_row1 + 75), "#ECFDF5", "#059669", radius=12)
    draw_badge(1445, y_row1 - 60, "STAGE 4: DRAFTING", "#059669")
    draw.text((1445, y_row1 - 32), "4. generate_proposal", fill="#0F172A", font=f_node_title)
    draw.text((1445, y_row1 - 5), "Synthesizes formal client SOW", fill="#475569", font=f_node_role)
    draw.text((1445, y_row1 + 15), "Integrates exact AST milestone totals", fill="#475569", font=f_node_role)
    draw.text((1445, y_row1 + 40), "Model: Gemini Flash (Temp: 0.3)", fill="#047857", font=f_node_tool)

    # Downward Arrow from Node 4 to Node 5 (Critique)
    draw.line([(1575, y_row1 + 75), (1575, y_row2 - 75)], fill="#7C3AED", width=3)
    draw.polygon([(1575, y_row2 - 75), (1569, y_row2 - 85), (1581, y_row2 - 85)], fill="#7C3AED")
    draw.text((1585, (y_row1 + y_row2) / 2 - 10), "Draft Review", fill="#7C3AED", font=f_edge)

    # 6. Node 5: Self-Correction Critique
    draw_card((1425, y_row2 - 75, 1725, y_row2 + 75), "#FAF5FF", "#7C3AED", radius=12)
    draw_badge(1445, y_row2 - 60, "STAGE 5: GOVERNANCE CRITIQUE", "#7C3AED")
    draw.text((1445, y_row2 - 32), "5. critique_proposal", fill="#0F172A", font=f_node_title)
    draw.text((1445, y_row2 - 5), "Validates 5 mandatory SOW sections", fill="#475569", font=f_node_role)
    draw.text((1445, y_row2 + 15), "Checks budget proof & SLA clauses", fill="#475569", font=f_node_role)
    draw.text((1445, y_row2 + 40), "Tool: ContractTemplateFormatterTool", fill="#6D28D9", font=f_node_tool)

    # Self-correction loop: Arrow from Critique back to Drafting (curved / left loop)
    draw.line([(1725, y_row2), (1765, y_row2), (1765, y_row1), (1725, y_row1)], fill="#D97706", width=2)
    draw.polygon([(1725, y_row1), (1735, y_row1 - 5), (1735, y_row1 + 5)], fill="#D97706")
    draw.text((1772, (y_row1 + y_row2) / 2 - 10), "Score < 80 (Loop max 2x)", fill="#D97706", font=f_edge)

    # Leftward Arrow from Critique to HITL Checkpoint
    draw_h_arrow(1425, y_row2, 1265, y_row2, color="#4F46E5", text="Score >= 80")

    # 7. Node 6: Human-in-the-Loop Checkpoint
    draw_card((935, y_row2 - 80, 1265, y_row2 + 80), "#EEF2FF", "#4F46E5", radius=14, border_width=3)
    draw_badge(955, y_row2 - 65, "STAGE 6: HUMAN-IN-THE-LOOP GATE", "#4F46E5")
    draw.text((955, y_row2 - 35), "6. human_checkpoint", fill="#0F172A", font=f_node_title)
    draw.text((955, y_row2 - 10), "Consequential Action Gate: Partner Approval", fill="#312E81", font=f_node_role)
    draw.text((955, y_row2 + 12), "Holds legally binding SOW contract dispatch", fill="#475569", font=f_node_role)
    draw.text((955, y_row2 + 38), "Endpoint: POST /api/v1/approve", fill="#4338CA", font=f_node_tool)

    # Output Arrow to Final Output
    draw_h_arrow(935, y_row2, 775, y_row2, color="#10B981", text="Approved (True)")

    # 8. Final Contract Output Box
    draw_card((650, y_row2 - 60, 775, y_row2 + 60), "#ECFDF5", "#10B981", radius=12)
    draw_badge(665, y_row2 - 45, "DISPATCHED", "#10B981")
    draw.text((665, y_row2 - 18), "Signed SOW", fill="#065F46", font=f_node_title)
    draw.text((665, y_row2 + 8), "Binding Contract", fill="#047857", font=f_node_role)
    draw.text((665, y_row2 + 28), "Committed to Client", fill="#047857", font=f_node_role)

    # Bottom Container: FastAPI & Monitoring Architecture
    api_panel = (50, 830, 1870, 1110)
    draw_card(api_panel, "#F8FAFC", "#94A3B8", radius=14, border_width=2)
    draw.text((75, 850), "Production Deployment & Observability Infrastructure (FastAPI + Structured Middleware)", fill="#1E293B", font=f_sec)

    # 4 Sub-Boxes for API Architecture
    bw = 410
    bh = 175
    bx1 = 80
    gap = 40

    # Box A: Inbound REST API
    b1 = (bx1, 895, bx1 + bw, 895 + bh)
    draw_card(b1, "#FFFFFF", "#CBD5E1", radius=10)
    draw_badge(bx1 + 15, 910, "REST ENDPOINTS", "#2563EB")
    draw.text((bx1 + 15, 938), "FastAPI Application Layer", fill="#0F172A", font=f_node_title)
    draw.text((bx1 + 15, 965), "• POST /api/v1/onboard", fill="#334155", font=f_node_role)
    draw.text((bx1 + 15, 990), "• POST /api/v1/approve (HITL)", fill="#334155", font=f_node_role)
    draw.text((bx1 + 15, 1015), "• GET /health (Readiness)", fill="#334155", font=f_node_role)
    draw.text((bx1 + 15, 1040), "• GET /api/v1/metrics (Prometheus)", fill="#334155", font=f_node_role)

    # Box B: Security & Validation
    bx2 = bx1 + bw + gap
    b2 = (bx2, 895, bx2 + bw, 895 + bh)
    draw_card(b2, "#FFFFFF", "#CBD5E1", radius=10)
    draw_badge(bx2 + 15, 910, "VALIDATION & SECURITY", "#0284C7")
    draw.text((bx2 + 15, 938), "Pydantic Schema Defense", fill="#0F172A", font=f_node_title)
    draw.text((bx2 + 15, 965), "• OnboardingRequest payload validation", fill="#334155", font=f_node_role)
    draw.text((bx2 + 15, 990), "• Prompt injection keyword screening", fill="#334155", font=f_node_role)
    draw.text((bx2 + 15, 1015), "• $3,500 minimum budget enforcement", fill="#334155", font=f_node_role)
    draw.text((bx2 + 15, 1040), "• CORS middleware enabled", fill="#334155", font=f_node_role)

    # Box C: Observability Middleware
    bx3 = bx2 + bw + gap
    b3 = (bx3, 895, bx3 + bw, 895 + bh)
    draw_card(b3, "#FFFFFF", "#CBD5E1", radius=10)
    draw_badge(bx3 + 15, 910, "OBSERVABILITY", "#D97706")
    draw.text((bx3 + 15, 938), "Telemetry & Logging", fill="#0F172A", font=f_node_title)
    draw.text((bx3 + 15, 965), "• Per-request HTTP latency logging", fill="#334155", font=f_node_role)
    draw.text((bx3 + 15, 990), "• Input/Output token cost tracking", fill="#334155", font=f_node_role)
    draw.text((bx3 + 15, 1015), "• Tool call timing & AST audits", fill="#334155", font=f_node_role)
    draw.text((bx3 + 15, 1040), "• In-memory error & rejection rates", fill="#334155", font=f_node_role)

    # Box D: Production Monitoring Checklist
    bx4 = bx3 + bw + gap
    b4 = (bx4, 895, bx4 + bw, 895 + bh)
    draw_card(b4, "#FFFFFF", "#CBD5E1", radius=10)
    draw_badge(bx4 + 15, 910, "SRE RUNBOOK", "#7C3AED")
    draw.text((bx4 + 15, 938), "Production Alert Thresholds", fill="#0F172A", font=f_node_title)
    draw.text((bx4 + 15, 965), "• Error Rate: Alert if > 2.0% in 5m", fill="#334155", font=f_node_role)
    draw.text((bx4 + 15, 990), "• Latency p95: Alert if > 4.5 seconds", fill="#334155", font=f_node_role)
    draw.text((bx4 + 15, 1015), "• Cost Drift: Alert if > $0.005/request", fill="#334155", font=f_node_role)
    draw.text((bx4 + 15, 1040), "• Quality Drift: Bi-weekly re-eval", fill="#334155", font=f_node_role)

    # Save diagram
    img.save(IMG_PATH, "PNG", dpi=(300, 300))
    print(f"Successfully generated workflow diagram: {IMG_PATH} ({W}x{H})")


if __name__ == "__main__":
    create_diagram()
