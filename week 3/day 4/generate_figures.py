"""
Week 3 Day 4 — Architectural & Evaluation Figure Generator
==========================================================
Generates high-resolution production figures:
1. `figures/langgraph_architecture.png`: End-to-end LangGraph execution graph topology.
2. `figures/routing_accuracy_benchmark.png`: Category-level classification benchmark chart.
3. `figures/probabilistic_prediction_flow.png`: Pipeline from nickname resolution to disclaimer synthesis.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# Figure 1: LangGraph System Topology
# ------------------------------------------------------------------------------
def generate_architecture_figure():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    bg_color = "#0f172a"
    card_bg = "#1e293b"
    border_color = "#38bdf8"
    pred_color = "#f59e0b"
    ret_color = "#10b981"
    fact_color = "#6366f1"
    ref_color = "#ef4444"
    val_color = "#ec4899"
    text_color = "#f8fafc"
    muted_text = "#94a3b8"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Title
    ax.text(7, 9.5, "Week 3 Day 4 — AFL LangGraph Multi-Track Orchestration Topology",
            ha='center', va='center', fontsize=18, fontweight='bold', color=text_color)
    ax.text(7, 9.05, "Deterministic Intent Routing, Calibrated ML Predictions, Self-Correction & Fallbacks",
            ha='center', va='center', fontsize=12, color=muted_text)

    # Helper function for drawing boxes
    def draw_node(x, y, w, h, title, subtitle, color, icon=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.3",
                                      facecolor=card_bg, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, f"{icon} {title}", ha='center', va='center',
                fontsize=11, fontweight='bold', color=color)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center',
                fontsize=8.5, color=muted_text)

    # 1. START & Router Node
    draw_node(0.8, 4.2, 2.5, 1.4, "Router Node", "Regex & Semantic Intent\nClassification (4 Paths)", border_color, "🧭")
    ax.text(0.3, 4.9, "START", ha='center', va='center', fontsize=10, fontweight='bold', color="#38bdf8")

    # Connect START to Router
    ax.annotate('', xy=(0.8, 4.9), xytext=(0.4, 4.9),
                arrowprops=dict(arrowstyle="->", color=border_color, lw=2))

    # 2. Four Parallel Tool Branches
    draw_node(4.4, 7.2, 2.8, 1.3, "Prediction Node", "Nicknames -> ML Pipeline\nCalibrated Winner & Player", pred_color, "🔮")
    draw_node(4.4, 5.2, 2.8, 1.3, "Retrieval Node", "Structured Feature Tables\nPlayer & Team Stat Lookups", ret_color, "📊")
    draw_node(4.4, 3.2, 2.8, 1.3, "Factual Node", "AFL Semantic VectorStore\nRules, Grounds, Heritage", fact_color, "📖")
    draw_node(4.4, 1.2, 2.8, 1.3, "Off-Topic Refusal", "Polite Scope Redirection\nSafe AFL Footy Boundaries", ref_color, "🛡️")

    # Arrows from Router to 4 nodes
    ax.annotate('', xy=(4.4, 7.85), xytext=(3.3, 5.3), arrowprops=dict(arrowstyle="->", color=pred_color, lw=1.8))
    ax.annotate('', xy=(4.4, 5.85), xytext=(3.3, 5.0), arrowprops=dict(arrowstyle="->", color=ret_color, lw=1.8))
    ax.annotate('', xy=(4.4, 3.85), xytext=(3.3, 4.7), arrowprops=dict(arrowstyle="->", color=fact_color, lw=1.8))
    ax.annotate('', xy=(4.4, 1.85), xytext=(3.3, 4.4), arrowprops=dict(arrowstyle="->", color=ref_color, lw=1.8))

    # Labels on edges
    ax.text(3.7, 6.8, "prediction", fontsize=9, color=pred_color, fontweight='bold', rotation=30)
    ax.text(3.7, 5.6, "retrieval", fontsize=9, color=ret_color, fontweight='bold', rotation=12)
    ax.text(3.7, 4.1, "factual", fontsize=9, color=fact_color, fontweight='bold', rotation=-12)
    ax.text(3.7, 2.8, "off_topic", fontsize=9, color=ref_color, fontweight='bold', rotation=-30)

    # 3. Validation Node
    draw_node(8.2, 4.6, 2.5, 1.5, "Validation Node", "Tool Payload Audit\nCheck Error & Resolvability", val_color, "🔍")

    # Arrows into Validation
    ax.annotate('', xy=(8.2, 5.8), xytext=(7.2, 7.85), arrowprops=dict(arrowstyle="->", color=pred_color, lw=1.5))
    ax.annotate('', xy=(8.2, 5.4), xytext=(7.2, 5.85), arrowprops=dict(arrowstyle="->", color=ret_color, lw=1.5))
    ax.annotate('', xy=(8.2, 5.0), xytext=(7.2, 3.85), arrowprops=dict(arrowstyle="->", color=fact_color, lw=1.5))

    # Clarification Node (Self-Correction & Fallbacks)
    draw_node(8.2, 2.0, 2.5, 1.3, "Clarification Node", "Interactive Prompt Loop\nUnsupported Stat Fallback", "#eab308", "🔄")

    # Arrow from Validation to Clarification
    ax.annotate('', xy=(9.45, 3.3), xytext=(9.45, 4.6),
                arrowprops=dict(arrowstyle="->", color="#eab308", lw=2, linestyle="--"))
    ax.text(9.55, 3.9, "needs_clarification\nor unsupported", fontsize=8, color="#eab308", fontweight='bold')

    # 4. Response Formatter Node
    draw_node(11.4, 4.2, 2.4, 1.8, "Response Formatter", "Probabilistic Framing\nKey Drivers & Bounds\nMandatory Disclaimer", "#a855f7", "✍️")

    # Arrows into Formatter
    ax.annotate('', xy=(11.4, 5.4), xytext=(10.7, 5.4), arrowprops=dict(arrowstyle="->", color=val_color, lw=2))
    ax.text(10.75, 5.6, "valid", fontsize=8.5, color=val_color, fontweight='bold')

    ax.annotate('', xy=(11.4, 4.8), xytext=(10.7, 2.65), arrowprops=dict(arrowstyle="->", color="#eab308", lw=1.8))

    # Refusal directly to Formatter
    ax.annotate('', xy=(12.2, 4.2), xytext=(7.2, 1.85),
                arrowprops=dict(arrowstyle="->", color=ref_color, lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))

    # Formatter to END
    ax.text(13.6, 5.1, "END", ha='center', va='center', fontsize=11, fontweight='bold', color="#a855f7")
    ax.annotate('', xy=(13.3, 5.1), xytext=(12.8, 5.1),
                arrowprops=dict(arrowstyle="->", color="#a855f7", lw=2))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "langgraph_architecture.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Generated: {out_path}")


# ------------------------------------------------------------------------------
# Figure 2: Router Benchmark Accuracy
# ------------------------------------------------------------------------------
def generate_benchmark_figure():
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bg_color = "#0f172a"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor("#1e293b")

    categories = ['Prediction\n(Matches & Players)', 'Retrieval\n(Player & Club Stats)', 'Factual Knowledge\n(Rules & Grounds)', 'Off-Topic Refusal\n(Scope Guardrails)']
    accuracies = [100.0, 100.0, 100.0, 100.0]
    colors = ['#f59e0b', '#10b981', '#6366f1', '#ef4444']

    bars = ax.bar(categories, accuracies, color=colors, width=0.55, edgecolor='#38bdf8', linewidth=1.5)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=12, fontweight='bold', color='#f8fafc')
    ax.set_title("LangGraph Intent Router Evaluation Benchmark (20/20 Passed — 100% Accuracy)",
                 fontsize=14, fontweight='bold', color='#f8fafc', pad=15)

    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f"{height:.1f}% (5/5)", ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#f8fafc')

    for spine in ax.spines.values():
        spine.set_color('#334155')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "routing_accuracy_benchmark.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Generated: {out_path}")


# ------------------------------------------------------------------------------
# Figure 3: Prediction Tool Pipeline & Probabilistic Framing
# ------------------------------------------------------------------------------
def generate_prediction_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    bg_color = "#0f172a"
    card_bg = "#1e293b"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.text(6, 6.5, "Task 3: Production Prediction Tool Pipeline & Probabilistic Framing",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#f8fafc')
    ax.text(6, 6.1, "From Raw Colloquial User Slang to Grounded, Calibrated Forecasts with Disclaimers",
            ha='center', va='center', fontsize=11, color='#94a3b8')

    stages = [
        {"x": 0.5, "title": "1. Slang Resolution", "sub": "Colloquial Nicknames:\n'Pies' -> Collingwood\n'Cats' -> Geelong\n'this week' -> 2025-09-27", "col": "#38bdf8"},
        {"x": 3.4, "title": "2. State Cache Lookup", "sub": "Rolling Margins (Diff: +51.6)\nLadder Differential\nVenue Win Rate (MCG)\nTravel Factor (VIC vs Interstate)", "col": "#f59e0b"},
        {"x": 6.3, "title": "3. Calibrated GBDT Model", "sub": "Isotonic Probabilities:\nHome Win: 37.4%\nAway Win: 62.6%\nTier: Clear Favorite\nMargin: 12-24 pts", "col": "#10b981"},
        {"x": 9.2, "title": "4. Grounded Synthesis", "sub": "Probabilistic Framing\nTop 2-3 Feature Drivers\nExact Confidence Bounds\nSafety Disclaimer", "col": "#a855f7"}
    ]

    for s in stages:
        rect = patches.FancyBboxPatch((s["x"], 1.5), 2.3, 3.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                                      facecolor=card_bg, edgecolor=s["col"], linewidth=2)
        ax.add_patch(rect)
        ax.text(s["x"] + 1.15, 4.8, s["title"], ha='center', va='center',
                fontsize=11, fontweight='bold', color=s["col"])
        ax.text(s["x"] + 1.15, 3.1, s["sub"], ha='center', va='center',
                fontsize=8.5, color='#cbd5e1', multialignment='center')

    # Arrows between stages
    for x in [2.8, 5.7, 8.6]:
        ax.annotate('', xy=(x + 0.6, 3.4), xytext=(x, 3.4),
                    arrowprops=dict(arrowstyle="->", color='#f8fafc', lw=2))

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "probabilistic_prediction_flow.png")
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SUCCESS] Generated: {out_path}")


if __name__ == "__main__":
    generate_architecture_figure()
    generate_benchmark_figure()
    generate_prediction_pipeline_figure()
