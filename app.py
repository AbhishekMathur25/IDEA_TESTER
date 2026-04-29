"""
Idea Tester — Main Streamlit Application
CEO Strategy Simulation Engine
"""

import sys
import os
import time
import json
import logging

# Enable backend terminal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True
)

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import settings

# ── Top-level engine imports (Optimization #2) ────────────
from engine.llm_client import LLMClient
from engine.knowledge_graph import KnowledgeGraph
from engine.persona_generator import PersonaGenerator
from engine.simulation_runner import SimulationRunner
from engine.report_agent import ReportAgent
from agents.research_orchestrator import ResearchOrchestrator
from agents.seed_packager import SeedPackager
from agents.report_formatter import ReportFormatter

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="Idea Tester — Strategy Simulator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root Variables ──────────────────────────────────── */
:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2235;
    --bg-card-hover: #1f2a42;
    --accent-primary: #6366f1;
    --accent-secondary: #8b5cf6;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --accent-danger: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-subtle: rgba(255,255,255,0.06);
    --glow-primary: rgba(99,102,241,0.15);
    --glow-success: rgba(16,185,129,0.15);
}

/* ── Global ──────────────────────────────────────────── */
.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hero Section ────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f1f5f9 0%, #6366f1 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Cards ───────────────────────────────────────────── */
.sim-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.sim-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-primary);
    box-shadow: 0 0 30px var(--glow-primary);
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* ── Decision Badge ──────────────────────────────────── */
.decision-go {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    font-size: 1.5rem;
    font-weight: 800;
    padding: 1rem 2rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 0 40px var(--glow-success);
}
.decision-nogo {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    color: white;
    font-size: 1.5rem;
    font-weight: 800;
    padding: 1rem 2rem;
    border-radius: 12px;
    text-align: center;
}
.decision-conditional {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white;
    font-size: 1.5rem;
    font-weight: 800;
    padding: 1rem 2rem;
    border-radius: 12px;
    text-align: center;
}

/* ── Progress ────────────────────────────────────────── */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
}
.pipeline-step.active {
    border-color: var(--accent-primary);
    box-shadow: 0 0 20px var(--glow-primary);
}
.pipeline-step.done {
    border-color: var(--accent-success);
    opacity: 0.8;
}

/* ── Risk Matrix ─────────────────────────────────────── */
.risk-high { color: #ef4444; font-weight: 700; }
.risk-medium { color: #f59e0b; font-weight: 600; }
.risk-low { color: #10b981; font-weight: 500; }

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
}
</style>
""", unsafe_allow_html=True)


# ── State Initialization ─────────────────────────────────
def init_state():
    defaults = {
        "pipeline_stage": "input",       # input, researching, packaging, simulating, reporting, done
        "idea": "",
        "company": "",
        "sector": "",
        "geography": "",
        "research_output": None,
        "seed_document": None,
        "sim_rounds": None,
        "report": None,
        "formatted_report": None,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Shared LLM Client (Optimization #1) ────────────────────
def get_llm() -> LLMClient:
    """Return a shared LLMClient instance, creating it once per session."""
    provider = st.session_state.get("llm_provider", "gemini")
    cache_key = f"_llm_{provider}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = LLMClient(provider=provider)
    return st.session_state[cache_key]


# ── Graph Visualization ────────────────────────────────────

# Color palette for entity types
ENTITY_COLORS = {
    "Company": "#6366f1",
    "Consumer": "#10b981",
    "Regulator": "#f59e0b",
    "Competitor": "#ef4444",
    "Investor": "#8b5cf6",
    "Employee": "#06b6d4",
    "Media": "#ec4899",
    "Technology": "#14b8a6",
    "Market": "#f97316",
    "Organization": "#a855f7",
}
ENTITY_SHAPES = {
    "Company": "diamond",
    "Competitor": "diamond",
    "Regulator": "square",
}
DEFAULT_COLOR = "#64748b"


def show_graph(kg, height=600, interactive_panel=False, key_suffix=""):
    """Render an interactive knowledge graph with optional filter + detail panel."""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        all_entities = kg.get_all_entities()
        all_rels = kg.relationships

        # ── Filter Controls ──────────────────────────
        if interactive_panel:
            # Discover all entity types present
            entity_types = sorted(set(e.entity_type for e in all_entities))
            
            filter_cols = st.columns(min(len(entity_types), 5))
            active_types = []
            for i, etype in enumerate(entity_types):
                col = filter_cols[i % len(filter_cols)]
                color = ENTITY_COLORS.get(etype, DEFAULT_COLOR)
                if col.checkbox(
                    f"🔵 {etype}",
                    value=True,
                    key=f"graph_filter_{etype}_{key_suffix}",
                ):
                    active_types.append(etype)

            # Filter entities
            filtered_entities = [e for e in all_entities if e.entity_type in active_types]
            filtered_ids = {e.id for e in filtered_entities}
            filtered_rels = [r for r in all_rels if r.source_id in filtered_ids and r.target_id in filtered_ids]
        else:
            filtered_entities = all_entities
            filtered_rels = all_rels

        if not filtered_entities:
            st.info("No entities visible. Enable at least one entity type above.")
            return None

        # ── Build Nodes ──────────────────────────────
        nodes = []
        for entity in filtered_entities:
            color = ENTITY_COLORS.get(entity.entity_type, DEFAULT_COLOR)
            shape = ENTITY_SHAPES.get(entity.entity_type, "circle")
            
            # Size based on influence
            influence = entity.attributes.get("influence", "medium") if entity.attributes else "medium"
            size = 35 if influence == "high" else 25 if influence == "medium" else 18

            # Build rich tooltip
            tooltip_parts = [
                f"<b>{entity.name}</b>",
                f"Type: {entity.entity_type}",
                f"Influence: {influence}",
            ]
            if entity.description:
                tooltip_parts.append(f"<br/>{entity.description[:200]}")

            nodes.append(Node(
                id=entity.id,
                label=entity.name,
                title="\n".join(tooltip_parts),
                symbolType=shape,
                size=size,
                color=color,
                font={"color": "#e2e8f0", "size": 12},
            ))

        # ── Build Edges ──────────────────────────────
        edges = []
        for rel in filtered_rels:
            edges.append(Edge(
                source=rel.source_id,
                target=rel.target_id,
                label=rel.relation_type.replace("_", " "),
                title=rel.fact or rel.relation_type,
                color="#4b556380",
                strokeWidth=1.5,
            ))

        # ── Config ───────────────────────────────────
        config = Config(
            width=None,
            height=height,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            staticGraphWithDragAndDrop=True,
            collapsible=True,
            node={
                "labelProperty": "label",
                "renderLabel": True,
            },
            link={
                "labelProperty": "label",
                "renderLabel": True,
                "fontSize": 10,
                "color": "#4b556380",
            },
        )

        # ── Check if we have a focused node ──────────
        focus_key = f"_graph_focus_{key_suffix}"
        focused_id = st.session_state.get(focus_key)

        if interactive_panel and focused_id:
            # Render NEIGHBORHOOD subgraph — selected node + all direct connections
            entity = next((e for e in all_entities if e.id == focused_id), None)
            if entity:
                rels = kg.get_entity_relationships(entity.id)
                neighbor_ids = set()
                for r in rels:
                    neighbor_ids.add(r.source_id)
                    neighbor_ids.add(r.target_id)
                neighbor_ids.add(entity.id)

                # Build focused nodes
                focus_nodes = []
                for ent in all_entities:
                    if ent.id not in neighbor_ids:
                        continue
                    color = ENTITY_COLORS.get(ent.entity_type, DEFAULT_COLOR)
                    shape = ENTITY_SHAPES.get(ent.entity_type, "circle")
                    is_center = ent.id == entity.id
                    focus_nodes.append(Node(
                        id=ent.id,
                        label=ent.name,
                        title=f"{ent.entity_type}: {ent.description[:200]}",
                        symbolType=shape,
                        size=45 if is_center else 25,
                        color=color,
                        font={"color": "#e2e8f0", "size": 14 if is_center else 12},
                        strokeColor="#ffffff" if is_center else None,
                        strokeWidth=3.0 if is_center else 0,
                    ))

                # Build focused edges
                focus_edges = []
                for r in rels:
                    focus_edges.append(Edge(
                        source=r.source_id,
                        target=r.target_id,
                        label=r.relation_type.replace("_", " "),
                        title=r.fact or r.relation_type,
                        color=ENTITY_COLORS.get(entity.entity_type, "#6366f1") + "90",
                        strokeWidth=2.5,
                    ))

                # Header
                color = ENTITY_COLORS.get(entity.entity_type, DEFAULT_COLOR)
                st.markdown(f"""
                <div style="padding:12px 16px; border-radius:10px; border-left:4px solid {color}; background:{color}10; margin-bottom:12px;">
                    <span style="font-size:1.2rem; font-weight:700; color:{color};">{entity.name}</span>
                    <span style="color:#94a3b8; margin-left:8px;">{entity.entity_type} · {len(rels)} connections</span>
                </div>
                """, unsafe_allow_html=True)

                # Back button
                if st.button("← Show Full Graph", key=f"back_full_{key_suffix}"):
                    del st.session_state[focus_key]
                    st.rerun()

                # Render focused subgraph
                focus_config = Config(
                    width=None, height=height - 60, directed=True,
                    physics=True, hierarchical=False,
                    nodeHighlightBehavior=True, highlightColor="#F7A7A6",
                    staticGraphWithDragAndDrop=True,
                    node={"labelProperty": "label", "renderLabel": True},
                    link={"labelProperty": "label", "renderLabel": True, "fontSize": 11},
                )
                agraph(nodes=focus_nodes, edges=focus_edges, config=focus_config)

                # Details below
                det_col1, det_col2 = st.columns([1, 1])
                with det_col1:
                    st.markdown(f"**Description:** {entity.description}")
                    if entity.attributes:
                        for k, v in entity.attributes.items():
                            st.caption(f"• {k}: **{v}**")

                with det_col2:
                    st.markdown(f"**Connected Entities ({len(rels)}):**")
                    for rel in rels:
                        other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
                        other = kg.get_entity(other_id)
                        other_name = other.name if other else other_id
                        other_type = other.entity_type if other else ""
                        direction = "→" if rel.source_id == entity.id else "←"
                        oc = ENTITY_COLORS.get(other_type, DEFAULT_COLOR)
                        st.markdown(
                            f"{direction} <span style='color:{oc};font-weight:600;'>{other_name}</span> "
                            f"<span style='color:#64748b;'>({rel.relation_type.replace('_',' ')})</span>",
                            unsafe_allow_html=True,
                        )
                        if rel.fact:
                            st.caption(f"  _{rel.fact[:150]}_")

        else:
            # ── Render Full Graph ─────────────────────────
            selected_node = agraph(nodes=nodes, edges=edges, config=config)

            # If a node was clicked, focus it
            if interactive_panel and selected_node:
                st.session_state[focus_key] = selected_node
                st.rerun()

        # ── Legend ───────────────────────────────────
        if interactive_panel:
            present_types = sorted(set(e.entity_type for e in filtered_entities))
            legend_html = " &nbsp;".join(
                f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:12px;">'
                f'<span style="width:10px;height:10px;border-radius:50%;background:{ENTITY_COLORS.get(t, DEFAULT_COLOR)};display:inline-block;"></span>'
                f'<span style="color:#94a3b8;font-size:12px;">{t}</span></span>'
                for t in present_types
            )
            st.markdown(f"<div style='text-align:center;margin-top:8px;'>{legend_html}</div>", unsafe_allow_html=True)

        return None

    except Exception as ex:
        st.error(f"Graph viz failed: {ex}")
        return None


# ── PDF Report Generator ──────────────────────────────────

def _generate_pdf_report(report: dict, recommendation: dict, idea: str) -> bytes:
    """Generate a professional PDF report from simulation results."""
    from fpdf import FPDF
    from datetime import datetime

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Cover Page ────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.ln(40)
    pdf.cell(0, 15, "Idea Tester", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "AI Strategy Simulation Report", ln=True, align="C")
    pdf.ln(20)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Idea: {idea}", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=True, align="C")

    # Verdict
    verdict = recommendation.get("verdict", "N/A")
    confidence = recommendation.get("confidence", "N/A")
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 12, f"Verdict: {verdict}  |  Confidence: {confidence}", ln=True, align="C")

    # ── Executive Summary ─────────────────────────────
    summary = recommendation.get("summary", report.get("executive_summary", ""))
    if summary:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, _clean_text(summary))

    # ── Report Sections ──────────────────────────────
    for section in report.get("sections", []):
        title = section.get("title", "Section")
        content = section.get("content", "")
        if content:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 10, _clean_text(title), ln=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, _clean_text(content))

    # ── Risk Matrix ──────────────────────────────────
    risk_matrix = report.get("risk_matrix", [])
    if risk_matrix:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Risk Matrix", ln=True)
        pdf.ln(4)
        for risk in risk_matrix:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 7, f"- {_clean_text(risk.get('risk', 'Unknown'))}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 6, f"  Impact: {risk.get('impact', 'N/A')} | Likelihood: {risk.get('likelihood', 'N/A')}", ln=True)
            mitigation = risk.get("mitigation", "")
            if mitigation:
                pdf.cell(0, 6, f"  Mitigation: {_clean_text(mitigation)}", ln=True)
            pdf.ln(3)

    # ── Key Metrics ──────────────────────────────────
    metrics = report.get("key_metrics", [])
    if metrics:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Key Metrics to Track", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for m in metrics:
            pdf.cell(0, 7, f"- {_clean_text(str(m))}", ln=True)

    # ── Next Steps ───────────────────────────────────
    next_steps = recommendation.get("next_steps", [])
    if next_steps:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 10, "Recommended Next Steps", ln=True)
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        for i, step in enumerate(next_steps, 1):
            pdf.multi_cell(0, 6, f"{i}. {_clean_text(str(step))}")
            pdf.ln(2)

    # ── Footer on all pages ──────────────────────────
    # (fpdf2 doesn't have built-in footers, skip for simplicity)

    return pdf.output()


def _clean_text(text: str) -> str:
    """Remove markdown/HTML and non-latin1 chars for PDF compatibility."""
    import re
    text = re.sub(r'[*#_`>]', '', str(text))
    text = re.sub(r'<[^>]+>', '', text)
    # Replace non-latin1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0;">
        <span style="font-size:1.5rem;">🧪</span>
        <span style="font-size:1.1rem; font-weight:700; margin-left:6px;">Idea Tester</span>
        <p style="color:#94a3b8; font-size:0.75rem; margin:4px 0 0 0;">AI Strategy Simulator</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── LLM Provider Selector ─────────────────────────
    st.markdown("### 🤖 LLM Provider")

    providers = settings.available_providers()
    if not providers:
        st.error("No LLM provider configured")
        selected_provider = None
    elif len(providers) == 1:
        selected_provider = providers[0]
        st.info(f"Using **{selected_provider}**")
    else:
        selected_provider = st.selectbox(
            "Choose LLM Provider",
            providers,
            index=0,
            key="llm_provider_select",
        )

    # Store in session state for pipeline use
    if selected_provider:
        st.session_state["llm_provider"] = selected_provider.lower()

    # Show model info
    if selected_provider == "Gemini":
        st.caption(f"**Model:** `{settings.GEMINI_MODEL_NAME}`")
        st.caption(f"**Thinking:** `{settings.GEMINI_THINKING_LEVEL}`")
    elif selected_provider == "Groq":
        st.caption(f"**Model:** `{settings.GROQ_MODEL_NAME}`")
        st.caption("⚡ Ultra-fast inference")
    elif selected_provider == "Ollama":
        st.caption(f"**Model:** `{settings.OLLAMA_MODEL_NAME}`")
        st.caption("🏠 Local Inference")

    st.divider()

    st.markdown("### 🔧 Simulation Settings")

    # ── Research Depth Selector ────────────────────────
    research_mode = st.radio(
        "Research Depth",
        ["⚡ Quick", "🔬 Deep"],
        index=0,
        key="research_depth_select",
        horizontal=True,
    )
    is_deep = research_mode == "🔬 Deep"
    st.session_state["research_mode"] = "deep" if is_deep else "quick"

    if is_deep:
        st.caption("**30-40 entities** · **50-60 relationships** · Richer simulation")
    else:
        st.caption("**10-15 entities** · **15-25 relationships** · Faster results")

    default_rounds = settings.SIM_ROUNDS if is_deep else max(8, settings.SIM_ROUNDS // 2)
    sim_rounds = st.slider("Simulation Rounds", 5, 40, default_rounds, key="sidebar_rounds")




# ── Hero Section ──────────────────────────────────────────
if st.session_state.pipeline_stage == "input":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">AI-Powered Strategy Simulation</div>
        <div class="hero-title">Idea Tester</div>
        <div class="hero-subtitle">
            Test your business strategy in a simulated market before risking real capital.
            AI agents play competitors, consumers, and regulators to predict outcomes.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Input Form ────────────────────────────────────────────
if st.session_state.pipeline_stage == "input":
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💡 Your Idea")
        idea = st.text_area(
            "Describe your business strategy or idea",
            placeholder="e.g., Launch a premium drone delivery service for last-mile logistics in urban Southeast Asia, starting with Singapore and Bangkok...",
            height=150,
            key="idea_input",
        )

        st.markdown("### 🏢 Context")
        c1, c2, c3 = st.columns(3)
        with c1:
            company = st.text_input("Company Name", placeholder="e.g., Acme Corp", key="company_input")
        with c2:
            sector = st.text_input("Industry Sector", placeholder="e.g., Logistics / E-commerce", key="sector_input")
        with c3:
            geography = st.text_input("Geography", placeholder="e.g., Southeast Asia", key="geo_input")

    with col2:
        st.markdown("### 📋 Pipeline")
        steps = [
            ("🔍", "Research", "5 agents scan the web"),
            ("📦", "Package", "Build simulation world"),
            ("🎮", "Simulate", "Run agent interactions"),
            ("📊", "Report", "Generate predictions"),
        ]
        for icon, name, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                <span style="font-size:1.3rem">{icon}</span>
                <div>
                    <div style="font-weight:600">{name}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted)">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    col_btn, _, _ = st.columns([1, 1, 1])
    with col_btn:
        run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True, disabled=not idea)

    if run_btn and idea:
        st.session_state.idea = idea
        st.session_state.company = company
        st.session_state.sector = sector
        st.session_state.geography = geography
        st.session_state.pipeline_stage = "researching"
        st.rerun()


# ── Pipeline Execution ────────────────────────────────────
if st.session_state.pipeline_stage in ("researching", "packaging", "simulating", "reporting"):
    st.markdown(f"## 🧪 Testing: *{st.session_state.idea[:80]}...*")

    progress_bar = st.progress(0)
    status_text = st.empty()
    detail_container = st.container()

    try:
        # ── Stage 1: Research ────────────────────────────
        if st.session_state.pipeline_stage == "researching":
            status_text.markdown("### 🔍 Stage 1/4 — Researching the market...")
            progress_bar.progress(5)

            llm = get_llm()
            orchestrator = ResearchOrchestrator(llm=llm, tavily_api_key=settings.TAVILY_API_KEY)

            agent_statuses = {}
            agent_placeholder = detail_container.empty()

            def research_progress(agent_name, status):
                agent_statuses[agent_name] = status
                done = sum(1 for s in agent_statuses.values() if s == "done")
                total = 5
                progress_bar.progress(5 + int(done / total * 20))
                lines = []
                for name, s in agent_statuses.items():
                    icon = "✅" if s == "done" else "⏳" if s == "running" else "❌"
                    lines.append(f"{icon} **{name.title()}** Agent — {s}")
                agent_placeholder.markdown("\n\n".join(lines))

            output = orchestrator.research(
                idea=st.session_state.idea,
                company=st.session_state.company,
                sector=st.session_state.sector,
                geography=st.session_state.geography,
                progress_callback=research_progress,
            )
            st.session_state.research_output = output
            st.session_state.pipeline_stage = "packaging"
            st.rerun()

        # ── Stage 2: Package Seed ────────────────────────
        if st.session_state.pipeline_stage == "packaging":
            status_text.markdown("### 📦 Stage 2/4 — Building simulation world...")
            progress_bar.progress(30)

            llm = get_llm()
            packager = SeedPackager(llm=llm)
            research_text = st.session_state.research_output.to_full_text()

            seed = packager.package(
                idea=st.session_state.idea,
                company=st.session_state.company,
                sector=st.session_state.sector,
                geography=st.session_state.geography,
                research_text=research_text,
                research_mode=st.session_state.get("research_mode", "quick"),
            )
            st.session_state.seed_document = seed
            progress_bar.progress(40)
            st.session_state.pipeline_stage = "simulating"
            st.rerun()

        # ── Stage 3: Simulate ────────────────────────────
        if st.session_state.pipeline_stage == "simulating":
            try:
                import oasis
                oasis_mode = True
            except ImportError:
                oasis_mode = False

            engine_label = "OASIS Twitter+Reddit" if oasis_mode else "LLM-based"
            status_text.markdown(f"### 🎮 Stage 3/4 — Running {engine_label} simulation...")
            progress_bar.progress(42)

            # ── Graph Component UI ─────────────────────────
            sim_col_left, sim_col_right = detail_container.columns([1, 1.2])

            llm = get_llm()
            graph = KnowledgeGraph(llm=llm)
            graph.build_from_seed(st.session_state.seed_document)

            with sim_col_right:
                st.markdown("### 🕸️ Market Knowledge Graph")
                show_graph(graph, height=500)

            with sim_col_left:
                st.markdown(
                    f"📊 Knowledge graph: **{len(graph.entities)}** entities, "
                    f"**{len(graph.relationships)}** relationships | "
                    f"Engine: **{engine_label}**"
                )

                # Generate personas
                round_placeholder = st.empty()
                gen = PersonaGenerator(llm=llm)

            # Ollama cooldown — let previous request fully release
            if llm.provider == "ollama":
                import time as _t
                round_placeholder.markdown("⏳ Waiting for Ollama to be ready...")
                _t.sleep(5)

            def persona_progress(current, total, name):
                progress_bar.progress(42 + int(current / total * 15))
                round_placeholder.markdown(f"🧬 Generating persona {current}/{total}: **{name}**")

            personas = gen.generate_personas(
                graph=graph,
                idea_description=st.session_state.idea,
                progress_callback=persona_progress,
            )

            # Run simulation
            runner = SimulationRunner(llm=llm)

            def sim_progress(round_num, total, summary):
                pct = 57 + int(round_num / total * 28)
                progress_bar.progress(min(pct, 85))
                round_placeholder.markdown(f"🔄 {summary}")

            rounds = runner.run(
                graph=graph,
                personas=personas,
                idea_description=st.session_state.idea,
                total_rounds=sim_rounds,
                progress_callback=sim_progress,
            )

            st.session_state.sim_rounds = rounds
            st.session_state._graph = graph
            st.session_state._personas = personas
            st.session_state.pipeline_stage = "reporting"
            st.rerun()

        # ── Stage 4: Report ──────────────────────────────
        if st.session_state.pipeline_stage == "reporting":
            status_text.markdown("### 📊 Stage 4/4 — Generating prediction report...")
            progress_bar.progress(87)

            llm = get_llm()
            reporter = ReportAgent(llm=llm)

            graph = st.session_state._graph
            personas = st.session_state._personas

            def report_progress(stage, pct, msg):
                progress_bar.progress(87 + int(pct * 0.13))
                detail_container.markdown(f"📝 {msg}")

            raw_report = reporter.generate_report(
                graph=graph,
                personas=personas,
                rounds=st.session_state.sim_rounds,
                idea_description=st.session_state.idea,
                progress_callback=report_progress,
            )

            formatter = ReportFormatter(llm=llm)
            formatted = formatter.format(raw_report, st.session_state.idea)

            st.session_state.report = raw_report
            st.session_state.formatted_report = formatted
            st.session_state.pipeline_stage = "done"
            progress_bar.progress(100)
            st.rerun()

    except Exception as e:
        st.error(f"❌ Pipeline error: {str(e)}")
        st.exception(e)
        st.session_state.error = str(e)
        if st.button("🔄 Restart"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ── Results Display ───────────────────────────────────────
if st.session_state.pipeline_stage == "done" and st.session_state.formatted_report:
    report = st.session_state.formatted_report
    recommendation = report.get("recommendation", {})

    # ── Decision Header ──────────────────────────────────
    decision = recommendation.get("decision", "PENDING")
    confidence = recommendation.get("confidence_percentage", 0)

    decision_class = "decision-conditional"
    if "GO" in decision and "NO" not in decision and "CONDITIONAL" not in decision:
        decision_class = "decision-go"
    elif "NO" in decision:
        decision_class = "decision-nogo"

    st.markdown(f"""
    <div class="{decision_class}">
        {decision} &nbsp;—&nbsp; {confidence}% Confidence
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Stats Row ────────────────────────────────────────
    cols = st.columns(4)
    sim_rounds_data = st.session_state.sim_rounds or []
    total_actions = sum(len(r.actions) for r in sim_rounds_data)

    stats = [
        (f"{len(st.session_state._personas)}", "Agents Simulated"),
        (f"{len(sim_rounds_data)}", "Rounds Completed"),
        (f"{total_actions}", "Total Actions"),
        (f"{confidence}%", "Confidence"),
    ]
    for col, (val, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Executive Summary ────────────────────────────────
    st.markdown("## 📋 Executive Summary")
    st.markdown(report.get("executive_summary", ""))

    st.divider()

    # ── Report Sections ──────────────────────────────────
    for section in report.get("sections", []):
        if section.get("content"):
            with st.expander(section["title"], expanded=False):
                st.markdown(section["content"])

    # ── Risk Matrix ──────────────────────────────────────
    risk_matrix = report.get("risk_matrix", [])
    if risk_matrix:
        st.markdown("## ⚠️ Risk Matrix")
        for risk in risk_matrix:
            likelihood = risk.get("likelihood", "unknown")
            impact = risk.get("impact", "unknown")
            cls = "risk-high" if impact == "high" else "risk-medium" if impact == "medium" else "risk-low"

            st.markdown(f"""
            <div class="sim-card">
                <strong>{risk.get('risk', 'Unknown risk')}</strong><br/>
                <span class="{cls}">Impact: {impact.upper()}</span> &nbsp;|&nbsp;
                Likelihood: {likelihood} &nbsp;|&nbsp;
                <em>Source: {risk.get('source', 'N/A')}</em><br/>
                <span style="color:var(--text-secondary)">Mitigation: {risk.get('mitigation', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Key Metrics ──────────────────────────────────────
    metrics = report.get("key_metrics", [])
    if metrics:
        st.markdown("## 📈 Key Metrics to Track")
        for m in metrics:
            st.markdown(f"- {m}")

    # ── Next Steps ───────────────────────────────────────
    next_steps = recommendation.get("next_steps", [])
    if next_steps:
        st.markdown("## 🚀 Recommended Next Steps")
        for i, step in enumerate(next_steps, 1):
            st.markdown(f"**{i}.** {step}")

    st.divider()

    # ── Interactive Graph Section ────────────────────────
    st.markdown("## 🕸️ Market Knowledge Graph Exploration")
    st.caption("Click a node to inspect it. Use the checkboxes to filter entity types. Drag nodes to rearrange the layout.")
    
    if hasattr(st.session_state, "_graph"):
        show_graph(st.session_state._graph, height=700, interactive_panel=True, key_suffix="results")

    st.divider()

    # ── Action Buttons ───────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Test Another Idea", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with col2:
        pdf_bytes = _generate_pdf_report(report, recommendation, st.session_state.idea)
        st.download_button(
            "📥 Download PDF Report",
            data=pdf_bytes,
            file_name="idea_tester_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col3:
        report_json = json.dumps(report, indent=2, ensure_ascii=False)
        st.download_button(
            "📋 Download JSON",
            data=report_json,
            file_name="idea_tester_report.json",
            mime="application/json",
            use_container_width=True,
        )
