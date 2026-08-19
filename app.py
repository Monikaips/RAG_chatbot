import streamlit as st
import pandas as pd
import uuid

from io import BytesIO

from rag.pdf_loader import (
    load_resumes,
    load_single_pdf
)

from rag.candidate_matcher import evaluate_candidate
from rag.text_splitter import split_documents
from rag.vector_store import create_vector_store
from rag.chatbot import ask_chatbot

from rag.candidate_extractor import (
    group_resumes,
    extract_candidate_details
)

from rag.report_columns import (
    DEFAULT_RANKING_COLUMNS,
    PRESET_RANKING_COLUMNS,
    merge_output_columns,
    build_ranking_row,
    column_to_field,
)

from rag.openai_usage import usage_context
from rag.openai_pricing import calculate_cost, get_model_pricing
from rag.usage_store import (
    list_filter_values,
    query_usage_records,
    summarize_by_resume,
    summarize_records,
)


# ==================================================
# PRESENTATION HELPERS (UI only — no business logic)
# ==================================================

UI_STEPS = [
    (1, "Upload", "upload"),
    (2, "Configure", "columns"),
    (3, "Analyze", "play"),
    (4, "Details", "table"),
    (5, "Ranking", "trophy"),
    (6, "Ask", "message"),
    (7, "Usage", "activity"),
]

_ICON_PATHS = {
    "file-text": (
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/>'
        '<line x1="16" y1="17" x2="8" y2="17"/>'
        '<line x1="10" y1="9" x2="8" y2="9"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" y1="3" x2="12" y2="15"/>'
    ),
    "clipboard": (
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
    ),
    "columns": (
        '<path d="M12 3h7a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-7"/>'
        '<path d="M5 3h7v18H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/>'
        '<line x1="12" y1="3" x2="12" y2="21"/>'
    ),
    "play": (
        '<circle cx="12" cy="12" r="10"/>'
        '<polygon points="10 8 16 12 10 16 10 8"/>'
    ),
    "brain": (
        '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>'
        '<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>'
        '<path d="M12 5v13"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
    ),
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/>'
        '<path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
        '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<line x1="7" y1="12" x2="17" y2="12"/>'
    ),
    "trophy": (
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/>'
        '<path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
        '<path d="M4 22h16"/>'
        '<path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
        '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
        '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>'
    ),
    "download": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>'
    ),
    "pin": (
        '<path d="M12 17v5"/>'
        '<path d="M9 10.76a2 2 0 0 1-1.11 1.79l-1.78.9A2 2 0 0 0 5 15.24V16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-.76a2 2 0 0 0-1.11-1.79l-1.78-.9A2 2 0 0 1 15 10.76V7a1 1 0 0 1 1-1 2 2 0 0 0 0-4H8a2 2 0 0 0 0 4 1 1 0 0 1 1 1z"/>'
    ),
    "message": (
        '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
    ),
    "table": (
        '<path d="M12 3v18"/>'
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<path d="M3 9h18"/>'
        '<path d="M3 15h18"/>'
    ),
    "check": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
    "x": (
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/>'
    ),
    "arrow-left": (
        '<line x1="19" y1="12" x2="5" y2="12"/>'
        '<polyline points="12 19 5 12 12 5"/>'
    ),
    "activity": (
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'
    ),
}


def _svg_icon(name, size=18, stroke="currentColor"):
    paths = _ICON_PATHS.get(name, _ICON_PATHS["file-text"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" '
        f'height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" class="ats-icon" aria-hidden="true">'
        f"{paths}</svg>"
    )


def inject_ats_styles():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Afacad:wght@400;500;600;700&display=swap');

:root {
  --ats-bg: #FAFAF9;
  --ats-surface: #FFFFFF;
  --ats-border: #E7E5E4;
  --ats-text: #1C1917;
  --ats-muted: #78716C;
  --ats-primary: #2563EB;
  --ats-primary-soft: #DBEAFE;
  --ats-success: #16A34A;
  --ats-success-soft: #DCFCE7;
  --ats-warn: #D97706;
  --ats-warn-soft: #FEF3C7;
  --ats-danger: #DC2626;
  --ats-danger-soft: #FEE2E2;
  --ats-space-1: 4px;
  --ats-space-2: 8px;
  --ats-space-3: 12px;
  --ats-space-4: 16px;
  --ats-space-5: 24px;
  --ats-space-6: 32px;
  --ats-radius: 10px;
  --ats-radius-sm: 8px;
  --ats-shadow: 0 1px 3px rgba(0,0,0,0.06);
  --ats-font: "Afacad", system-ui, -apple-system, "Segoe UI", sans-serif;
}

html, body, [class*="css"], .stApp {
  font-family: var(--ats-font) !important;
  color: var(--ats-text);
}

html, body {
  height: auto !important;
  overflow-y: auto !important;
}

.stApp {
  background: var(--ats-bg);
  height: auto !important;
  min-height: 100vh;
  overflow: visible !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main,
.main .block-container {
  overflow: visible !important;
}

[data-testid="stHeader"] {
  background: transparent;
  height: 2.5rem;
}

[data-testid="stDecoration"] {
  display: none;
}

[data-testid="stToolbar"] {
  right: 0.5rem;
  top: 0.35rem;
}

[data-testid="stAppViewContainer"] > .main {
  padding-top: 0;
  overflow-y: auto !important;
}

.block-container {
  max-width: 1100px;
  padding-top: 0.75rem !important;
  padding-bottom: var(--ats-space-6) !important;
  padding-left: var(--ats-space-4) !important;
  padding-right: var(--ats-space-4) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: var(--ats-space-3);
}

.ats-topbar {
  background: var(--ats-surface);
  border: 1px solid var(--ats-border);
  border-radius: var(--ats-radius);
  box-shadow: var(--ats-shadow);
  padding: var(--ats-space-3) var(--ats-space-4);
  margin-top: 0;
  margin-bottom: var(--ats-space-3);
  position: relative;
  z-index: 1;
}

.ats-topbar-title {
  display: flex;
  align-items: flex-start;
  gap: var(--ats-space-3);
  margin-bottom: 0;
  flex-wrap: wrap;
}

.ats-topbar-title h1 {
  margin: 0;
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--ats-text);
  letter-spacing: -0.02em;
}

.ats-topbar-title p {
  margin: var(--ats-space-1) 0 0 0;
  color: var(--ats-muted);
  font-size: 0.95rem;
  font-weight: 400;
  line-height: 1.4;
}

.ats-brand-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--ats-radius-sm);
  background: var(--ats-primary-soft);
  color: var(--ats-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.ats-stepper {
  display: flex;
  gap: var(--ats-space-2);
  flex-wrap: wrap;
  margin-bottom: var(--ats-space-2);
}

.ats-step-jump {
  margin-bottom: var(--ats-space-2);
}

.ats-step-jump [data-testid="stHorizontalBlock"] {
  gap: var(--ats-space-2) !important;
  flex-wrap: wrap !important;
}

.ats-step-jump [data-testid="column"] {
  min-width: 2.5rem !important;
  flex: 0 1 auto !important;
  width: auto !important;
}

.ats-nav-bar {
  position: sticky;
  bottom: 0;
  z-index: 100;
  background: rgba(250, 250, 249, 0.96);
  border-top: 1px solid var(--ats-border);
  padding: var(--ats-space-3) 0 var(--ats-space-4) 0;
  margin-top: var(--ats-space-4);
}

.ats-nav-bar [data-testid="stHorizontalBlock"] {
  gap: var(--ats-space-3) !important;
  flex-wrap: nowrap !important;
}

.ats-nav-bar [data-testid="column"] {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  width: auto !important;
}

.ats-step-pill {
  display: inline-flex;
  align-items: center;
  gap: var(--ats-space-2);
  padding: var(--ats-space-2) var(--ats-space-3);
  border-radius: 999px;
  border: 1px solid var(--ats-border);
  background: var(--ats-surface);
  color: var(--ats-muted);
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
}

.ats-step-pill .ats-step-num {
  width: 22px;
  height: 22px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.75rem;
  font-weight: 700;
  background: #F5F5F4;
  color: var(--ats-muted);
}

.ats-step-pill.active {
  border-color: var(--ats-primary);
  background: var(--ats-primary-soft);
  color: var(--ats-primary);
}

.ats-step-pill.active .ats-step-num {
  background: var(--ats-primary);
  color: #FAFAF9;
}

.ats-step-pill.done {
  border-color: #BBF7D0;
  background: var(--ats-success-soft);
  color: var(--ats-success);
}

.ats-step-pill.done .ats-step-num {
  background: var(--ats-success);
  color: #FAFAF9;
}

[data-testid="stVerticalBlock"] {
  gap: var(--ats-space-3);
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--ats-surface);
  border: 1px solid var(--ats-border) !important;
  border-radius: var(--ats-radius) !important;
  box-shadow: var(--ats-shadow) !important;
  padding: var(--ats-space-4) !important;
  margin-bottom: var(--ats-space-3) !important;
}

.ats-screen-header {
  display: flex;
  align-items: flex-start;
  gap: var(--ats-space-3);
  margin-bottom: var(--ats-space-4);
  padding-bottom: var(--ats-space-3);
  border-bottom: 1px solid var(--ats-border);
}

.ats-screen-header .ats-section-icon {
  width: 36px;
  height: 36px;
  border-radius: var(--ats-radius-sm);
  background: var(--ats-primary-soft);
  color: var(--ats-primary);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.ats-screen-header h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--ats-text);
}

.ats-screen-header p {
  margin: 4px 0 0 0;
  color: var(--ats-muted);
  font-size: 0.92rem;
  font-weight: 400;
}

.ats-empty {
  border: 1px dashed var(--ats-border);
  border-radius: var(--ats-radius);
  padding: var(--ats-space-5);
  text-align: center;
  color: var(--ats-muted);
  background: #FAFAF9;
  line-height: 1.5;
}

.ats-progress-list {
  display: flex;
  flex-direction: column;
  gap: var(--ats-space-2);
  margin: var(--ats-space-3) 0;
}

.ats-progress-item {
  display: flex;
  align-items: center;
  gap: var(--ats-space-3);
  padding: var(--ats-space-3);
  border: 1px solid var(--ats-border);
  border-radius: var(--ats-radius-sm);
  background: #FAFAF9;
  color: var(--ats-muted);
  font-weight: 500;
}

.ats-progress-item .dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #D6D3D1;
  flex-shrink: 0;
}

.ats-progress-item.active {
  border-color: var(--ats-primary);
  background: var(--ats-primary-soft);
  color: var(--ats-primary);
}

.ats-progress-item.active .dot {
  background: var(--ats-primary);
}

.ats-progress-item.done {
  border-color: #BBF7D0;
  background: var(--ats-success-soft);
  color: var(--ats-success);
}

.ats-progress-item.done .dot {
  background: var(--ats-success);
}

.ats-eval-card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--ats-space-4);
  flex-wrap: wrap;
  margin-bottom: var(--ats-space-2);
}

.ats-score-block {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  line-height: 1;
}

.ats-score-block .num {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--ats-text);
}

.ats-score-block .lbl {
  font-size: 0.75rem;
  color: var(--ats-muted);
  font-weight: 500;
  margin-top: 4px;
}

.ats-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid transparent;
}

.ats-badge-strong {
  background: var(--ats-success-soft);
  color: var(--ats-success);
  border-color: #BBF7D0;
}

.ats-badge-possible {
  background: var(--ats-warn-soft);
  color: var(--ats-warn);
  border-color: #FDE68A;
}

.ats-badge-weak {
  background: var(--ats-danger-soft);
  color: var(--ats-danger);
  border-color: #FECACA;
}

.ats-split-label {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  margin-bottom: 6px;
}

.ats-split-label.strength { color: var(--ats-success); }
.ats-split-label.gap { color: var(--ats-danger); }

.ats-chat-source {
  display: inline-block;
  margin-top: 8px;
  padding: 4px 10px;
  border-radius: 999px;
  background: #F5F5F4;
  color: var(--ats-muted);
  font-size: 0.78rem;
  border: 1px solid var(--ats-border);
  font-weight: 500;
}

.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
  background: var(--ats-primary) !important;
  color: #FAFAF9 !important;
  border: 1px solid var(--ats-primary) !important;
  border-radius: var(--ats-radius-sm) !important;
  font-weight: 600 !important;
  min-height: 2.5rem;
  box-shadow: none !important;
  padding: 0 var(--ats-space-4) !important;
}

.stButton > button:not([kind="primary"]) {
  background: var(--ats-surface) !important;
  color: var(--ats-text) !important;
  border: 1px solid var(--ats-border) !important;
  border-radius: var(--ats-radius-sm) !important;
  font-weight: 500 !important;
  min-height: 2.5rem;
  padding: 0 var(--ats-space-3) !important;
}

[data-testid="stFileUploader"] {
  border: 1px dashed var(--ats-border);
  border-radius: var(--ats-radius);
  background: #FAFAF9;
  padding: var(--ats-space-3);
}

[data-testid="stFileUploader"] section {
  padding: 0 !important;
}

[data-testid="stForm"] {
  margin-bottom: 0 !important;
}

[data-testid="stHorizontalBlock"] {
  gap: var(--ats-space-4);
  align-items: stretch;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"] {
  min-width: 0;
}

[data-testid="stDataFrame"],
[data-testid="stDataFrameResizable"] {
  border: 1px solid var(--ats-border);
  border-radius: var(--ats-radius);
  overflow: auto;
  box-shadow: var(--ats-shadow);
  width: 100%;
}

[data-testid="stDataFrame"] table thead th,
[data-testid="stDataFrameResizable"] table thead th {
  background: #F5F5F4 !important;
  color: var(--ats-text) !important;
  font-weight: 600 !important;
  border-bottom: 1px solid var(--ats-border) !important;
  border-right: none !important;
}

[data-testid="stDataFrame"] table tbody td,
[data-testid="stDataFrameResizable"] table tbody td {
  border-bottom: none !important;
  border-right: none !important;
  padding-top: 10px !important;
  padding-bottom: 10px !important;
}

[data-testid="stDataFrame"] table tbody tr:hover td,
[data-testid="stDataFrameResizable"] table tbody tr:hover td {
  background: #F5F5F4 !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--ats-border) !important;
  border-radius: var(--ats-radius) !important;
  background: var(--ats-surface) !important;
  box-shadow: var(--ats-shadow);
  margin-bottom: var(--ats-space-3) !important;
}

[data-testid="stChatMessage"] {
  background: var(--ats-surface) !important;
  border: 1px solid var(--ats-border) !important;
  border-radius: var(--ats-radius) !important;
  margin-bottom: var(--ats-space-3) !important;
  padding: var(--ats-space-2) !important;
}

div[data-testid="stAlert"] {
  border-radius: var(--ats-radius-sm) !important;
  border-width: 1px !important;
}

.stDownloadButton > button {
  border-radius: var(--ats-radius-sm) !important;
  border: 1px solid var(--ats-border) !important;
  background: #FAFAF9 !important;
  color: var(--ats-text) !important;
  font-weight: 600 !important;
}

[data-testid="stMarkdownContainer"] p {
  line-height: 1.45;
}

.ats-upload-grid [data-testid="stHorizontalBlock"] {
  gap: var(--ats-space-3) !important;
  flex-wrap: wrap !important;
}

.ats-upload-grid [data-testid="column"] {
  min-width: min(100%, 280px) !important;
  flex: 1 1 280px !important;
}

@media (max-width: 1000px) {
  .block-container {
    padding-left: var(--ats-space-3) !important;
    padding-right: var(--ats-space-3) !important;
  }

  [data-testid="stVerticalBlockBorderWrapper"] {
    padding: var(--ats-space-3) !important;
  }

  .ats-upload-grid [data-testid="column"] {
    flex: 1 1 100% !important;
    width: 100% !important;
  }
}

@media (max-width: 600px) {
  .block-container {
    padding-top: 0.5rem !important;
    padding-left: var(--ats-space-2) !important;
    padding-right: var(--ats-space-2) !important;
    padding-bottom: var(--ats-space-5) !important;
  }

  .ats-topbar,
  [data-testid="stVerticalBlockBorderWrapper"] {
    padding: var(--ats-space-3) !important;
  }

  .ats-topbar-title h1 {
    font-size: 1.2rem;
  }

  .ats-step-pill {
    padding: 6px 10px;
    font-size: 0.8rem;
  }

  .ats-screen-header {
    margin-bottom: var(--ats-space-3);
  }

  .ats-nav-bar {
    padding: var(--ats-space-2) 0 var(--ats-space-3) 0;
  }
}

.ats-cost-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--ats-space-4);
  flex-wrap: wrap;
  background: var(--ats-primary-soft);
  border: 1px solid #BFDBFE;
  border-radius: var(--ats-radius);
  padding: var(--ats-space-4) var(--ats-space-5);
  margin: var(--ats-space-4) 0;
}

.ats-cost-label {
  font-size: 0.85rem;
  color: var(--ats-muted);
  margin-bottom: 2px;
}

.ats-cost-amount {
  font-size: 2rem;
  font-weight: 700;
  color: var(--ats-primary);
  line-height: 1.1;
}

.ats-cost-meta {
  font-size: 0.9rem;
  color: var(--ats-muted);
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] {
  background: transparent;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def format_usd(value) -> str:
    if value is None:
        return "Unknown"

    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "Unknown"

    formatted = f"{amount:.8f}".rstrip("0").rstrip(".")
    if formatted in {"", "-"}:
        formatted = "0.00"

    if "." not in formatted:
        formatted = f"{formatted}.00"

    return f"${formatted}"


def render_total_cost_banner(summary: dict):
    st.markdown(
        f"""
<div class="ats-cost-banner">
  <div>
    <div class="ats-cost-label">Total OpenAI cost for this analysis</div>
    <div class="ats-cost-amount">{format_usd(summary["total_cost"])}</div>
  </div>
  <div class="ats-cost-meta">
    {summary["total_resumes"]} resumes
    &middot; {summary["total_calls"]} API calls
    &middot; {summary["total_tokens"]:,} tokens
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_cost_calculation(records: list):
    """Show the actual token × price math used for the dollar total."""

    st.markdown("**How this dollar amount is calculated**")
    st.caption(
        "input_cost = actual_input_tokens × (model_input_price / 1,000,000). "
        "output_cost = actual_output_tokens × (model_output_price / 1,000,000). "
        "total_cost = input_cost + output_cost. "
        "Token counts come from OpenAI's usage object on each API response."
    )

    if not records:
        pricing = get_model_pricing("gpt-4o-mini")
        embed_pricing = get_model_pricing("text-embedding-3-small")
        st.warning(
            "No OpenAI usage has been saved for this run yet. "
            "Re-run Analyze so each API response can be recorded. "
            f"Current rates: gpt-4o-mini input "
            f"${pricing['input_per_million']}/1M tokens, output "
            f"${pricing['output_per_million']}/1M tokens. "
            f"text-embedding-3-small input "
            f"${embed_pricing['input_per_million']}/1M tokens."
        )
        return

    by_model = {}
    for row in records:
        model = row.get("model") or "unknown"
        bucket = by_model.setdefault(
            model,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        bucket["calls"] += 1
        if row.get("input_tokens") is not None:
            bucket["input_tokens"] += int(row["input_tokens"])
        if row.get("output_tokens") is not None:
            bucket["output_tokens"] += int(row["output_tokens"])

    rows = []
    for model, bucket in by_model.items():
        pricing = get_model_pricing(model)
        costs = calculate_cost(
            model,
            bucket["input_tokens"],
            bucket["output_tokens"],
        )
        rows.append({
            "Model": model,
            "API Calls": bucket["calls"],
            "Input tokens": bucket["input_tokens"],
            "Input price / 1M": f"${pricing['input_per_million']}",
            "Input cost": format_usd(costs["input_cost"]),
            "Output tokens": bucket["output_tokens"],
            "Output price / 1M": f"${pricing['output_per_million']}",
            "Output cost": format_usd(costs["output_cost"]),
            "Model total": format_usd(costs["total_cost"]),
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_usage_summary_metrics(summary: dict):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Resumes processed", summary["total_resumes"])
        st.metric("API calls", summary["total_calls"])

    with col2:
        st.metric("Input tokens", f"{summary['input_tokens']:,}")
        st.metric("Output tokens", f"{summary['output_tokens']:,}")

    with col3:
        st.metric("Total tokens", f"{summary['total_tokens']:,}")
        st.metric("Total OpenAI cost", format_usd(summary["total_cost"]))

    with col4:
        st.metric(
            "Avg cost per resume",
            format_usd(summary["average_cost_per_resume"]),
        )


def render_ai_cost_table(job_id=None, run_id=None, ranked_candidates=None):
    records = query_usage_records(job_id=job_id, run_id=run_id)

    if not records:
        render_cost_calculation([])
        return

    summary = summarize_records(records)
    by_resume = summarize_by_resume(records)

    st.caption(
        "These figures come from the usage object returned by each "
        "OpenAI API response. Nothing is estimated."
    )
    render_total_cost_banner(summary)
    render_cost_calculation(records)
    render_usage_summary_metrics(summary)

    name_by_source = {}
    score_by_source = {}

    for candidate in ranked_candidates or []:
        source = candidate.get("source", "Unknown")
        name_by_source[source] = candidate.get(
            "candidate_name",
            source,
        )
        score_by_source[source] = candidate.get("score", 0)

    rows = []

    for resume_id, bucket in by_resume.items():
        rows.append({
            "Candidate": name_by_source.get(resume_id, resume_id),
            "Resume": resume_id,
            "AI Score": score_by_source.get(resume_id, "—"),
            "API Calls": bucket["calls"],
            "Input Tokens": bucket["input_tokens"],
            "Output Tokens": bucket["output_tokens"],
            "Total Tokens": bucket["total_tokens"],
            "AI Cost": format_usd(bucket["total_cost"]),
        })

    if rows:
        rows.append({
            "Candidate": "TOTAL",
            "Resume": "",
            "AI Score": "",
            "API Calls": summary["total_calls"],
            "Input Tokens": summary["input_tokens"],
            "Output Tokens": summary["output_tokens"],
            "Total Tokens": summary["total_tokens"],
            "AI Cost": format_usd(summary["total_cost"]),
        })
        cost_df = pd.DataFrame(rows)
        st.dataframe(
            cost_df,
            width="stretch",
            hide_index=True,
        )


def section_header(icon_name, title, subtitle=None):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class="ats-screen-header">
  <div class="ats-section-icon">{_svg_icon(icon_name, 18)}</div>
  <div>
    <h2>{title}</h2>
    {subtitle_html}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def verdict_class(verdict):
    key = (verdict or "").lower()
    if "strong" in key:
        return "ats-badge-strong"
    if "possible" in key:
        return "ats-badge-possible"
    if "weak" in key:
        return "ats-badge-weak"
    return "ats-badge-possible"


def render_progress_tracker(current_index):
    """current_index: 0-6 in progress, 7 = all done."""
    stages = [
        "Reading resumes",
        "Preparing candidate resumes",
        "Extracting candidate information",
        "Building resume knowledge base",
        "Processing Job Description",
        "Finding relevant resume evidence",
        "Scoring and ranking candidates",
    ]
    items = []
    for i, label in enumerate(stages):
        if current_index > i:
            state = "done"
        elif current_index == i:
            state = "active"
        else:
            state = ""
        items.append(
            f'<div class="ats-progress-item {state}">'
            f'<span class="dot"></span>'
            f'<span>{i + 1}. {label}</span>'
            f"</div>"
        )
    st.markdown(
        '<div class="ats-progress-list">'
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


def max_reachable_step(uploaded_files, uploaded_jd, output_columns):
    reachable = 1

    if uploaded_files and uploaded_jd:
        reachable = 2

    if reachable >= 2 and output_columns:
        reachable = 3

    if st.session_state.candidate_details:
        reachable = max(reachable, 4)

    if st.session_state.ranked_candidates:
        reachable = max(reachable, 5)

    if st.session_state.vector_store is not None:
        reachable = max(reachable, 6)

    if (
        st.session_state.ranked_candidates
        or st.session_state.vector_store is not None
    ):
        reachable = max(reachable, 7)

    return reachable


def render_stepper(current, reachable):
    pills = []
    for num, label, _icon in UI_STEPS:
        if num == current:
            klass = "active"
        elif num < current and num <= reachable:
            klass = "done"
        else:
            klass = ""

        pills.append(
            f'<div class="ats-step-pill {klass}">'
            f'<span class="ats-step-num">{num}</span>'
            f'<span>{label}</span>'
            f"</div>"
        )

    st.markdown(
        '<div class="ats-stepper">' + "".join(pills) + "</div>",
        unsafe_allow_html=True,
    )

    jump_cols = st.columns(len(UI_STEPS))
    for idx, (num, label, _icon) in enumerate(UI_STEPS):
        with jump_cols[idx]:
            disabled = num > reachable or num == current
            if st.button(
                f"{num}",
                key=f"step_jump_{num}",
                width="stretch",
                disabled=disabled,
                help=f"Go to {label}" if not disabled else label,
            ):
                st.session_state.ui_step = num
                st.rerun()


def nav_controls(current, reachable, can_next_extra=True):
    back_col, next_col = st.columns(2)

    with back_col:
        if current > 1:
            if st.button("Back", width="stretch", key=f"nav_back_{current}"):
                st.session_state.ui_step = current - 1
                st.rerun()

    with next_col:
        next_allowed = (
            current < min(reachable, 7)
            and current < 7
            and can_next_extra
        )
        if current < 7:
            if st.button(
                "Next",
                type="primary",
                width="stretch",
                key=f"nav_next_{current}",
                disabled=not next_allowed,
            ):
                if next_allowed:
                    st.session_state.ui_step = current + 1
                    st.rerun()


def render_step_chrome(current, reachable):
    """Stepper + nav always rendered near the top of the page."""
    render_stepper(current, reachable)
    nav_controls(current, reachable)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon=".streamlit/app_icon.svg",
    layout="wide"
)

inject_ats_styles()


# ==================================================
# SESSION STATE
# ==================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "resume_count" not in st.session_state:
    st.session_state.resume_count = 0

if "candidate_details" not in st.session_state:
    st.session_state.candidate_details = []

if "grouped_resumes" not in st.session_state:
    st.session_state.grouped_resumes = {}

if "ranked_candidates" not in st.session_state:
    st.session_state.ranked_candidates = []

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "output_columns" not in st.session_state:
    st.session_state.output_columns = list(
        DEFAULT_RANKING_COLUMNS
    )

if "custom_columns" not in st.session_state:
    st.session_state.custom_columns = []

if "ui_step" not in st.session_state:
    st.session_state.ui_step = 1

if "current_job_id" not in st.session_state:
    st.session_state.current_job_id = None

if "current_run_id" not in st.session_state:
    st.session_state.current_run_id = None


# ==================================================
# TOP BAR + STEPPER
# ==================================================

st.markdown(
    f"""
<div class="ats-topbar">
  <div class="ats-topbar-title">
    <div class="ats-brand-icon">{_svg_icon("file-text", 20)}</div>
    <div>
      <h1>AI Resume Screening System</h1>
      <p>Guided screening workflow — upload, configure, analyze, review, ask.</p>
    </div>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# PERSISTENT INPUTS
# File/column widgets stay mounted every run so step
# changes do not drop uploads or selections. On the
# Upload/Configure screens they appear in-main; on
# later steps they live collapsed in the sidebar.
# ==================================================

ui_step = st.session_state.ui_step


def render_upload_fields():
    uploaded_files_local = st.file_uploader(
        "Upload Resume Documents",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_upload",
    )

    uploaded_jd_local = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx"],
        key="jd_upload",
    )

    return uploaded_files_local, uploaded_jd_local


def render_column_fields():
    with st.form(
        "add_custom_column_form",
        clear_on_submit=True,
    ):
        add_col1, add_col2 = st.columns(
            [4, 1],
            vertical_alignment="bottom",
        )

        with add_col1:
            new_custom_column = st.text_input(
                "Add a custom column",
                placeholder=(
                    "e.g. Certifications, Notice Period, "
                    "Clearance Level"
                ),
                help=(
                    "Type a column name and press Enter "
                    "or click Add. It will appear in the "
                    "column list below."
                ),
            )

        with add_col2:
            add_clicked = st.form_submit_button(
                "Add",
                width="stretch",
            )

    if add_clicked:
        name = (new_custom_column or "").strip()

        if not name:
            st.warning("Enter a column name before adding.")
        elif name in PRESET_RANKING_COLUMNS:
            st.info(
                f'"{name}" is already a preset column. '
                "Select it from the list below."
            )
        elif name in st.session_state.custom_columns:
            st.info(f'"{name}" is already added.')
        else:
            st.session_state.custom_columns.append(name)

            current_selection = list(
                st.session_state.get(
                    "column_multiselect",
                    DEFAULT_RANKING_COLUMNS,
                )
            )

            if name not in current_selection:
                current_selection.append(name)

            st.session_state.column_multiselect = (
                current_selection
            )
            st.rerun()

    if st.session_state.custom_columns:
        st.write("**Your custom columns:**")

        chips = st.columns(
            min(len(st.session_state.custom_columns), 4)
        )

        removed_name = None

        for index, custom_name in enumerate(
            st.session_state.custom_columns
        ):
            with chips[index % len(chips)]:
                if st.button(
                    f"Remove {custom_name}",
                    key=f"remove_custom_col_{index}",
                    help=f'Remove "{custom_name}"',
                ):
                    removed_name = custom_name

        if removed_name:
            st.session_state.custom_columns = [
                item
                for item in st.session_state.custom_columns
                if item != removed_name
            ]

            current_selection = list(
                st.session_state.get("column_multiselect", [])
            )

            st.session_state.column_multiselect = [
                item
                for item in current_selection
                if item != removed_name
            ]
            st.rerun()

    all_column_options = list(PRESET_RANKING_COLUMNS)

    for custom_name in st.session_state.custom_columns:
        if custom_name not in all_column_options:
            all_column_options.append(custom_name)

    if "column_multiselect" not in st.session_state:
        st.session_state.column_multiselect = list(
            DEFAULT_RANKING_COLUMNS
        )

    st.session_state.column_multiselect = [
        item
        for item in st.session_state.column_multiselect
        if item in all_column_options
    ]

    selected_presets = st.multiselect(
        "Select columns",
        options=all_column_options,
        key="column_multiselect",
        help=(
            "Pick from common report fields and any "
            "custom columns you added above."
        ),
    )

    output_columns_local = merge_output_columns(
        selected_presets,
        "",
    )

    if not output_columns_local:
        st.warning(
            "Select at least one ranking column "
            "before analyzing candidates."
        )
    else:
        st.success(
            "Columns for next analysis: "
            + " | ".join(output_columns_local)
        )

        if (
            st.session_state.ranked_candidates
            and st.session_state.get("output_columns")
            and st.session_state.output_columns
            != output_columns_local
        ):
            st.info(
                "Column selection changed. "
                "Re-run Analyze Candidates to refresh "
                "the ranking report with the new columns."
            )

    return output_columns_local


# Mount upload + column widgets once per run
if ui_step == 1:
    uploaded_files = None
    uploaded_jd = None
    # rendered inside step screen below
elif ui_step == 2:
    uploaded_files = None
    uploaded_jd = None
else:
    with st.sidebar:
        st.markdown("**Session inputs**")
        st.caption("Kept available so analysis can run.")
        uploaded_files, uploaded_jd = render_upload_fields()
        st.divider()
        st.markdown("**Report columns**")
        output_columns = render_column_fields()


# ==================================================
# STEP 1 — UPLOAD
# ==================================================

if ui_step == 1:

    chrome_slot = st.empty()

    with st.container(border=True):
        section_header(
            "upload",
            "Upload documents",
            "Add candidate resumes and one job description to continue.",
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                f"**{_svg_icon('upload', 16)} Resumes**",
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Upload Resume Documents",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="resume_upload",
            )

        with right:
            st.markdown(
                f"**{_svg_icon('clipboard', 16)} Job Description**",
                unsafe_allow_html=True,
            )
            uploaded_jd = st.file_uploader(
                "Upload Job Description",
                type=["pdf", "docx"],
                key="jd_upload",
            )

        if not uploaded_files and not uploaded_jd:
            st.markdown(
                '<div class="ats-empty">No files yet — upload '
                "resumes and a JD to unlock the next step.</div>",
                unsafe_allow_html=True,
            )
        elif uploaded_files and not uploaded_jd:
            st.info("Please upload the Job Description before continuing.")
        elif uploaded_jd and not uploaded_files:
            st.info("Please upload at least one resume before continuing.")
        else:
            st.success(
                f"Ready — {len(uploaded_files)} resume(s) and "
                "1 job description uploaded."
            )

    # Column fields must also stay mounted on step 1 (sidebar)
    with st.sidebar:
        st.markdown("**Report columns**")
        st.caption("Editable now or on the Configure step.")
        output_columns = render_column_fields()

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    with chrome_slot.container():
        render_step_chrome(ui_step, reachable)


# ==================================================
# STEP 2 — CONFIGURE
# ==================================================

elif ui_step == 2:

    chrome_slot = st.empty()

    with st.sidebar:
        st.markdown("**Session files**")
        uploaded_files, uploaded_jd = render_upload_fields()

    with st.container(border=True):
        section_header(
            "columns",
            "Configure ranking columns",
            "Choose which fields appear in the ranking report. "
            "Change this per JD without editing code.",
        )

        output_columns = render_column_fields()

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    with chrome_slot.container():
        render_step_chrome(ui_step, reachable)


# ==================================================
# STEP 3 — ANALYZE
# ==================================================

elif ui_step == 3:

    # uploads + columns already rendered in sidebar branch above
    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    render_step_chrome(ui_step, reachable)

    with st.container(border=True):
        section_header(
            "play",
            "Analyze candidates",
            "Run screening against the uploaded JD. Progress appears below.",
        )

        if uploaded_files and uploaded_jd:
            st.success(
                f"Ready to analyze {len(uploaded_files)} resume(s) "
                "against the uploaded Job Description."
            )
        elif uploaded_files and not uploaded_jd:
            st.info(
                "Please upload the Job Description "
                "before starting the analysis."
            )
        elif uploaded_jd and not uploaded_files:
            st.info(
                "Please upload at least one resume "
                "before starting the analysis."
            )
        else:
            st.warning("Go back to Upload and add resumes + JD.")

        tracker_slot = st.empty()

        if st.session_state.ranked_candidates:
            st.success(
                f"Analysis on file — "
                f"{len(st.session_state.ranked_candidates)} "
                "candidate(s) ranked. Continue to Details, or re-run."
            )
            with tracker_slot.container():
                render_progress_tracker(7)

        if uploaded_files and uploaded_jd and output_columns:

            if st.button(
                "Analyze Candidates",
                type="primary",
                width="stretch",
            ):
                try:
                    # ------------------------------------------
                    # CLEAR PREVIOUS RESULTS
                    # ------------------------------------------

                    st.session_state.vector_store = None
                    st.session_state.resume_count = 0
                    st.session_state.candidate_details = []
                    st.session_state.grouped_resumes = {}
                    st.session_state.ranked_candidates = []
                    st.session_state.jd_text = ""
                    st.session_state.output_columns = list(
                        output_columns
                    )

                    job_id = getattr(
                        uploaded_jd,
                        "name",
                        "unknown-jd",
                    )
                    run_id = str(uuid.uuid4())
                    st.session_state.current_job_id = job_id
                    st.session_state.current_run_id = run_id

                    # ------------------------------------------
                    # PROGRESS INDICATOR
                    # ------------------------------------------

                    with tracker_slot.container():
                        render_progress_tracker(0)

                    # ==================================================
                    # STEP 1 — PROCESS RESUMES
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(0)

                    documents = load_resumes(
                        uploaded_files
                    )

                    if not documents:
                        st.error(
                            "No text could be extracted "
                            "from the uploaded resumes."
                        )
                        st.stop()

                    # ==================================================
                    # STEP 2 — GROUP COMPLETE RESUMES
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(1)

                    grouped_resumes = group_resumes(
                        documents
                    )

                    if not grouped_resumes:
                        st.error(
                            "Could not prepare the uploaded resumes."
                        )
                        st.stop()

                    st.session_state.grouped_resumes = (
                        grouped_resumes
                    )

                    # ==================================================
                    # STEP 3 — EXTRACT CANDIDATE DETAILS
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(2)

                    candidate_details = []

                    for source, resume_text in (
                        grouped_resumes.items()
                    ):
                        try:
                            with usage_context(
                                operation="resume_extraction",
                                resume_id=source,
                                job_id=job_id,
                                run_id=run_id,
                            ):
                                candidate = extract_candidate_details(
                                    resume_text
                                )

                            if not isinstance(
                                candidate,
                                dict
                            ):
                                candidate = {
                                    "candidate_name":
                                        "Not Mentioned",
                                    "position":
                                        "Not Mentioned",
                                    "skills": [],
                                    "email":
                                        "Not Mentioned",
                                    "mobile":
                                        "Not Mentioned",
                                    "location":
                                        "Not Mentioned",
                                    "visa_category":
                                        "Not Mentioned",
                                    "experience":
                                        "Not Mentioned"
                                }

                            candidate["source"] = source
                            candidate_details.append(
                                candidate
                            )

                        except Exception as candidate_error:
                            st.warning(
                                f"Could not fully extract "
                                f"details from {source}: "
                                f"{candidate_error}"
                            )

                            candidate_details.append({
                                "candidate_name":
                                    "Extraction Failed",
                                "position":
                                    "Not Mentioned",
                                "skills": [],
                                "email":
                                    "Not Mentioned",
                                "mobile":
                                    "Not Mentioned",
                                "location":
                                    "Not Mentioned",
                                "visa_category":
                                    "Not Mentioned",
                                "experience":
                                    "Not Mentioned",
                                "source":
                                    source
                            })

                    st.session_state.candidate_details = (
                        candidate_details
                    )

                    # ==================================================
                    # STEP 4 — CREATE RESUME RAG KNOWLEDGE BASE
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(3)

                    chunks = split_documents(
                        documents
                    )

                    if not chunks:
                        st.error(
                            "No resume chunks were created."
                        )
                        st.stop()

                    with usage_context(
                        operation="resume_embedding",
                        job_id=job_id,
                        run_id=run_id,
                    ):
                        vector_store = create_vector_store(
                            chunks
                        )

                    st.session_state.vector_store = (
                        vector_store
                    )
                    st.session_state.resume_count = (
                        len(uploaded_files)
                    )

                    # ==================================================
                    # STEP 5 — PROCESS JOB DESCRIPTION
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(4)

                    jd_text = load_single_pdf(
                        uploaded_jd
                    )

                    if not jd_text:
                        st.error(
                            "Could not extract text "
                            "from the Job Description."
                        )
                        st.stop()

                    st.session_state.jd_text = jd_text

                    # ==================================================
                    # STEP 6 — JD → RAG → RETRIEVE EVIDENCE
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(5)

                    retrieval_k = min(
                        len(uploaded_files) * 5,
                        100
                    )

                    results = (
                        vector_store
                        .similarity_search(
                            jd_text,
                            k=retrieval_k
                        )
                    )

                    candidate_chunks = {}

                    for result in results:
                        source = result.metadata.get(
                            "source",
                            "Unknown Resume"
                        )

                        if source not in candidate_chunks:
                            candidate_chunks[source] = []

                        candidate_chunks[source].append(
                            result.page_content
                        )

                    # ==================================================
                    # STEP 7 — EVALUATE AND RANK CANDIDATES
                    # ==================================================

                    with tracker_slot.container():
                        render_progress_tracker(6)

                    ranked_candidates = []

                    for candidate in candidate_details:
                        source = candidate.get(
                            "source",
                            "Unknown Resume"
                        )

                        full_resume = (
                            grouped_resumes.get(
                                source,
                                ""
                            )
                        )

                        if not full_resume:
                            continue

                        retrieved_evidence = (
                            candidate_chunks.get(
                                source,
                                []
                            )
                        )

                        if retrieved_evidence:
                            evidence_text = (
                                "\n\n".join(
                                    retrieved_evidence
                                )
                            )

                            candidate_text = (
                                "FULL RESUME:\n"
                                + full_resume
                                + "\n\n"
                                + "RAG RETRIEVED "
                                "RELEVANT EVIDENCE:\n"
                                + evidence_text
                            )
                        else:
                            candidate_text = full_resume

                        with usage_context(
                            operation="resume_matching",
                            resume_id=source,
                            job_id=job_id,
                            run_id=run_id,
                        ):
                            evaluation = evaluate_candidate(
                                jd_text,
                                candidate_text,
                                output_columns=output_columns,
                            )

                        merged = {
                            **candidate,
                            **evaluation,
                        }

                        for field in (
                            "candidate_name",
                            "email",
                            "mobile",
                            "location",
                            "experience",
                            "visa_category",
                            "position",
                        ):
                            extracted = candidate.get(field)
                            evaluated = evaluation.get(field)

                            if (
                                extracted
                                and extracted != "Not Mentioned"
                                and (
                                    not evaluated
                                    or evaluated == "Not Mentioned"
                                )
                            ):
                                merged[field] = extracted

                        ranked_candidates.append(merged)

                    ranked_candidates.sort(
                        key=lambda x: x.get(
                            "score",
                            0
                        ),
                        reverse=True
                    )

                    st.session_state.ranked_candidates = (
                        ranked_candidates
                    )

                    with tracker_slot.container():
                        render_progress_tracker(7)

                    st.success(
                        f"Analysis completed! "
                        f"{len(ranked_candidates)} "
                        "candidate(s) ranked."
                    )

                    st.session_state.ui_step = 4
                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Error during candidate analysis: {e}"
                    )


# ==================================================
# STEP 4 — CANDIDATE DETAILS REPORT
# ==================================================

elif ui_step == 4:

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    render_step_chrome(ui_step, reachable)

    if st.session_state.candidate_details:

        with st.container(border=True):
            section_header(
                "table",
                "Extracted Candidate Details",
                "Parsed profile fields from uploaded resumes",
            )

            display_data = []

            for candidate in (
                st.session_state.candidate_details
            ):

                row = candidate.copy()

                if isinstance(
                    row.get("skills"),
                    list
                ):

                    row["skills"] = ", ".join(
                        str(skill)
                        for skill in row["skills"]
                    )

                row.setdefault(
                    "candidate_name",
                    "Not Mentioned"
                )

                row.setdefault(
                    "position",
                    "Not Mentioned"
                )

                row.setdefault(
                    "skills",
                    "Not Mentioned"
                )

                row.setdefault(
                    "email",
                    "Not Mentioned"
                )

                row.setdefault(
                    "mobile",
                    "Not Mentioned"
                )

                row.setdefault(
                    "location",
                    "Not Mentioned"
                )

                row.setdefault(
                    "visa_category",
                    "Not Mentioned"
                )

                row.setdefault(
                    "experience",
                    "Not Mentioned"
                )

                row.setdefault(
                    "source",
                    "Unknown"
                )

                display_data.append(row)

            df = pd.DataFrame(
                display_data
            )

            df = df.rename(
                columns={

                "candidate_name":
                    "Candidate Name",

                "position":
                    "Position I Looked For",

                "skills":
                    "Skills",

                "email":
                    "Email ID",

                "mobile":
                    "Mobile Number",

                "location":
                    "Location",

                "visa_category":
                    "Visa Category",

                "experience":
                    "Experience",

                "source":
                    "Resume File"
                }
            )

            required_columns = [

            "Candidate Name",

            "Position I Looked For",

            "Skills",

            "Email ID",

            "Mobile Number",

            "Location",

            "Visa Category",

            "Experience",

            "Resume File"
            ]

            for column in required_columns:

                if column not in df.columns:

                    df[column] = "Not Mentioned"

            df = df[
                required_columns
            ]

            st.dataframe(
                df,
                width="stretch",
                hide_index=True
            )

            st.caption(
                f"Total candidates extracted: {len(df)}"
            )

        # ==================================================
        # DOWNLOAD CANDIDATE DETAILS
        # ==================================================

            excel_buffer = BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Candidate Details"
                )

                worksheet = writer.sheets[
                    "Candidate Details"
                ]

                worksheet.freeze_panes = "A2"

                worksheet.auto_filter.ref = (
                    worksheet.dimensions
                )

                column_widths = {

                "A": 25,
                "B": 30,
                "C": 60,
                "D": 35,
                "E": 22,
                "F": 30,
                "G": 22,
                "H": 20,
                "I": 35
                }

                for column, width in (
                    column_widths.items()
                ):

                    worksheet.column_dimensions[
                        column
                    ].width = width

            st.download_button(

                label=(
                    "Download Candidate Details Excel"
                ),

                data=excel_buffer.getvalue(),

                file_name=(
                    "candidate_details.xlsx"
                ),

                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),

                width="stretch"
            )

    else:
        st.warning("No candidate details yet — run Analyze first.")


# ==================================================
# STEP 5 — RANKED RESULTS + EVALUATION
# ==================================================

elif ui_step == 5:

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    render_step_chrome(ui_step, reachable)

    if st.session_state.ranked_candidates:

        with st.container(border=True):
            section_header(
                "trophy",
                "Candidates Ranked by JD Relevance",
                "Scored and sorted against the uploaded job description",
            )

            active_columns = (
                st.session_state.get("output_columns")
                or DEFAULT_RANKING_COLUMNS
            )

            ranking_data = []

            for rank, candidate in enumerate(

            st.session_state.ranked_candidates,

            start=1
            ):

                ranking_data.append(
                    build_ranking_row(
                        candidate,
                        active_columns,
                        rank=rank,
                    )
                )

            ranking_df = pd.DataFrame(
                ranking_data
            )

            st.dataframe(
                ranking_df,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Report columns: "
                + " | ".join(active_columns)
            )

            ranking_usage = query_usage_records(
                job_id=st.session_state.get("current_job_id"),
                run_id=st.session_state.get("current_run_id"),
            )
            if ranking_usage:
                render_total_cost_banner(
                    summarize_records(ranking_usage)
                )
            render_cost_calculation(ranking_usage)

        # ==================================================
        # DOWNLOAD RANKING REPORT
        # ==================================================

            ranking_excel_buffer = BytesIO()

            with pd.ExcelWriter(
                ranking_excel_buffer,
                engine="openpyxl"
            ) as writer:

                ranking_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Candidate Ranking"
                )

                worksheet = writer.sheets[
                    "Candidate Ranking"
                ]

                worksheet.freeze_panes = "A2"

                worksheet.auto_filter.ref = (
                    worksheet.dimensions
                )

                for index, column in enumerate(
                    ranking_df.columns,
                    start=1
                ):

                    letter = worksheet.cell(
                        row=1,
                        column=index
                    ).column_letter

                    header = str(column).lower()

                    if "strength" in header or "gap" in header:
                        width = 45
                    elif "skill" in header or "reason" in header:
                        width = 40
                    elif "name" in header:
                        width = 28
                    elif "email" in header:
                        width = 32
                    elif "score" in header or "rank" in header:
                        width = 14
                    else:
                        width = 22

                    worksheet.column_dimensions[
                        letter
                    ].width = width

            st.download_button(

                label=(
                    "Download Candidate Ranking Report"
                ),

                data=ranking_excel_buffer.getvalue(),

                file_name=(
                    "candidate_ranking_report.xlsx"
                ),

                mime=(
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),

                width="stretch"
            )

        with st.container(border=True):
            section_header(
                "pin",
                "Candidate Evaluation",
                "Open a card for strengths, gaps, skills, and rationale",
            )

            for rank, candidate in enumerate(
                st.session_state.ranked_candidates,
                start=1
            ):

                name = candidate.get(
                    "candidate_name",
                    "Unknown Candidate"
                )

                score = candidate.get(
                    "score",
                    0
                )

                verdict = candidate.get(
                    "verdict",
                    ""
                )

                title = f"#{rank} {name}"

                v_class = verdict_class(verdict)
                verdict_html = (
                    f'<span class="ats-badge {v_class}">{verdict}</span>'
                    if verdict else ""
                )

                st.markdown(
                    f"""
<div class="ats-eval-card-head">
  <div>{verdict_html}</div>
  <div class="ats-score-block">
    <div class="num">{score}</div>
    <div class="lbl">Match /100</div>
  </div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(title):
                    key_strengths = candidate.get(
                        "key_strengths",
                        []
                    )

                    key_gaps = candidate.get(
                        "key_gaps",
                        []
                    )

                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown(
                            '<div class="ats-split-label strength">'
                            "Key Strengths</div>",
                            unsafe_allow_html=True,
                        )
                        if key_strengths:
                            if isinstance(key_strengths, list):
                                for item in key_strengths:
                                    st.write(f"- {item}")
                            else:
                                st.write(key_strengths)
                        else:
                            st.caption("None listed")

                    with col_b:
                        st.markdown(
                            '<div class="ats-split-label gap">'
                            "Key Gaps</div>",
                            unsafe_allow_html=True,
                        )
                        if key_gaps:
                            if isinstance(key_gaps, list):
                                for item in key_gaps:
                                    st.write(f"- {item}")
                            else:
                                st.write(key_gaps)
                        else:
                            st.caption("None listed")

                    st.write(
                        "**Matched Skills:**"
                    )

                    matched_skills = candidate.get(
                        "matched_skills",
                        []
                    )

                    if matched_skills:
                        st.write(
                            ", ".join(
                                str(skill)
                                for skill in matched_skills
                            )
                        )
                    else:
                        st.write(
                            "No matched skills identified."
                        )

                    st.write(
                        "**Missing Skills:**"
                    )

                    missing_skills = candidate.get(
                        "missing_skills",
                        []
                    )

                    if missing_skills:
                        st.write(
                            ", ".join(
                                str(skill)
                                for skill in missing_skills
                            )
                        )
                    else:
                        st.write(
                            "No important missing skills identified."
                        )

                    st.write(
                        "**Why this candidate received "
                        f"{score}/100:**"
                    )

                    st.write(
                        candidate.get(
                            "reason",
                            "No explanation available."
                        )
                    )

                    known_fields = {
                        "candidate_name",
                        "score",
                        "verdict",
                        "key_strengths",
                        "key_gaps",
                        "matched_skills",
                        "missing_skills",
                        "reason",
                        "source",
                        "skills",
                        "email",
                        "mobile",
                        "location",
                        "experience",
                        "visa_category",
                        "position",
                        "experience_assessment",
                        "seniority_assessment",
                        "domain_assessment",
                        "education_assessment",
                    }

                    extra_shown = False

                    for column in active_columns:
                        field = column_to_field(column)

                        if field in known_fields:
                            continue

                        value = candidate.get(field)

                        if value in (None, "", "Not Mentioned", []):
                            continue

                        if not extra_shown:
                            st.write("**Additional report fields:**")
                            extra_shown = True

                        if isinstance(value, list):
                            st.write(
                                f"**{column}:** "
                                + ", ".join(str(v) for v in value)
                            )
                        else:
                            st.write(f"**{column}:** {value}")

        with st.container(border=True):
            section_header(
                "activity",
                "AI Usage and Cost",
                "Actual OpenAI token usage and cost for this analysis",
            )
            render_ai_cost_table(
                job_id=st.session_state.get("current_job_id"),
                run_id=st.session_state.get("current_run_id"),
                ranked_candidates=st.session_state.ranked_candidates,
            )

    else:
        st.warning("No ranking results yet — run Analyze first.")


# ==================================================
# STEP 6 — RAG CHAT
# ==================================================

elif ui_step == 6:

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    render_step_chrome(ui_step, reachable)

    with st.container(border=True):
        section_header(
            "message",
            "Ask About Candidates",
            f"{st.session_state.resume_count} resume(s) currently indexed.",
        )

        if st.session_state.vector_store is not None:

            question = st.chat_input(
                "Example: Who has AWS experience?"
            )

            if question:

                with st.chat_message("user"):

                    st.write(question)

                try:
                    with usage_context(
                        operation="resume_chat",
                        job_id=st.session_state.get("current_job_id"),
                        run_id=st.session_state.get("current_run_id"),
                    ):
                        results = (
                            st.session_state
                            .vector_store
                            .similarity_search(
                                question,
                                k=5
                            )
                        )

                        with st.chat_message("assistant"):

                            with st.spinner(
                                "Searching resumes..."
                            ):

                                answer = ask_chatbot(
                                    question,
                                    results
                                )

                            st.write(answer)

                            sources = []

                            for result in results:

                                source = result.metadata.get(
                                    "source",
                                    "Unknown"
                                )

                                if source not in sources:

                                    sources.append(
                                        source
                                    )

                            if sources:

                                st.markdown(
                                    '<div class="ats-chat-source">'
                                    "Sources: "
                                    + ", ".join(sources)
                                    + "</div>",
                                    unsafe_allow_html=True,
                                )

                except Exception as e:

                    st.error(
                        f"Error while searching "
                        f"resumes: {e}"
                    )

        else:
            st.warning(
                "No indexed resumes yet — run Analyze first."
            )


# ==================================================
# STEP 7 — AI USAGE DASHBOARD
# ==================================================

elif ui_step == 7:

    reachable = max_reachable_step(
        uploaded_files, uploaded_jd, output_columns
    )
    render_step_chrome(ui_step, reachable)

    with st.container(border=True):
        section_header(
            "activity",
            "AI Usage",
            "Actual OpenAI token usage and cost. Nothing here is estimated.",
        )

        period_label = st.radio(
            "Period",
            ["Today", "This Week", "This Month", "All Time"],
            horizontal=True,
            key="usage_period",
        )
        period_map = {
            "Today": "today",
            "This Week": "week",
            "This Month": "month",
            "All Time": None,
        }

        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

        job_options = [""] + list_filter_values("job_id")
        resume_options = [""] + list_filter_values("resume_id")
        user_options = [""] + list_filter_values("user_id")
        model_options = [""] + list_filter_values("model")

        with filter_col1:
            selected_job = st.selectbox(
                "Job / JD",
                job_options,
                key="usage_job_filter",
            )
        with filter_col2:
            selected_resume = st.selectbox(
                "Resume",
                resume_options,
                key="usage_resume_filter",
            )
        with filter_col3:
            selected_user = st.selectbox(
                "User / account",
                user_options,
                key="usage_user_filter",
            )
        with filter_col4:
            selected_model = st.selectbox(
                "Model",
                model_options,
                key="usage_model_filter",
            )

        records = query_usage_records(
            job_id=selected_job or None,
            resume_id=selected_resume or None,
            user_id=selected_user or None,
            model=selected_model or None,
            period=period_map[period_label],
        )
        summary = summarize_records(records)

        render_total_cost_banner(summary)
        render_cost_calculation(records)
        render_usage_summary_metrics(summary)

        if records:
            detail_rows = []
            for row in records:
                detail_rows.append({
                    "Time": row.get("created_at"),
                    "Operation": row.get("operation"),
                    "Job / JD": row.get("job_id"),
                    "Resume": row.get("resume_id"),
                    "Model": row.get("model"),
                    "Input Tokens": row.get("input_tokens"),
                    "Output Tokens": row.get("output_tokens"),
                    "Total Tokens": row.get("total_tokens"),
                    "Cost": format_usd(row.get("total_cost")),
                })

            st.dataframe(
                pd.DataFrame(detail_rows),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No OpenAI usage records match these filters.")

