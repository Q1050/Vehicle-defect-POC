import html
from io import BytesIO
import os
from uuid import uuid4

from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from modules.attachments import (
    build_analysis_attachment,
    build_attachment_payload,
    get_attachment_kind,
    get_message_attachments,
)
from modules.chat_storage import load_chat_state, save_chat_state
from modules.media_analysis import AttachmentAnalysisService
from ultralytics import YOLO

load_dotenv()

st.set_page_config(
    page_title="Vehicle Defect Detection",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --google-blue: #4285F4;
            --google-red: #EA4335;
            --google-yellow: #FBBC05;
            --google-green: #34A853;
            --sidebar-width: 18rem;
            --chat-max-width: 980px;
            --chat-dock-bg: #0f1724;
            --google-text: #edf2ff;
            --google-muted: #8ea1bc;
            --google-surface: #111a28;
            --google-bg: #08111d;
            --google-border: #22324f;
            --hero-title-color: #f7f9ff;
            --hero-subtitle-color: #a7b6cb;
            --hero-card-bg: linear-gradient(135deg, rgba(66, 133, 244, 0.18) 0%, rgba(52, 168, 83, 0.14) 42%, rgba(251, 188, 5, 0.14) 72%, rgba(234, 67, 53, 0.16) 100%);
            --hero-card-border: rgba(138, 180, 248, 0.28);
            --hero-title-shadow: none;
            --summary-bg: rgba(66, 133, 244, 0.1);
            --summary-border: rgba(138, 180, 248, 0.24);
            --summary-text: #d7e3f7;
            --result-accent: #4285F4;
            --button-bg: linear-gradient(90deg, #4285F4 0%, #34A853 100%);
            --button-bg-hover: linear-gradient(90deg, #5a95f5 0%, #41bb60 100%);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(66, 133, 244, 0.2) 0%, rgba(66, 133, 244, 0) 26%),
                radial-gradient(circle at top right, rgba(52, 168, 83, 0.14) 0%, rgba(52, 168, 83, 0) 24%),
                radial-gradient(circle at bottom right, rgba(251, 188, 5, 0.1) 0%, rgba(251, 188, 5, 0) 22%),
                linear-gradient(180deg, #07111d 0%, #091423 100%);
        }

        .main .block-container {
            max-width: 980px;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding-top: 2rem;
            padding-bottom: 1.25rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1220 0%, #0d1627 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        section[data-testid="stSidebar"] * {
            color: #ebf2ff;
        }

        .sidebar-title {
            font-size: 1.32rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }

        .sidebar-subtitle {
            font-size: 0.75rem;
            color: #7f92ad;
            line-height: 1.5;
            margin-bottom: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .sidebar-section-label {
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #6f83a1;
            margin: 1.15rem 0 0.65rem;
        }

        .sidebar-chat-meta {
            font-size: 0.78rem;
            color: #91a2ba;
            margin: -0.2rem 0 0.65rem;
            line-height: 1.45;
        }

        .chat-shell {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            margin-bottom: 1.25rem;
        }

        .st-key-chat_workspace_panel {
            display: flex;
            flex: 1;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
        }

        .st-key-chat_messages_panel {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            padding-right: 0.35rem;
            padding-bottom: 20rem;
            scrollbar-width: thin;
            scrollbar-color: rgba(66, 133, 244, 0.35) transparent;
        }

        .st-key-chat_messages_panel::-webkit-scrollbar {
            width: 8px;
        }

        .st-key-chat_messages_panel::-webkit-scrollbar-thumb {
            background: rgba(66, 133, 244, 0.28);
            border-radius: 999px;
        }

        .st-key-chat_sticky_panel {
            position: fixed;
            bottom: 1rem;
            left: calc(var(--sidebar-width) + max((100vw - var(--sidebar-width) - var(--chat-max-width)) / 2, 0px));
            width: min(var(--chat-max-width), calc(100vw - var(--sidebar-width) - 2rem));
            z-index: 40;
            flex-shrink: 0;
            background: var(--chat-dock-bg);
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            box-shadow: 0 -10px 28px rgba(10, 16, 22, 0.22);
            padding: 0.5rem 0.75rem 0.75rem;
            border-radius: 22px 22px 0 0;
        }

        @media (max-width: 1024px) {
            .st-key-chat_sticky_panel {
                left: 1rem;
                width: calc(100vw - 2rem);
            }
        }

        .chat-row {
            display: flex;
            width: 100%;
        }

        .chat-row.user {
            justify-content: flex-end;
        }

        .chat-row.assistant {
            justify-content: flex-start;
        }

        .chat-bubble {
            max-width: 78%;
            border-radius: 22px;
            padding: 0.95rem 1.1rem;
            border: 1px solid var(--google-border);
            box-shadow: 0 10px 24px rgba(5, 12, 20, 0.22);
        }

        .chat-row.user .chat-bubble {
            background: linear-gradient(135deg, rgba(66, 133, 244, 0.28) 0%, rgba(52, 168, 83, 0.28) 100%);
            border-color: rgba(138, 180, 248, 0.28);
        }

        .chat-row.assistant .chat-bubble {
            background: var(--google-surface);
        }

        .chat-role {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--google-blue);
            margin-bottom: 0.35rem;
        }

        .chat-text {
            color: var(--google-text);
            line-height: 1.5;
            font-size: 0.98rem;
        }

        .chat-stage {
            background: var(--hero-card-bg);
            border: 1px solid var(--hero-card-border);
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 18px 38px rgba(4, 10, 18, 0.26);
            margin-bottom: 1rem;
        }

        .hero-dots {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.8rem;
        }

        .hero-dot {
            width: 14px;
            height: 14px;
            border-radius: 999px;
            display: inline-block;
        }

        .section-label {
            color: var(--google-blue);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .hero-title {
            color: var(--hero-title-color);
            text-shadow: var(--hero-title-shadow);
            font-size: 2.45rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 0.75rem 0;
        }

        .hero-subtitle {
            color: var(--hero-subtitle-color);
            font-size: 1rem;
            margin: 0;
        }

        div[data-testid="stFileUploader"] {
            background: var(--google-surface);
            border: 1px solid var(--google-border);
            border-radius: 18px;
            padding: 0.75rem;
            box-shadow: 0 8px 18px rgba(40, 52, 64, 0.04);
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: none;
            background: transparent;
            padding: 0;
        }

        div[data-testid="stFileUploaderDropzoneInstructions"] {
            display: none;
        }

        div[data-testid="stButton"] > button {
            background: var(--button-bg);
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            color: #FFFFFF;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            box-shadow: 0 8px 20px rgba(66, 133, 244, 0.2);
        }

        div[data-testid="stButton"] > button:hover {
            background: var(--button-bg-hover);
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            box-shadow: none;
            color: #edf3ff;
            justify-content: flex-start;
            min-height: 3.25rem;
            padding: 0.8rem 0.95rem;
            text-align: left;
            white-space: pre-line;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: rgba(66, 133, 244, 0.18);
            border-color: rgba(138, 180, 248, 0.28);
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
            border-color: rgba(138, 180, 248, 0.2);
            box-shadow: 0 12px 28px rgba(66, 133, 244, 0.24);
            color: #ffffff;
        }

        .result-card {
            background: var(--google-surface);
            border: 1px solid var(--google-border);
            border-left: 4px solid var(--result-accent);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 24px rgba(6, 12, 20, 0.2);
            margin-bottom: 0.85rem;
        }

        .summary-card {
            background: var(--summary-bg);
            border: 1px solid var(--summary-border);
            border-radius: 16px;
            color: var(--summary-text);
            padding: 1rem 1.1rem;
            margin-top: 0.75rem;
            margin-bottom: 1.5rem;
        }

        .stage-copy {
            color: var(--hero-subtitle-color);
            font-size: 0.96rem;
            line-height: 1.55;
            margin: 0.35rem 0 0;
        }

        .composer-card {
            background: var(--google-surface);
            border: 1px solid var(--google-border);
            border-radius: 26px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: 0 12px 28px rgba(40, 52, 64, 0.08);
            margin-bottom: 1rem;
        }

        .chat-thread {
            padding-bottom: 0.4rem;
        }

        .composer-dock {
            position: relative;
            z-index: 20;
            padding-top: 0;
            background: transparent;
            backdrop-filter: none;
        }

        .image-rail {
            background: var(--google-surface);
            border: 1px solid var(--google-border);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 8px 18px rgba(40, 52, 64, 0.04);
            margin-bottom: 1.25rem;
        }

        .results-thread {
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
            margin-top: 1rem;
        }

        .attachment-caption {
            font-size: 0.78rem;
            color: var(--google-muted);
            margin: 0.2rem 0 0.85rem;
        }

        .composer-shell {
            background: linear-gradient(180deg, #1f2530 0%, #212833 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(15, 23, 32, 0.28);
            padding: 0.85rem 1rem;
            width: 100%;
        }

        .composer-shell textarea {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }

        .composer-shell [data-testid="stTextArea"] {
            margin-bottom: 0;
        }

        .composer-shell [data-testid="stTextArea"] label,
        .composer-shell [data-testid="stFileUploader"] label {
            display: none !important;
        }

        .composer-shell [data-testid="stTextArea"] textarea {
            min-height: 74px !important;
            color: #eef3f8 !important;
            font-size: 0.98rem !important;
            line-height: 1.5 !important;
            padding: 0.35rem 0 !important;
        }

        .composer-shell [data-testid="stTextArea"] textarea::placeholder {
            color: #9aa9ba !important;
        }

        .composer-shell [data-testid="stFileUploader"] {
            background: transparent;
            border: none;
            border-radius: 0;
            box-shadow: none;
            margin-top: 0;
            padding: 0;
        }

        .composer-shell [data-testid="stFileUploader"] > section {
            padding: 0 !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        .composer-shell [data-testid="stFileUploaderDropzone"] {
            padding: 0 !important;
            min-height: auto !important;
            background: transparent !important;
            border: none !important;
        }

        .composer-shell [data-testid="stFileUploaderDropzoneInstructions"] {
            display: none !important;
        }

        .composer-shell [data-testid="stFileUploaderDropzone"] button {
            background: rgba(255, 255, 255, 0.06) !important;
            border: 1px solid rgba(255, 255, 255, 0.09) !important;
            border-radius: 14px !important;
            color: #eef3f8 !important;
            font-weight: 600 !important;
            min-height: 42px !important;
            padding: 0.45rem 0.9rem !important;
            box-shadow: none;
        }

        .composer-shell [data-testid="stFileUploaderDropzone"] button:hover {
            background: rgba(255, 255, 255, 0.1) !important;
            border-color: rgba(255, 255, 255, 0.14) !important;
        }

        .composer-attachments {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 0 0 0.55rem;
        }

        .composer-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            color: #eef3f8;
            font-size: 0.83rem;
            line-height: 1.2;
            max-width: 100%;
            padding: 0.42rem 0.72rem;
        }

        .composer-chip-icon {
            color: #8ab4ff;
            font-weight: 700;
            letter-spacing: 0.04em;
            flex: 0 0 auto;
        }

        .composer-chip-name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .st-key-attach_button_panel div[data-testid="stButton"] > button {
            min-height: 3.2rem;
            width: 3.2rem;
            border-radius: 999px;
            padding: 0;
            font-size: 1.2rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: none;
        }

        .st-key-attach_button_panel div[data-testid="stPopover"] button {
            min-height: 3.2rem;
            width: 3.2rem;
            border-radius: 999px;
            padding: 0;
            font-size: 1.2rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: none;
            color: #eef3f8;
        }

        div[data-testid="stPopoverContent"] {
            background: #161d29;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            box-shadow: 0 20px 40px rgba(7, 12, 20, 0.34);
            padding: 0.35rem;
        }

        .st-key-attach_popover_panel {
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            min-width: 260px;
        }

        .attach-popover-header {
            color: #eef3f8;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }

        .attach-popover-subtitle {
            color: #9aa9ba;
            font-size: 0.78rem;
            line-height: 1.35;
            margin: -0.2rem 0 0;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] {
            margin: 0.1rem 0 0;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] > div {
            gap: 0.45rem;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            color: #dfe7f2;
            min-height: 2.8rem;
            padding: 0.65rem 0.8rem;
            transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] label:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(138, 180, 255, 0.22);
            transform: translateY(-1px);
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] label p {
            color: inherit !important;
            font-size: 0.83rem !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] input:checked + div {
            color: #eef3f8;
        }

        .st-key-attach_popover_panel [data-testid="stRadio"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(66, 133, 244, 0.18) 0%, rgba(251, 188, 5, 0.16) 52%, rgba(52, 168, 83, 0.18) 100%);
            border-color: rgba(138, 180, 255, 0.28);
            box-shadow: 0 12px 24px rgba(18, 28, 42, 0.28);
        }

        .attach-option-copy {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }

        .attach-option-icon {
            align-items: center;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            color: #8ab4ff;
            display: inline-flex;
            flex: 0 0 auto;
            font-size: 0.68rem;
            font-weight: 800;
            height: 1.6rem;
            justify-content: center;
            letter-spacing: 0.06em;
            min-width: 2.2rem;
            text-transform: uppercase;
        }

        .attach-option-text {
            display: flex;
            flex-direction: column;
            gap: 0.12rem;
        }

        .attach-option-title {
            color: #eef3f8;
            font-size: 0.83rem;
            font-weight: 700;
            line-height: 1.1;
        }

        .attach-option-subtitle {
            color: #9aa9ba;
            font-size: 0.72rem;
            line-height: 1.2;
        }

        .st-key-attach_popover_panel .stFileUploader {
            margin-top: 0.1rem;
        }

        .st-key-attach_popover_panel [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            box-shadow: none;
            padding: 0.45rem;
        }

        .st-key-attach_popover_panel [data-testid="stFileUploaderDropzone"] button {
            width: 100%;
        }

        .st-key-send_button_panel div[data-testid="stButton"] > button {
            min-height: 3.2rem;
            width: 3.2rem;
            border-radius: 999px;
            padding: 0;
            font-size: 1.15rem;
            background: linear-gradient(135deg, #4285F4 0%, #34A853 100%);
            border: 1px solid rgba(138, 180, 248, 0.24);
            box-shadow: 0 10px 22px rgba(66, 133, 244, 0.24);
        }

        .st-key-attach_menu_panel {
            background: #1b2330;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            box-shadow: 0 16px 32px rgba(7, 12, 20, 0.32);
            padding: 0.55rem;
            margin-bottom: 0.7rem;
        }

        .st-key-attach_menu_panel div[data-testid="stButton"] > button {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            box-shadow: none;
            color: #eef3f8;
            justify-content: flex-start;
            min-height: 2.55rem;
            text-align: left;
        }

        .st-key-attach_menu_panel div[data-testid="stButton"] > button:hover {
            background: rgba(66, 133, 244, 0.14);
            border-color: rgba(138, 180, 248, 0.24);
        }

        .composer-rail-note {
            color: #9aa9ba;
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }

        .composer-row {
            display: flex;
            align-items: flex-end;
            gap: 0.85rem;
        }

        .composer-hint {
            color: #9aa9ba;
            font-size: 0.82rem;
            line-height: 1.4;
        }

        .composer-preview-label {
            color: #9aa9ba;
            font-size: 0.82rem;
            margin: 0 0 0.45rem;
        }

        .composer-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0) 0%, rgba(255, 255, 255, 0.08) 16%, rgba(255, 255, 255, 0.08) 84%, rgba(255, 255, 255, 0) 100%);
            margin: 0.25rem 0 0.55rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "runs/detect/vehicle_defect_v3/weights/best.pt"
FALLBACK_MODEL_PATH = "models/best.pt"
SECONDARY_FALLBACK_MODEL_PATH = "dataset/yolo11n.pt"
DETECTION_CONFIDENCE_THRESHOLD = 0.25
CHAT_STATE_PATH = os.path.join("data", "chat_state.json")


def make_chat_session():
    return {
        "id": str(uuid4())[:8],
        "title": "New chat",
        "preview": "No messages yet",
        "messages": [],
        "reports": [],
    }


def persist_chat_state():
    save_chat_state(
        CHAT_STATE_PATH,
        st.session_state.chat_sessions,
        st.session_state.current_chat_id,
    )


def refresh_chat_metadata(chat_session):
    user_messages = [
        message["content"].strip()
        for message in chat_session["messages"]
        if message["role"] == "user" and message["content"].strip()
    ]
    if user_messages:
        chat_session["title"] = user_messages[0][:36]
        chat_session["preview"] = user_messages[-1][:54]
    elif chat_session["reports"]:
        report_names = ", ".join(report["filename"] for report in chat_session["reports"][:2])
        chat_session["title"] = "Inspection results"
        chat_session["preview"] = f"Results for {report_names}"
    else:
        chat_session["title"] = "New chat"
        chat_session["preview"] = "No messages yet"


def get_chat_button_label(chat_session):
    message_count = len(
        [
            message
            for message in chat_session["messages"]
            if message["role"] == "user" and message["content"].strip()
        ]
    )
    result_count = len(chat_session["reports"])
    meta_parts = []
    if message_count:
        meta_parts.append(f"{message_count} msg{'s' if message_count != 1 else ''}")
    if result_count:
        meta_parts.append(f"{result_count} result{'s' if result_count != 1 else ''}")
    meta_text = " | ".join(meta_parts) if meta_parts else "New conversation"
    preview = chat_session["preview"].strip()
    if preview and preview != "No messages yet":
        meta_text = f"{meta_text}\n{preview[:48]}"
    return f"{chat_session['title'][:32]}\n{meta_text}"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)

    if os.path.exists(FALLBACK_MODEL_PATH):
        st.warning("Primary model not found. Using previous local model for testing.")
        return YOLO(FALLBACK_MODEL_PATH)

    if os.path.exists(SECONDARY_FALLBACK_MODEL_PATH):
        st.warning("Local trained models not found. Using local fallback base model for testing.")
        return YOLO(SECONDARY_FALLBACK_MODEL_PATH)

    st.warning("Local models not found. Using default YOLO model for testing only.")
    return YOLO("yolov8n.pt")


def add_chat_message(chat_session, role, content, message_type="text", metadata=None):
    chat_session["messages"].append(
        {
            "id": str(uuid4())[:8],
            "role": role,
            "type": message_type,
            "content": content,
            "metadata": metadata or {},
        }
    )
    refresh_chat_metadata(chat_session)
    persist_chat_state()


def format_chat_text(content):
    return html.escape(content).replace("\n", "<br>")


model = load_model()

if "gemini_detail_cache" not in st.session_state:
    st.session_state.gemini_detail_cache = {}

if "chat_sessions" not in st.session_state:
    saved_chat_state = load_chat_state(CHAT_STATE_PATH)
    if saved_chat_state and saved_chat_state.get("chat_sessions"):
        st.session_state.chat_sessions = saved_chat_state["chat_sessions"]
        st.session_state.current_chat_id = (
            saved_chat_state.get("current_chat_id")
            or saved_chat_state["chat_sessions"][0]["id"]
        )
    else:
        first_chat = make_chat_session()
        st.session_state.chat_sessions = [first_chat]
        st.session_state.current_chat_id = first_chat["id"]
        persist_chat_state()

if "composer_upload_key" not in st.session_state:
    st.session_state.composer_upload_key = 0

if "composer_attachment_modes" not in st.session_state:
    st.session_state.composer_attachment_modes = {}

if "composer_reset_chat_id" not in st.session_state:
    st.session_state.composer_reset_chat_id = None

if "send_in_progress" not in st.session_state:
    st.session_state.send_in_progress = False

if "pending_send_payload" not in st.session_state:
    st.session_state.pending_send_payload = None

analysis_service = AttachmentAnalysisService(
    model,
    st.session_state.gemini_detail_cache,
    error_handler=st.error,
    confidence_threshold=DETECTION_CONFIDENCE_THRESHOLD,
)


def get_current_chat():
    for chat_session in st.session_state.chat_sessions:
        if chat_session["id"] == st.session_state.current_chat_id:
            return chat_session
    new_chat = make_chat_session()
    st.session_state.chat_sessions.insert(0, new_chat)
    st.session_state.current_chat_id = new_chat["id"]
    persist_chat_state()
    return new_chat


def get_chat_by_id(chat_id):
    for chat_session in st.session_state.chat_sessions:
        if chat_session["id"] == chat_id:
            return chat_session
    return None


with st.sidebar:
    st.markdown('<div class="sidebar-title">Chats</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">Create separate inspection threads for different vehicles, visits, or issue groups.</div>',
        unsafe_allow_html=True,
    )
    top_sidebar_col, bottom_sidebar_col = st.columns([1, 1])
    with top_sidebar_col:
        if st.button("New Chat", width="stretch"):
            new_chat = make_chat_session()
            st.session_state.chat_sessions.insert(0, new_chat)
            st.session_state.current_chat_id = new_chat["id"]
            persist_chat_state()
            st.rerun()
    with bottom_sidebar_col:
        if st.button(
            "Delete Chat",
            width="stretch",
            disabled=len(st.session_state.chat_sessions) == 1,
        ):
            st.session_state.chat_sessions = [
                chat_session
                for chat_session in st.session_state.chat_sessions
                if chat_session["id"] != st.session_state.current_chat_id
            ]
            st.session_state.current_chat_id = st.session_state.chat_sessions[0]["id"]
            persist_chat_state()
            st.rerun()

    st.markdown('<div class="sidebar-section-label">Conversation Threads</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-chat-meta">Pick up where you left off in any inspection thread.</div>',
        unsafe_allow_html=True,
    )

    for chat_session in st.session_state.chat_sessions:
        is_active = chat_session["id"] == st.session_state.current_chat_id
        if st.button(
            get_chat_button_label(chat_session),
            key=f"open_chat_{chat_session['id']}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state.current_chat_id = chat_session["id"]
                persist_chat_state()
                st.rerun()

current_chat = get_current_chat()
reports_state = current_chat["reports"]

if st.session_state.send_in_progress and st.session_state.pending_send_payload:
    pending_payload = st.session_state.pending_send_payload
    target_chat = get_chat_by_id(pending_payload["chat_id"])
    if target_chat is None:
        st.session_state.send_in_progress = False
        st.session_state.pending_send_payload = None
        st.rerun()

    add_chat_message(
        target_chat,
        "user",
        pending_payload["message_text"],
        metadata={"attachments": pending_payload["attachments"]},
    )

    if pending_payload["detection_inputs"]:
        reports, assistant_summary = analysis_service.analyze_attachments(
            target_chat,
            pending_payload["detection_inputs"],
        )
        target_chat["reports"].extend(reports)
        refresh_chat_metadata(target_chat)
        persist_chat_state()
        if assistant_summary:
            add_chat_message(
                target_chat,
                "assistant",
                assistant_summary,
                message_type="analysis",
                metadata={"report_count": len(reports)},
            )
    else:
        assistant_reply = analysis_service.generate_chat_reply(
            target_chat,
            pending_payload["message_text"],
        )
        if assistant_reply:
            add_chat_message(
                target_chat,
                "assistant",
                assistant_reply,
                message_type="chat",
            )

    st.session_state.composer_reset_chat_id = pending_payload["chat_id"]
    st.session_state.composer_upload_key += 1
    st.session_state.composer_attachment_modes[pending_payload["chat_id"]] = None
    st.session_state.send_in_progress = False
    st.session_state.pending_send_payload = None
    st.rerun()

st.caption(f"Active chat: {current_chat['title']}")
workspace_panel = st.container(key="chat_workspace_panel")

with workspace_panel:
    thread_panel = st.container(key="chat_messages_panel")
    composer_panel = st.container(key="chat_sticky_panel")

    with thread_panel:
        st.markdown(
            """
            <div class="chat-stage">
                <div class="hero-dots">
                    <span class="hero-dot" style="background:#4285F4;"></span>
                    <span class="hero-dot" style="background:#EA4335;"></span>
                    <span class="hero-dot" style="background:#FBBC05;"></span>
                    <span class="hero-dot" style="background:#34A853;"></span>
                </div>
                <div class="section-label">Vehicle Inspection AI</div>
                <h1 class="hero-title">Vehicle Defect Detection using YOLO</h1>
                <p class="stage-copy">
                    Start the conversation with the issue you are seeing, attach supporting vehicle images in the composer below, and the app will respond with detections and image-based summaries in the same chat flow.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_chat["messages"]:
            st.markdown('<div class="chat-thread">', unsafe_allow_html=True)
            for message in current_chat["messages"]:
                role_class = "user" if message["role"] == "user" else "assistant"
                role_label = "User" if message["role"] == "user" else "Assistant"
                st.markdown(
                    f"""
                    <div class="chat-row {role_class}">
                        <div class="chat-bubble">
                            <div class="chat-role">{role_label}</div>
                            <div class="chat-text">{format_chat_text(message['content'])}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                attachments = get_message_attachments(message)
                if attachments:
                    attachment_columns = st.columns(min(3, len(attachments)))
                    for index, attachment in enumerate(attachments):
                        target_column = attachment_columns[index % len(attachment_columns)]
                        if get_attachment_kind(attachment) == "video":
                            target_column.video(attachment["bytes"])
                            target_column.caption(attachment["name"])
                        else:
                            target_column.image(
                                Image.open(BytesIO(attachment["bytes"])).convert("RGB"),
                                caption=attachment["name"],
                                width="stretch",
                            )
            st.markdown("</div>", unsafe_allow_html=True)

        if reports_state:
            st.markdown(
                """
                <div class="chat-row assistant">
                    <div class="chat-bubble image-rail">
                        <div class="chat-role">Assistant</div>
                        <div class="chat-text">Here are the latest findings for this chat.</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for report in reports_state:
                st.markdown(f"## Results for `{report['filename']}`")
                if report["linked_messages"]:
                    concern_html = "<br>".join(
                        f"{index}. {statement}"
                        for index, statement in enumerate(report["linked_messages"], start=1)
                    )
                    st.markdown(
                        f"""
                        <div class="summary-card">
                            <strong>Linked Messages</strong><br>
                            {concern_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if report.get("report_type") == "video_analysis":
                    st.video(report["video_bytes"])
                    metadata = report.get("metadata", {})
                    if metadata:
                        st.caption(
                            f"{metadata.get('duration_seconds', 0):.2f}s | "
                            f"{metadata.get('width', 0)}x{metadata.get('height', 0)} | "
                            f"{metadata.get('fps', 0):.2f} FPS | "
                            f"{metadata.get('frames_analyzed', 0)} sampled frames"
                        )
                    st.subheader("Local Video Findings")
                    for event in report.get("events", []):
                        event_label = event["event"].replace("_", " ").title()
                        status = "Candidate detected" if event["detected"] else "Not strongly detected"
                        st.markdown(
                            f"**{event_label}:** {status}  \n"
                            f"Confidence: {event['confidence']:.2f} | Severity: {event['severity']:.2f}"
                        )
                    st.info(report.get("local_summary", "Local video analysis was unavailable."))
                    if report["gemini_error"]:
                        st.caption("Optional Gemini video interpretation is unavailable; local analysis is shown above.")
                    elif report["video_summary"]:
                        st.markdown(
                            f"""
                            <div class="summary-card">
                                <strong>Optional Gemini Interpretation</strong><br>
                                {report['video_summary']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.info("No optional Gemini video interpretation was returned.")
                    continue

                st.image(report["annotated_image"], width="stretch")
                st.subheader("Detected Items")
                if not report["detections"]:
                    st.info("No defects detected.")
                    continue
                for detection in report["detections"]:
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <strong>Label:</strong> {detection['label']}<br>
                            <strong>Confidence:</strong> {detection['confidence']:.2f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"""
                    <div class="summary-card">
                        <strong>Problem Statement</strong><br>
                        {report['problem_statement']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if report["gemini_error"]:
                    st.caption("Detailed AI interpretation is unavailable until a valid Gemini API key is configured.")
                elif report["detailed_interpretations"]:
                    detailed_html = "<br><br>".join(
                        f"<strong>{item['label']}</strong>: {item['detail']}"
                        for item in report["detailed_interpretations"]
                    )
                    st.markdown(
                        f"""
                        <div class="summary-card">
                            <strong>Detailed Image Interpretation</strong><br>
                            {detailed_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

composer_text_key = f"composer_text_{current_chat['id']}"
if composer_text_key not in st.session_state:
    st.session_state[composer_text_key] = ""
if st.session_state.composer_reset_chat_id == current_chat["id"]:
    st.session_state[composer_text_key] = ""
    st.session_state.composer_reset_chat_id = None

with composer_panel:
    st.markdown(
        """
        <div class="composer-dock">
            <div class="composer-shell">
        """,
        unsafe_allow_html=True,
    )
    attachment_panel_key = current_chat["id"]
    attachment_mode = st.session_state.composer_attachment_modes.get(attachment_panel_key, "images")
    image_upload_key = f"composer_image_upload_{current_chat['id']}_{st.session_state.composer_upload_key}"
    video_upload_key = f"composer_video_upload_{current_chat['id']}_{st.session_state.composer_upload_key}"

    composer_uploads = []
    if attachment_mode == "images":
        composer_uploads = st.session_state.get(image_upload_key) or []
    elif attachment_mode == "video":
        selected_video = st.session_state.get(video_upload_key)
        if selected_video:
            composer_uploads = [selected_video]

    rail_col, main_col, send_col = st.columns([0.8, 5.8, 0.8], vertical_alignment="bottom")

    with rail_col:
        with st.container(key="attach_button_panel"):
            with st.popover("+"):
                with st.container(key="attach_popover_panel"):
                    st.markdown(
                        """
                        <div class="attach-popover-header">Add attachment</div>
                        <div class="attach-popover-subtitle">Choose vehicle images or one supporting video for this message.</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    selected_mode = st.radio(
                        "Attachment type",
                        options=["images", "video"],
                        index=0 if attachment_mode == "images" else 1,
                        horizontal=True,
                        key=f"attach_mode_selector_{current_chat['id']}",
                        label_visibility="collapsed",
                        format_func=lambda option: (
                            "[IMG] Images" if option == "images" else "[VID] Video"
                        ),
                    )
                    st.session_state.composer_attachment_modes[attachment_panel_key] = selected_mode

                    if selected_mode == "video":
                        selected_video = st.file_uploader(
                            "Upload one video",
                            type=["mp4", "mov", "avi", "mkv", "webm"],
                            accept_multiple_files=False,
                            key=video_upload_key,
                            label_visibility="collapsed",
                        )
                        composer_uploads = [selected_video] if selected_video else []
                    else:
                        composer_uploads = st.file_uploader(
                            "Select multiple images",
                            type=["jpg", "jpeg", "png"],
                            accept_multiple_files=True,
                            key=image_upload_key,
                            label_visibility="collapsed",
                        ) or []

    with main_col:
        if composer_uploads:
            chip_markup = []
            for uploaded_file in composer_uploads:
                attachment_kind = "video" if (uploaded_file.type or "").startswith("video/") else "image"
                icon_text = "VID" if attachment_kind == "video" else "IMG"
                chip_markup.append(
                    f"""
                    <div class="composer-chip">
                        <span class="composer-chip-icon">{icon_text}</span>
                        <span class="composer-chip-name">{html.escape(uploaded_file.name)}</span>
                    </div>
                    """
                )
            st.markdown('<div class="composer-preview-label">Attachments added to this message</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="composer-attachments">{"".join(chip_markup)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="composer-divider"></div>', unsafe_allow_html=True)
        composer_message = st.text_area(
            "Describe the issue for this message",
            key=composer_text_key,
            height=84,
            placeholder="Example: I noticed rust near the rear wheel arch and a crack around the bumper edge.",
            label_visibility="collapsed",
        )

    with send_col:
        send_button_panel = st.container(key="send_button_panel")
        send_button_label = (
            "..."
            if st.session_state.send_in_progress and st.session_state.pending_send_payload
            and st.session_state.pending_send_payload.get("chat_id") == current_chat["id"]
            else "^"
        )
        with send_button_panel:
            if st.button(
                send_button_label,
                width="stretch",
                key=f"send_message_{current_chat['id']}",
                disabled=st.session_state.send_in_progress,
            ):
                if not composer_message.strip() and not composer_uploads:
                    st.warning("Add a message, an image, or both before sending.")
                else:
                    message_text = composer_message.strip() or "Attached vehicle image(s) for inspection context."
                    attachments = [build_attachment_payload(uploaded_file) for uploaded_file in composer_uploads or []]
                    detection_inputs = [
                        build_analysis_attachment(uploaded_file, message_text)
                        for uploaded_file in composer_uploads or []
                    ]
                    st.session_state.pending_send_payload = {
                        "chat_id": current_chat["id"],
                        "message_text": message_text,
                        "attachments": attachments,
                        "detection_inputs": detection_inputs,
                    }
                    st.session_state.send_in_progress = True
                    st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)
