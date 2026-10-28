import html
import io
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import tempfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import folium
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs" / "detect" / \
    "railway_defect_model-2" / "weights" / "best.pt"
TRACK_CSV = BASE_DIR / "track_locations.csv"
DEFAULT_IMAGE_DIR = BASE_DIR / "track_dataset" / "images"
DB_PATH = BASE_DIR / "itms_inspections.db"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

st.set_page_config(
    page_title="ITMS | Railway Control Centre",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {
    0: "Corrosion",
    1: "Spalling",
    2: "Cracks",
    3: "No Problem",
    4: "RCF",
}
CLASS_LIST = ["Cracks", "Spalling", "Corrosion", "RCF", "No Problem"]
MAP_COLORS = {
    "Cracks": "red",
    "Spalling": "orange",
    "Corrosion": "beige",
    "RCF": "purple",
    "No Problem": "green",
}

# ============================================================
# SQLITE PERSISTENT STORAGE
# ============================================================


def get_db_connection():
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                priority TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                track_km TEXT,
                sequence INTEGER
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_image_name
            ON inspections(image_name)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_timestamp
            ON inspections(timestamp)
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                inspection_id INTEGER,
                image_name TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                track_km TEXT,
                status TEXT NOT NULL DEFAULT 'Open',
                assigned_to TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                repair_notes TEXT DEFAULT '',
                FOREIGN KEY (inspection_id) REFERENCES inspections(id)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_maintenance_status
            ON maintenance_cases(status)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_maintenance_image
            ON maintenance_cases(image_name)
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('Administrator', 'Inspector', 'Maintenance')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        connection.commit()


def save_inspection(detection):
    with get_db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO inspections (
                timestamp, image_name, predicted_class, priority,
                latitude, longitude, track_km, sequence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.get("Time"),
            detection.get("Frame ID"),
            detection.get("Type"),
            detection.get("Priority"),
            detection.get("Latitude"),
            detection.get("Longitude"),
            str(detection.get("KM", "Unknown")),
            detection.get("Sequence"),
        ))
        connection.commit()
        return cursor.lastrowid


def load_inspection_history(search_text="", start_date=None, end_date=None):
    query = """
        SELECT id, timestamp, image_name, predicted_class, priority,
               latitude, longitude, track_km, sequence
        FROM inspections
        WHERE 1=1
    """
    params = []

    if search_text.strip():
        query += """
            AND (
                image_name LIKE ? COLLATE NOCASE
                OR predicted_class LIKE ? COLLATE NOCASE
                OR priority LIKE ? COLLATE NOCASE
            )
        """
        pattern = f"%{search_text.strip()}%"
        params.extend([pattern, pattern, pattern])

    if start_date:
        query += " AND date(timestamp) >= date(?)"
        params.append(start_date.isoformat())

    if end_date:
        query += " AND date(timestamp) <= date(?)"
        params.append(end_date.isoformat())

    query += " ORDER BY id DESC"

    with get_db_connection() as connection:
        rows = connection.execute(query, params).fetchall()

    return [{
        "Database ID": row["id"],
        "Time": row["timestamp"],
        "Frame ID": row["image_name"],
        "Type": row["predicted_class"],
        "Priority": row["priority"],
        "Latitude": row["latitude"],
        "Longitude": row["longitude"],
        "KM": row["track_km"],
        "Sequence": row["sequence"],
    } for row in rows]


def count_all_inspections():
    with get_db_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]


def delete_all_inspection_history():
    """Only call from an explicit user action."""
    with get_db_connection() as connection:
        connection.execute("DELETE FROM inspections")
        connection.commit()


def create_maintenance_case(inspection):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO maintenance_cases (
                created_at, updated_at, inspection_id, image_name, defect_type,
                priority, latitude, longitude, track_km, status, assigned_to,
                notes, repair_notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open', '', '', '')
        """, (
            now, now, inspection.get("Database ID"), inspection.get(
                "Frame ID", "Unknown"),
            inspection.get("Type", "Unknown"), inspection.get(
                "Priority", "NORMAL"),
            inspection.get("Latitude"), inspection.get("Longitude"),
            str(inspection.get("KM", "Unknown"))
        ))
        connection.commit()
        return cursor.lastrowid


def load_maintenance_cases(search_text=""):
    query = "SELECT * FROM maintenance_cases WHERE 1=1"
    params = []
    if search_text.strip():
        pattern = f"%{search_text.strip()}%"
        query += """ AND (
            image_name LIKE ? COLLATE NOCASE
            OR defect_type LIKE ? COLLATE NOCASE
            OR status LIKE ? COLLATE NOCASE
            OR assigned_to LIKE ? COLLATE NOCASE
            OR track_km LIKE ? COLLATE NOCASE
        )"""
        params.extend([pattern, pattern, pattern, pattern, pattern])
    query += " ORDER BY id DESC"
    with get_db_connection() as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def update_maintenance_case(case_id, status, assigned_to, notes, repair_notes):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as connection:
        connection.execute("""
            UPDATE maintenance_cases
            SET status = ?, assigned_to = ?, notes = ?, repair_notes = ?, updated_at = ?
            WHERE id = ?
        """, (
            status, assigned_to.strip(), notes.strip(), repair_notes.strip(), now, int(case_id)
        ))
        connection.commit()


def delete_maintenance_case(case_id):
    with get_db_connection() as connection:
        connection.execute(
            "DELETE FROM maintenance_cases WHERE id = ?", (int(case_id),))
        connection.commit()


def create_maintenance_pdf(cases):
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4), rightMargin=10*mm, leftMargin=10*mm,
        topMargin=12*mm, bottomMargin=12*mm, title="ITMS Maintenance Cases"
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MaintenanceTitle", parent=styles["Title"], fontSize=18,
        leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#12364a"),
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="MaintenanceCell", parent=styles["Normal"], fontSize=7, leading=9,
        wordWrap="CJK"
    ))
    story = [
        Paragraph("ITMS — Maintenance Management Report",
                  styles["MaintenanceTitle"]),
        Paragraph(
            f"Generated: {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
            styles["Normal"]
        ),
        Spacer(1, 10)
    ]
    headers = ["Case ID", "Created", "Image", "Defect",
               "Priority", "Status", "Assigned To", "Track KM", "Notes"]
    rows = [headers]
    for item in cases:
        rows.append([
            str(item.get("id", "")),
            Paragraph(escape(str(item.get("created_at", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("image_name", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("defect_type", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("priority", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("status", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("assigned_to", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("track_km", ""))),
                      styles["MaintenanceCell"]),
            Paragraph(escape(str(item.get("notes", ""))),
                      styles["MaintenanceCell"]),
        ])
    table = Table(rows, colWidths=[
                  15*mm, 29*mm, 36*mm, 22*mm, 20*mm, 27*mm, 28*mm, 18*mm, 50*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12364a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c4d1d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f5f8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))
    story.append(table)
    doc.build(story)
    output.seek(0)
    return output.getvalue()


initialize_database()


# ============================================================
# AUTHENTICATION AND ROLE-BASED ACCESS
# ============================================================
PASSWORD_ITERATIONS = 310_000
ROLES = ["Administrator", "Inspector", "Maintenance"]


def hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(
        "utf-8"), salt, PASSWORD_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password, salt_hex, expected_hash):
    actual_hash, _ = hash_password(password, salt_hex)
    return hmac.compare_digest(actual_hash, expected_hash)


def user_count():
    with get_db_connection() as connection:
        return connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def authenticate_user(username, password):
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT username, password_hash, salt, role, active FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),)
        ).fetchone()
    if not row or not row["active"]:
        return None
    if verify_password(password, row["salt"], row["password_hash"]):
        return {"username": row["username"], "role": row["role"]}
    return None


def create_user(username, password, role):
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Username must contain at least 3 characters.")
    if len(password) < 12:
        raise ValueError("Password must contain at least 12 characters.")
    if role not in ROLES:
        raise ValueError("Invalid role.")
    password_hash, salt = hash_password(password)
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO users (username, password_hash, salt, role, active, created_at) VALUES (?, ?, ?, ?, 1, ?)",
            (username, password_hash, salt, role,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        connection.commit()


def list_users():
    with get_db_connection() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT id, username, role, active, created_at FROM users ORDER BY username"
        ).fetchall()]


def set_user_active(user_id, active):
    with get_db_connection() as connection:
        connection.execute(
            "UPDATE users SET active = ? WHERE id = ?", (int(bool(active)), int(user_id)))
        connection.commit()


def render_auth_gate():
    st.markdown('<div class="main-title">Railway Control Centre</div>',
                unsafe_allow_html=True)
    st.caption("ITMS secure access")
    if user_count() == 0:
        st.info(
            "First-time setup: create the initial Administrator account. Store its credentials securely.")
        with st.form("initial_admin_setup"):
            username = st.text_input("Administrator username")
            password = st.text_input(
                "Password (minimum 12 characters)", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button(
                "Create administrator", width="stretch")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    create_user(username, password, "Administrator")
                    st.success("Administrator account created. Sign in below.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not create administrator: {exc}")
    with st.form("itms_login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password", type="password", key="login_password")
        submitted = st.form_submit_button("Sign in", width="stretch")
    if submitted:
        user = authenticate_user(username, password)
        if user:
            st.session_state["itms_user"] = user
            st.rerun()
        else:
            st.error("Invalid username/password or account disabled.")


# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --bg:#07101d; --panel:#0b1726; --panel2:#0e1c2e; --line:#203247; --muted:#8497ad; --text:#e8eef7; --blue:#2e8bff; --green:#42d98b; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
/* Hide Streamlit's own top header/Deploy toolbar so it does not cover the app. */
header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, #MainMenu { display:none !important; visibility:hidden !important; height:0 !important; }
[data-testid="stAppViewContainer"] > .main { padding-top:0 !important; }
[data-testid="stAppViewContainer"] > .main .block-container { padding-top:.55rem !important; }
.stApp { background:#060d18; color:var(--text); }
.block-container { padding: .55rem 1.05rem 1.2rem; max-width:1900px; }
[data-testid="stSidebar"] { background:#081321; border-right:1px solid #1a2b40; min-width:220px; }
[data-testid="stSidebar"] > div:first-child { padding-top:.7rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#a9b8ca; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { font-size:12px; }
[data-testid="stMetric"] { background:#0c1929; border:1px solid #1d3046; border-radius:9px; padding:9px 12px; min-height:76px; }
[data-testid="stMetricLabel"] { color:#8fa2b8 !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color:#edf5ff !important; font-weight:800; font-size:25px !important; }
.main-title { font-size:23px; font-weight:800; color:#f2f6fc; }
.subtitle { font-size:10px; letter-spacing:.5px; color:#8b9db3; margin-top:2px; }
.muted { color:#8b9db3; font-size:12px; }
.eyebrow { color:#8ca2bc; font-size:9px; font-weight:800; letter-spacing:1px; text-transform:uppercase; margin-bottom:5px; }
.section-title { font-size:14px; font-weight:800; color:#edf4fc; margin:0 0 9px; }
.panel { background:#0b1726; border:1px solid #203247; border-radius:10px; padding:12px; }
.panel-label { color:#8195ad; font-size:9px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; margin:10px 0 4px; }
.pill { display:inline-block; border-radius:99px; padding:3px 7px; font-size:9px; font-weight:800; border:1px solid transparent; margin-right:3px; }
.pill-cyan { color:#7bd6ff; background:#102e48; border-color:#214c71; }
.pill-green { color:#73efad; background:#103225; border-color:#236247; }
.pill-red { color:#ff929b; background:#3b1b27; border-color:#743347; }
.pill-orange { color:#ffc17c; background:#3c2a17; border-color:#76512b; }
.pill-yellow { color:#ffe37a; background:#39320f; border-color:#6f6221; }
.pill-purple { color:#d5aaff; background:#2e2040; border-color:#5b3b7a; }
.pill-normal { color:#73efad; background:#103225; border-color:#236247; }
.status { text-align:center; padding:8px; border-radius:9px; font-weight:800; font-size:10px; }
.status-ready { background:#0d2a22; border:1px solid #1e6047; color:#73efad; }
.status-running { background:#3b2815; border:1px solid #92602a; color:#ffd09a; }
div.stButton > button, [data-testid="stDownloadButton"] button { border-radius:7px; font-weight:700; min-height:32px; border:1px solid #2768a8; background:#135ea5; color:white; font-size:12px; }
div.stButton > button:hover { background:#1c75c6; border-color:#3b8ee1; }
[data-testid="stFileUploader"] { border:1px dashed #29435f; border-radius:9px; background:#0b1726; }
[data-testid="stDataFrame"] { border:1px solid #203247; border-radius:9px; }
hr { border-color:#1c2d42; }
.small-note { font-size:10px; color:#8297af; }
.itms-topbar { display:flex; align-items:center; justify-content:space-between; background:#081321; border:1px solid #1a2b40; border-radius:10px; padding:9px 14px; margin:0 0 12px 0; }
.itms-brand { display:flex; align-items:center; gap:9px; }
.itms-logo { display:flex; align-items:center; justify-content:center; width:30px; height:30px; border-radius:8px; background:#123154; color:#66b8ff; font-size:17px; font-weight:900; }
.itms-brand-name { color:#f0f5fc; font-size:15px; font-weight:800; line-height:1.1; }
.itms-brand-sub { color:#8296ad; font-size:9px; margin-top:3px; }
.itms-top-status { color:#63e8a0; font-size:10px; font-weight:700; }
.itms-top-user { color:#c2cede; font-size:10px; text-align:right; }
.dash-panel { background:#0a1625; border:1px solid #1e3045; border-radius:10px; padding:10px; margin-bottom:10px; }
.dash-panel-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; color:#eaf1fa; font-weight:800; font-size:12px; }
.dash-small { color:#8195ad; font-size:9px; }
[data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"] { background:transparent; }
</style>
""", unsafe_allow_html=True)

if "itms_user" not in st.session_state:
    render_auth_gate()
    st.stop()

current_user = st.session_state["itms_user"]
with st.sidebar:
    st.markdown(f"**Signed in:** {html.escape(current_user['username'])}")
    st.caption(f"Role: {current_user['role']}")
    if st.button("Sign out", key="itms_sign_out", width="stretch"):
        st.session_state.pop("itms_user", None)
        st.rerun()

# ============================================================
# TRACK COORDINATES
# ============================================================


@st.cache_data
def load_track_data(csv_path):
    df = pd.read_csv(csv_path)
    required = ["image_name", "latitude", "longitude", "km", "sequence"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["image_key"] = df["image_name"].str.lower()
    for col in ["latitude", "longitude", "sequence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "sequence"]).copy()
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ]
    df["sequence"] = df["sequence"].astype(int)
    return df.sort_values("sequence").reset_index(drop=True)


try:
    TRACK_DATA = load_track_data(str(TRACK_CSV))
    TRACK_ERROR = None
except Exception as exc:
    TRACK_DATA = pd.DataFrame(columns=[
        "image_name", "image_key", "latitude", "longitude", "km", "sequence"
    ])
    TRACK_ERROR = str(exc)

TRACK_POINTS = (
    list(zip(TRACK_DATA["latitude"].astype(float),
         TRACK_DATA["longitude"].astype(float)))
    if not TRACK_DATA.empty else []
)


def get_track_info(filename):
    if TRACK_DATA.empty:
        return None, None, "Unknown", None
    key = Path(str(filename)).name.strip().lower()
    rows = TRACK_DATA[TRACK_DATA["image_key"] == key]
    if rows.empty:
        return None, None, "Unknown", None
    row = rows.iloc[0]
    return float(row["latitude"]), float(row["longitude"]), str(row["km"]), int(row["sequence"])


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
    MODEL_ERROR = None
except Exception as exc:
    model = None
    MODEL_ERROR = str(exc)

# ============================================================
# SESSION STATE
# ============================================================


def init_state():
    defaults = {
        "processed": 0,
        "detections": [],
        "counts": {name: 0 for name in CLASS_LIST},
        "gallery": [],
        "latest_detection": None,
        "latest_image": None,
        "latest_processing_time": 0.0,
        "processed_batch": None,
        "inspection_complete": False,
        "inspection_running": False,
        "last_run_successful": 0,
        "last_run_failed": 0,
        "db_history_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def sync_session_from_database():
    history = load_inspection_history()
    st.session_state.detections = history
    st.session_state.processed = len(history)
    counts = {name: 0 for name in CLASS_LIST}
    for record in history:
        if record["Type"] in counts:
            counts[record["Type"]] += 1
    st.session_state.counts = counts
    st.session_state.db_history_loaded = True


if not st.session_state.db_history_loaded:
    try:
        sync_session_from_database()
    except Exception as exc:
        st.error(f"Unable to restore inspection history: {exc}")


def priority_for(defect):
    return {
        "Cracks": "CRITICAL",
        "RCF": "HIGH",
        "Spalling": "HIGH",
        "Corrosion": "MODERATE",
        "No Problem": "NORMAL",
    }.get(defect, "NORMAL")


def pill_class(defect):
    return {
        "Cracks": "pill-red",
        "Spalling": "pill-orange",
        "Corrosion": "pill-yellow",
        "RCF": "pill-purple",
        "No Problem": "pill-normal",
    }.get(defect, "pill-cyan")


def priority_pill(priority):
    return {
        "CRITICAL": "pill-red",
        "HIGH": "pill-orange",
        "MODERATE": "pill-yellow",
        "NORMAL": "pill-green",
    }.get(priority, "pill-cyan")


def section(title, eyebrow=None):
    if eyebrow:
        st.markdown(
            f'<div class="eyebrow">{html.escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def render_latest_details(item):
    st.markdown(f"""
    <div class="panel">
      <div class="eyebrow">LATEST CLASSIFICATION</div>
      <div style="margin:10px 0">
        <span class="pill {pill_class(item['Type'])}">{html.escape(str(item['Type']))}</span>
        <span class="pill {priority_pill(item['Priority'])}">{html.escape(str(item['Priority']))}</span>
      </div>
      <div class="panel-label">Frame ID</div>
      <div style="font-weight:700;overflow-wrap:anywhere">{html.escape(str(item['Frame ID']))}</div>
      <div class="panel-label">Track kilometre</div>
      <div style="font-weight:700">{html.escape(str(item['KM']))}</div>
      <div class="panel-label">Inspection time</div>
      <div style="font-weight:700">{html.escape(str(item['Time']))}</div>
      <div class="panel-label">Coordinates</div>
      <div style="font-weight:700;overflow-wrap:anywhere">{item.get('Latitude')}, {item.get('Longitude')}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DATASET DISCOVERY
# ============================================================
def find_dataset_images(folder):
    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower()
         in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def order_images_by_sequence(images):
    if TRACK_DATA.empty:
        return sorted(images, key=lambda p: p.name.lower())
    lookup = {str(row["image_key"]).lower(): int(row["sequence"])
              for _, row in TRACK_DATA.iterrows()}
    return sorted(images, key=lambda p: (lookup.get(p.name.strip().lower(), 999999), p.name.lower()))


# ============================================================
# MAP
# ============================================================
def build_map(current_lat=None, current_lon=None, current_defect=None, records=None):
    points = TRACK_POINTS
    if not points:
        return None

    valid_current = current_lat is not None and current_lon is not None
    center = [current_lat, current_lon] if valid_current else points[0]
    # Use a reliable public basemap by default. CARTO's older dark_matter
    # tiles can show an "API KEY REQUIRED" watermark in some environments.
    fmap = folium.Map(
        location=center, zoom_start=15, tiles="OpenStreetMap", control_scale=True
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite imagery", overlay=False, control=True,
    ).add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)

    if len(points) > 1:
        folium.PolyLine(points, color="#23b9e8", weight=5,
                        opacity=0.85, tooltip="CSV inspection route").add_to(fmap)

    folium.Marker(points[0], tooltip="Route start", icon=folium.Icon(
        color="blue", icon="play")).add_to(fmap)

    map_records = records if records is not None else st.session_state.detections
    for item in map_records:
        lat, lon = item.get("Latitude"), item.get("Longitude")
        if lat is None or lon is None:
            continue
        defect = item["Type"]
        popup = (
            f"<b>{html.escape(str(defect))}</b><br>"
            f"Priority: {html.escape(str(item['Priority']))}<br>"
            f"Image: {html.escape(str(item['Frame ID']))}<br>"
            f"KM: {html.escape(str(item['KM']))}<br>"
            f"Coordinates: {lat:.6f}, {lon:.6f}"
        )
        color = MAP_COLORS.get(defect, "blue")
        folium.CircleMarker(
            location=[lat, lon], radius=8, color=color, fill=True,
            fill_color=color, fill_opacity=0.9,
            tooltip=f"{defect} | {item['Frame ID']}",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(fmap)

    if valid_current:
        folium.Marker(
            [current_lat, current_lon],
            tooltip="Latest inspection position",
            popup=f"<b>Latest inspection</b><br>{html.escape(str(current_defect or 'Inspecting'))}",
            icon=folium.Icon(color="blue", icon="train", prefix="fa"),
        ).add_to(fmap)

    if len(points) > 1:
        fmap.fit_bounds(points, padding=(25, 25))
    return fmap


def render_map(fmap, height=600, key="itms_map"):
    if fmap is None:
        st.warning("No valid route coordinates. Check track_locations.csv.")
    else:
        st_folium(fmap, height=height, width=None,
                  key=key, returned_objects=[])


# ============================================================
# INFERENCE AND PERSISTENCE
# ============================================================
def process_one(source):
    if isinstance(source, (str, Path)):
        image_path = Path(source)
        image_name = image_path.name
        image = Image.open(image_path).convert("RGB")
    elif isinstance(source, Image.Image):
        image_name = f"camera_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        image = source.convert("RGB")
    else:
        source.seek(0)
        image_name = source.name
        image = Image.open(source).convert("RGB")

    started = time.time()
    result = model.predict(source=image, conf=0.25,
                           device="cpu", verbose=False)[0]
    elapsed = time.time() - started
    annotated_rgb = result.plot()[:, :, ::-1].copy()

    if result.boxes is not None and len(result.boxes) > 0:
        best_idx = int(result.boxes.conf.argmax().item())
        class_id = int(result.boxes.cls[best_idx].item())
        defect = CLASS_NAMES.get(class_id, "No Problem")
    else:
        defect = "No Problem"

    lat, lon, km, sequence = get_track_info(image_name)
    detection = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Frame ID": image_name,
        "Type": defect,
        "Priority": priority_for(defect),
        "Latitude": lat,
        "Longitude": lon,
        "KM": km,
        "Sequence": sequence,
    }

    # Persist before updating the in-memory dashboard.
    database_id = save_inspection(detection)
    detection["Database ID"] = database_id

    st.session_state.processed += 1
    st.session_state.detections.insert(0, detection)
    st.session_state.counts[defect] += 1
    st.session_state.gallery.insert(0, {**detection, "Image": annotated_rgb})
    st.session_state.latest_detection = detection
    st.session_state.latest_image = annotated_rgb
    st.session_state.latest_processing_time = elapsed
    return detection, annotated_rgb


# ============================================================
# PDF REPORT
# ============================================================
def create_inspection_pdf(detections, counts):
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(A4), rightMargin=12*mm,
        leftMargin=12*mm, topMargin=12*mm, bottomMargin=12*mm,
        title="ITMS Railway Inspection Report", author="ITMS",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ITMSTitle", parent=styles["Title"], fontSize=20,
        leading=24, alignment=TA_CENTER, textColor=colors.HexColor("#12364a"),
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ITMSSubtitle", parent=styles["Normal"], fontSize=9,
        leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#526575"),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="ITMSCell", parent=styles["Normal"], fontSize=7, leading=9,
        wordWrap="CJK",
    ))

    total = len(detections)
    normal = sum(1 for x in detections if x.get("Type") == "No Problem")
    defects = total - normal
    mapped = sum(1 for x in detections if x.get("Latitude")
                 is not None and x.get("Longitude") is not None)
    story = [
        Paragraph("ITMS — Railway Inspection Report", styles["ITMSTitle"]),
        Paragraph("Vision-Based Contactless Track Crack and Gauge Detection System",
                  styles["ITMSSubtitle"]),
        Paragraph(
            f"<b>Report generated:</b> {escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", styles["Normal"]),
        Spacer(1, 8),
    ]

    summary_data = [
        ["Inspection Metric", "Value", "Inspection Metric", "Value"],
        ["Images processed", str(
            total), "Defect classifications", str(defects)],
        ["Normal classifications", str(
            normal), "GPS-mapped images", str(mapped)],
    ]
    summary_table = Table(summary_data, colWidths=[
                          48*mm, 24*mm, 52*mm, 30*mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12364a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#eef4f7")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c4d1d9")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([summary_table, Spacer(1, 14), Paragraph(
        "Defect Classification Summary", styles["Heading2"])])

    class_rows = [["Class", "Number of Images", "Display Priority"]]
    for name in CLASS_LIST:
        class_rows.append([name, str(counts.get(name, 0)), priority_for(name)])
    class_table = Table(class_rows, colWidths=[
                        55*mm, 45*mm, 45*mm], repeatRows=1)
    class_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176b83")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c4d1d9")),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([class_table, Spacer(1, 14), Paragraph(
        "Individual Inspection Records", styles["Heading2"])])

    rows = [[
        "Inspection Time", "Image Filename", "Predicted Class", "Priority",
        "Latitude", "Longitude", "Track KM", "Sequence",
    ]]
    for item in reversed(detections):
        lat, lon = item.get("Latitude"), item.get("Longitude")
        rows.append([
            Paragraph(escape(str(item.get("Time", "Unknown"))),
                      styles["ITMSCell"]),
            Paragraph(escape(str(item.get("Frame ID", "Unknown"))),
                      styles["ITMSCell"]),
            Paragraph(escape(str(item.get("Type", "Unknown"))),
                      styles["ITMSCell"]),
            Paragraph(escape(str(item.get("Priority", "Unknown"))),
                      styles["ITMSCell"]),
            "" if lat is None else f"{lat:.6f}",
            "" if lon is None else f"{lon:.6f}",
            Paragraph(escape(str(item.get("KM", "Unknown"))),
                      styles["ITMSCell"]),
            "" if item.get("Sequence") is None else str(item["Sequence"]),
        ])
    record_table = Table(
        rows,
        colWidths=[32*mm, 43*mm, 27*mm, 24*mm, 25*mm, 25*mm, 25*mm, 18*mm],
        repeatRows=1,
    )
    record_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12364a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c4d1d9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f0f5f8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 7), ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.extend([record_table, Spacer(1, 12), Paragraph(
        "GPS coordinates are included only when filenames match valid entries in track_locations.csv. "
        "Predictions and priority labels are prototype outputs and require verification before maintenance decisions.",
        styles["Normal"],
    )])

    def page_number(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526575"))
        canvas.drawRightString(landscape(
            A4)[0] - 12*mm, 7*mm, f"ITMS Inspection Report | Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    output.seek(0)
    return output.getvalue()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom:14px">
      <div style="font-family:'Space Grotesk';font-size:27px;font-weight:700;color:#eef8ff">
        <span style="color:#42d6f5">◈</span> ITMS
      </div>
      <div class="subtitle">RAIL CONTROL CENTRE</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    all_nav = {
        "◫  Overview": "Dashboard",
        "◉  Live monitor": "Live Monitoring",
        "▦  Image gallery": "Inspection Gallery",
        "⌖  Track map": "Map View",
        "≋  Detection log": "Detections",
        "▥  Analytics": "Analytics",
        "▤  Inspection reports": "Reports",
        "◷  Inspection history": "History",
        "🛠  Maintenance management": "Maintenance",
        "♙  User management": "User Management",
    }
    role_pages = {
        "Administrator": set(all_nav.values()),
        "Inspector": {"Dashboard", "Live Monitoring", "Inspection Gallery", "Map View", "Detections", "Analytics", "Reports", "History"},
        "Maintenance": {"Dashboard", "Map View", "History", "Maintenance"},
    }
    allowed_pages = role_pages.get(current_user["role"], {"Dashboard"})
    nav = {label: target for label, target in all_nav.items()
           if target in allowed_pages}
    page = nav[st.radio("WORKSPACE", list(nav.keys()),
                        key="main_workspace_navigation")]

    st.divider()
    if current_user["role"] in ("Administrator", "Inspector"):
        st.markdown("#### Inspection session")
    image_source = st.radio(
        "Image source", ["Existing dataset folder", "Manual upload", "Live camera"], key="itms_image_source")
    dataset_folder = str(DEFAULT_IMAGE_DIR)
    dataset_images = []
    uploaded_files = []

    if image_source == "Existing dataset folder":
        dataset_folder = st.text_input(
            "Dataset folder path", value=dataset_folder)
        try:
            dataset_images = find_dataset_images(dataset_folder)
            st.caption(f"{len(dataset_images)} image(s) found")
        except Exception as exc:
            st.error(str(exc))
    elif image_source == "Manual upload":
        uploaded_files = st.file_uploader(
            "Add track images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        ) or []
        st.caption(f"{len(uploaded_files)} file(s) queued")
    else:
        camera_connection = st.selectbox(
            "Camera connection type",
            ["Local USB / built-in camera",
                "External IP camera (RTSP)", "Video file (simulation)"],
            key="itms_camera_connection_type",
        )
        video_upload = None
        if camera_connection == "Video file (simulation)":
            video_upload = st.file_uploader(
                "Upload railway video for testing",
                type=["mp4", "avi", "mov", "mkv", "m4v"],
                accept_multiple_files=False,
                key="itms_simulation_video_upload",
                help="Simulates a camera feed using a prerecorded video. It does not represent a live train camera connection.",
            )
            camera_index = 0
            rtsp_url = ""
            if video_upload is not None:
                st.caption(
                    f"Video selected: {video_upload.name} · {video_upload.size / (1024 * 1024):.1f} MB")
            else:
                st.caption(
                    "Upload a railway video to test live-style frame processing without camera hardware.")
        elif camera_connection == "Local USB / built-in camera":
            camera_index = st.number_input(
                "Camera device index", min_value=0, max_value=10, value=0, step=1,
                key="itms_local_camera_index",
            )
            rtsp_url = ""
            st.caption(
                "For a camera physically connected to the computer running Streamlit.")
        else:
            rtsp_url = st.text_input(
                "Camera RTSP URL",
                value="",
                placeholder="rtsp://CAMERA_IP:554/STREAM_PATH",
                help="Enter the RTSP stream address supplied by the camera/NVR administrator. Avoid sharing URLs containing credentials.",
                key="itms_rtsp_url",
            ).strip()
            camera_index = 0
            st.caption(
                "For a network camera reachable from the computer running Streamlit. "
                "The camera and remote computer must be on a routed network/VPN that permits the stream."
            )
        camera_duration = st.slider(
            "Monitoring duration (seconds)", 5, 300, 30, 5,
            key="itms_camera_duration",
        )
        camera_sample_interval = st.slider(
            "Save/process interval (seconds)", 0.5, 5.0, 1.0, 0.5,
            key="itms_camera_sample_interval",
        )

    delay = st.slider("Frame interval (seconds)", 0.0, 1.5, 0.2, 0.1)
    if image_source == "Existing dataset folder":
        available_count = len(dataset_images)
    elif image_source == "Manual upload":
        available_count = len(uploaded_files)
    else:
        available_count = 1
    camera_source_ready = (
        image_source != "Live camera"
        or camera_connection == "Local USB / built-in camera"
        or (camera_connection == "External IP camera (RTSP)" and bool(rtsp_url))
        or (camera_connection == "Video file (simulation)" and video_upload is not None)
    )
    start_button = st.button(
        "▶ Start inspection",
        width="stretch",
        disabled=(available_count == 0 or model is None or not camera_source_ready
                  or current_user["role"] == "Maintenance")
    )
    if current_user["role"] == "Maintenance":
        st.caption(
            "Inspection execution is restricted to Inspectors and Administrators.")
    reset_button = st.button("↻ Reset session", width="stretch")

    st.divider()
    st.markdown("#### System health")
    st.markdown(
        f'<span class="pill {"pill-green" if model is not None else "pill-red"}">'
        f'{"● MODEL READY" if model is not None else "● MODEL ERROR"}</span>',
        unsafe_allow_html=True,
    )
    st.caption("YOLO Model-2 · CPU inference")
    st.caption(f"Persistent database: {DB_PATH.name}")
    if MODEL_ERROR:
        st.error(MODEL_ERROR)
    if TRACK_ERROR:
        st.error(f"Track CSV: {TRACK_ERROR}")
    elif not TRACK_DATA.empty:
        st.caption(f"{len(TRACK_DATA)} route rows loaded")
    else:
        st.warning("No valid route coordinates loaded.")

# ============================================================
# RESET CURRENT SESSION (DATABASE IS NOT DELETED)
# ============================================================
if reset_button:
    st.session_state.processed = 0
    st.session_state.detections = []
    st.session_state.counts = {name: 0 for name in CLASS_LIST}
    st.session_state.gallery = []
    st.session_state.latest_detection = None
    st.session_state.latest_image = None
    st.session_state.latest_processing_time = 0.0
    st.session_state.processed_batch = None
    st.session_state.inspection_complete = False
    st.session_state.inspection_running = False
    # Reset the current view, then reload persistent metadata.
    sync_session_from_database()
    st.session_state.gallery = []
    st.session_state.latest_detection = None
    st.session_state.latest_image = None
    st.rerun()

# ============================================================
# HEADER
# ============================================================
left, right = st.columns([3.5, 1.1], vertical_alignment="center")
with left:
    st.markdown('<div class="eyebrow">INTELLIGENT TRACK MONITORING SYSTEM</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">Railway Control Centre</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="muted">Vision-based contactless track inspection and defect intelligence</div>',
                unsafe_allow_html=True)
with right:
    if st.session_state.inspection_running:
        st.markdown(
            '<div class="status status-running">● INSPECTION RUNNING</div>', unsafe_allow_html=True)
    elif st.session_state.inspection_complete:
        st.markdown(
            '<div class="status status-ready">✓ RUN COMPLETE</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status status-ready">● SYSTEM STANDBY</div>', unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 20px">
  <span class="pill pill-cyan">MODEL 02 · YOLO</span>
  <span class="pill pill-green">CPU INFERENCE</span>
  <span class="pill pill-cyan">CONTACTLESS VISION</span>
  <span class="pill pill-normal">5 CLASSES</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INSPECTION RUN
# ============================================================
if start_button:
    if image_source == "Live camera":
        camera = None
        try:
            simulation_temp_path = None
            if camera_connection == "External IP camera (RTSP)":
                # Prefer FFmpeg for RTSP streams; fall back to OpenCV's default backend.
                camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not camera.isOpened():
                    camera.release()
                    camera = cv2.VideoCapture(rtsp_url)
                camera_description = "External RTSP camera"
            elif camera_connection == "Video file (simulation)":
                if video_upload is None:
                    st.error(
                        "Upload a railway video before starting the simulation.")
                    camera = None
                else:
                    suffix = Path(video_upload.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                        temp_video.write(video_upload.getbuffer())
                        simulation_temp_path = temp_video.name
                    camera = cv2.VideoCapture(simulation_temp_path)
                    camera_description = f"Video simulation ({video_upload.name})"
            else:
                camera = cv2.VideoCapture(
                    int(camera_index), cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
                if not camera.isOpened():
                    camera.release()
                    camera = cv2.VideoCapture(int(camera_index))
                camera_description = f"Local camera device {int(camera_index)}"

            if not camera.isOpened():
                st.error(
                    "Unable to open the video source. For RTSP, verify the URL, camera credentials, "
                    "network/VPN reachability, stream path, and firewall rules. For a local camera, "
                    "check permissions, close other apps using it, and verify the device index."
                )
            else:
                st.session_state.inspection_complete = False
                st.session_state.inspection_running = True
                st.markdown("#### Live camera feed")
                st.caption(
                    f"Source: {camera_description}. Frames are sampled at the selected interval, "
                    "processed by the existing YOLO model, and saved to the existing SQLite inspection history."
                )
                live_frame = st.empty()
                live_info = st.empty()
                progress = st.progress(0, text="Starting camera monitoring")
                started_at = time.time()
                last_sample_at = 0.0
                successful = 0
                failed = []
                frame_count = 0

                while time.time() - started_at < camera_duration:
                    ok, frame_bgr = camera.read()
                    if not ok:
                        if camera_connection == "Video file (simulation)":
                            st.info("The prerecorded video reached its end.")
                        else:
                            st.warning(
                                "A video frame could not be read. The stream may have disconnected; "
                                "monitoring stopped. Check network/VPN connectivity and camera stream settings.")
                        break

                    now = time.time()
                    frame_count += 1
                    # Show a live preview even between YOLO sampling intervals.
                    preview_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    with live_frame.container():
                        st.image(preview_rgb, caption="Live camera preview",
                                 width="stretch")

                    if now - last_sample_at >= camera_sample_interval:
                        try:
                            pil_frame = Image.fromarray(preview_rgb)
                            detection, annotated = process_one(pil_frame)
                            successful += 1
                            with live_info.container():
                                a, b, c = st.columns(3)
                                a.metric("Frames saved", successful)
                                b.metric("Latest classification",
                                         detection["Type"])
                                c.metric("Track KM", detection["KM"])
                                st.caption(
                                    f"Saved frame: {detection['Frame ID']} · "
                                    f"YOLO inference: {st.session_state.latest_processing_time:.2f}s"
                                )
                        except Exception as exc:
                            failed.append(str(exc))
                            st.error(f"Frame processing failed: {exc}")
                        last_sample_at = now

                    elapsed = min(time.time() - started_at,
                                  float(camera_duration))
                    progress.progress(
                        min(elapsed / camera_duration, 1.0),
                        text=f"Camera monitoring · {int(elapsed)}/{camera_duration} seconds"
                    )
                    # Limit CPU use while keeping the preview responsive.
                    time.sleep(0.03)

                st.session_state.inspection_running = False
                st.session_state.inspection_complete = True
                st.session_state.last_run_successful = successful
                st.session_state.last_run_failed = len(failed)
                if successful:
                    st.success(
                        f"Camera monitoring finished. {successful} sampled frame(s) saved to SQLite."
                    )
                elif camera.isOpened():
                    st.warning(
                        "Monitoring ended, but no frames were successfully saved.")
                if failed:
                    with st.expander(f"Frame processing errors ({len(failed)})"):
                        for error in failed[:20]:
                            st.write(error)
        except Exception as exc:
            st.error(f"Live camera monitoring failed: {exc}")
            st.session_state.inspection_running = False
        finally:
            if camera is not None:
                camera.release()
            if 'simulation_temp_path' in locals() and simulation_temp_path:
                try:
                    Path(simulation_temp_path).unlink(missing_ok=True)
                except OSError:
                    pass
            st.session_state.inspection_running = False
        st.rerun()

    else:
        if image_source == "Existing dataset folder":
            try:
                source_items = order_images_by_sequence(
                    find_dataset_images(dataset_folder))
            except Exception as exc:
                source_items = []
                st.error(f"Cannot read dataset folder: {exc}")
        else:
            source_items = sorted(
                uploaded_files,
                key=lambda f: (
                    get_track_info(f.name)[3] if get_track_info(
                        f.name)[3] is not None else 999999,
                    f.name.lower(),
                ),
            )

        if image_source == "Existing dataset folder":
            batch_key = (
                "folder",
                str(Path(dataset_folder).expanduser().resolve()),
                tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                      for p in source_items),
            )
        else:
            batch_key = ("upload", tuple(sorted((f.name, f.size)
                         for f in uploaded_files)))

        if not source_items:
            st.error("No supported images found to inspect.")
        elif st.session_state.processed_batch == batch_key:
            st.warning(
                "This exact batch was already processed in this session. Reset the session before running it again.")
        else:
            st.session_state.processed_batch = batch_key
            st.session_state.inspection_complete = False
            st.session_state.inspection_running = True
            progress = st.progress(
                0, text=f"Preparing inspection · 0/{len(source_items)}")
            live_frame = st.empty()
            live_info = st.empty()
            failed = []
            successful = 0

            for index, source in enumerate(source_items, start=1):
                filename = Path(source).name if isinstance(
                    source, (str, Path)) else source.name
                try:
                    detection, annotated = process_one(source)
                    successful += 1
                    with live_frame.container():
                        st.image(
                            annotated, caption=f"Processing {index}/{len(source_items)} · {filename}", width="stretch")
                    with live_info.container():
                        a, b, c = st.columns(3)
                        a.metric("Frames completed",
                                 f"{index}/{len(source_items)}")
                        b.metric("Classification", detection["Type"])
                        c.metric("Track KM", detection["KM"])
                except Exception as exc:
                    failed.append((filename, str(exc)))
                    st.error(f"Failed to process {filename}: {exc}")
                progress.progress(
                    index / len(source_items), text=f"Inspection progress · {index}/{len(source_items)}")
                if delay:
                    time.sleep(delay)

            st.session_state.inspection_running = False
            st.session_state.inspection_complete = len(failed) == 0
            st.session_state.last_run_successful = successful
            st.session_state.last_run_failed = len(failed)
            if successful:
                st.success(
                    f"Inspection finished: {successful} of {len(source_items)} images processed and saved to SQLite.")
            else:
                st.error("No images were processed successfully.")
            if failed:
                with st.expander(f"Failed images ({len(failed)})"):
                    for filename, error in failed:
                        st.write(f"**{filename}** — {error}")
            st.rerun()

# ============================================================
# CURRENT SESSION METRICS
# ============================================================
counts = st.session_state.counts
latest = st.session_state.latest_detection
latest_image = st.session_state.latest_image
total_frames = st.session_state.processed
defect_total = total_frames - counts["No Problem"]
mapped_count = sum(1 for x in st.session_state.detections if x.get(
    "Latitude") is not None and x.get("Longitude") is not None)
processing_fps = 1 / \
    st.session_state.latest_processing_time if st.session_state.latest_processing_time > 0 else 0

# Compact persistent header inspired by the requested control-centre reference.
st.markdown(f"""
<div class="itms-topbar">
  <div class="itms-brand">
    <div class="itms-logo">⌁</div>
    <div><div class="itms-brand-name">ITMS <span style="font-weight:500;color:#aebdd0">RAILWAY CONTROL CENTRE</span></div>
    <div class="itms-brand-sub">Vision-Based Track Crack &amp; Gauge Detection System</div></div>
  </div>
  <div class="itms-top-status">● System Online</div>
  <div class="itms-top-user">{html.escape(current_user['username'])}<br><span style="color:#8195ad">{html.escape(current_user['role'])} · {datetime.now().strftime('%d %b %Y, %H:%M:%S')}</span></div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE — USER MANAGEMENT (ADMINISTRATOR ONLY)
# ============================================================
if page == "User Management":
    if current_user["role"] != "Administrator":
        st.error("Administrator access is required.")
        st.stop()
    section("User management", "SECURITY / ACCESS CONTROL")
    st.caption("Create named accounts and assign least-privilege roles. Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes.")
    with st.form("create_itms_user_form"):
        new_username = st.text_input("Username", key="new_user_username")
        new_password = st.text_input(
            "Password (minimum 12 characters)", type="password", key="new_user_password")
        new_role = st.selectbox(
            "Role", ["Inspector", "Maintenance", "Administrator"], key="new_user_role")
        create_submitted = st.form_submit_button(
            "Create user", width="stretch")
    if create_submitted:
        try:
            create_user(new_username, new_password, new_role)
            st.success(
                f"User '{new_username.strip()}' created with role {new_role}.")
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("That username already exists.")
        except Exception as exc:
            st.error(str(exc))

    users_df = pd.DataFrame(list_users())
    if not users_df.empty:
        st.dataframe(users_df.drop(
            columns=["id"]), width="stretch", hide_index=True)
        st.markdown("#### Enable or disable an account")
        choices = {
            f"{u['username']} · {u['role']} · {'Active' if u['active'] else 'Disabled'}": u for u in list_users()}
        chosen_label = st.selectbox("Account", list(
            choices.keys()), key="manage_user_choice")
        chosen = choices[chosen_label]
        if chosen["username"].lower() == current_user["username"].lower():
            st.info("You cannot disable the account currently signed in.")
        else:
            new_active = st.toggle("Account active", value=bool(
                chosen["active"]), key=f"user_active_{chosen['id']}")
            if st.button("Save account status", key="save_user_status"):
                set_user_active(chosen["id"], new_active)
                st.success("Account status updated.")
                st.rerun()

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
elif page == "Dashboard":
    # Dense control-centre overview: route map, selected defect, KPI row and live camera.
    map_col, detail_col = st.columns([2.35, 1], gap="small")
    with map_col:
        st.markdown('<div class="dash-panel"><div class="dash-panel-head"><span>RAILWAY TRACK MAP</span><span class="dash-small">● No Defect &nbsp; <span style="color:#ff666d">● Crack</span> &nbsp; <span style="color:#ffad45">● Spalling</span> &nbsp; <span style="color:#d7c35a">● Corrosion</span></span></div>', unsafe_allow_html=True)
        if TRACK_POINTS:
            render_map(build_map(
                latest["Latitude"] if latest else None,
                latest["Longitude"] if latest else None,
                latest["Type"] if latest else None,
            ), height=330, key="overview_route_map")
        else:
            st.info("Route map requires valid coordinates in track_locations.csv.")
        st.markdown('</div>', unsafe_allow_html=True)
    with detail_col:
        st.markdown('<div class="dash-panel"><div class="dash-panel-head"><span>SELECTED DEFECT</span><span class="dash-small">LATEST RESULT</span></div>', unsafe_allow_html=True)
        if latest is not None:
            st.markdown(
                f'<span class="pill {pill_class(latest["Type"])}">{html.escape(str(latest["Type"]))}</span> <span class="pill {priority_pill(latest["Priority"])}">{html.escape(str(latest["Priority"]))}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-label">Location</div><b>{html.escape(str(latest.get("KM", "Unknown")))}</b>', unsafe_allow_html=True)
            lat, lon = latest.get("Latitude"), latest.get("Longitude")
            coord_text = "Coordinates unavailable" if lat is None or lon is None else f"{float(lat):.6f}, {float(lon):.6f}"
            st.markdown(
                f'<div class="panel-label">GPS coordinates</div><div style="font-size:11px;color:#c2cede">{html.escape(coord_text)}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="panel-label">Frame / time</div><div style="font-size:11px;color:#c2cede">{html.escape(str(latest.get("Frame ID", "Unknown")))}<br>{html.escape(str(latest.get("Time", "Unknown")))}</div>', unsafe_allow_html=True)
            if latest_image is not None:
                st.image(latest_image,
                         caption="Latest processed frame", width="stretch")
            if latest["Type"] == "Cracks":
                st.error("Critical defect candidate — inspector review required")
            elif latest["Priority"] == "HIGH":
                st.warning("High-priority defect candidate")
            else:
                st.caption(
                    "Prediction requires inspection review before maintenance action.")
        else:
            st.info("No defect selected yet. Process images to populate this panel.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="dash-panel"><div class="dash-panel-head"><span>LIVE CAMERA FEED</span><span class="dash-small">● READY</span></div>', unsafe_allow_html=True)
        if latest_image is not None:
            st.image(
                latest_image, caption="Latest available frame · not a continuous feed", width="stretch")
        else:
            st.markdown('<div style="height:105px;display:flex;align-items:center;justify-content:center;border:1px dashed #29435f;border-radius:7px;color:#8195ad;font-size:11px">No camera frame available</div>', unsafe_allow_html=True)
        st.caption(
            "Choose Live camera in the sidebar to connect a device or RTSP stream.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">OVERVIEW STATISTICS</div>',
                unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5, gap="small")
    k1.metric("Inspections", count_all_inspections())
    k2.metric("Defect findings", defect_total)
    k3.metric("Cracks · Critical", counts["Cracks"])
    k4.metric("Spalling", counts["Spalling"])
    k5.metric("Normal track", counts["No Problem"])

    chart_col, trend_col, recent_col = st.columns([1, 1.25, 1.2], gap="small")
    with chart_col:
        st.markdown(
            '<div class="dash-panel"><div class="dash-panel-head">DEFECT DISTRIBUTION</div>', unsafe_allow_html=True)
        profile = pd.DataFrame(
            {"Detections": [counts[n] for n in CLASS_LIST]}, index=CLASS_LIST)
        st.bar_chart(profile, height=185, color="#3b91f5")
        st.markdown('</div>', unsafe_allow_html=True)
    with trend_col:
        st.markdown(
            '<div class="dash-panel"><div class="dash-panel-head">DETECTION TREND</div>', unsafe_allow_html=True)
        if st.session_state.detections:
            trend_df = pd.DataFrame(st.session_state.detections)
            if "Time" in trend_df and "Type" in trend_df:
                trend_df["Time"] = pd.to_datetime(
                    trend_df["Time"], errors="coerce")
                trend_df = trend_df.dropna(subset=["Time"]).sort_values("Time")
                if not trend_df.empty:
                    trend_pivot = trend_df.groupby(
                        [pd.Grouper(key="Time", freq="min"), "Type"]).size().unstack(fill_value=0)
                    st.line_chart(trend_pivot, height=185)
                else:
                    st.caption(
                        "Trend will appear after timestamped inspections are recorded.")
            else:
                st.caption(
                    "Trend data is not available in the current session.")
        else:
            st.caption("Process images to populate the detection trend.")
        st.markdown('</div>', unsafe_allow_html=True)
    with recent_col:
        st.markdown(
            '<div class="dash-panel"><div class="dash-panel-head">RECENT DETECTIONS</div>', unsafe_allow_html=True)
        if st.session_state.detections:
            recent_df = pd.DataFrame(st.session_state.detections).head(6)
            cols = [c for c in ["Time", "Frame ID", "Type",
                                "KM", "Priority"] if c in recent_df.columns]
            st.dataframe(recent_df[cols], width="stretch",
                         hide_index=True, height=185)
        else:
            st.caption("No inspection records found yet.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 2 — LIVE MONITOR
# ============================================================
elif page == "Live Monitoring":
    section("Live inspection monitor", "WORKSPACE / VISION FEED")
    st.caption("Select Live camera in the sidebar to sample frames from a connected camera; dataset mode remains available for offline inspection.")
    if latest is not None and latest_image is not None:
        a, b = st.columns([1.8, 1])
        with a:
            st.image(latest_image,
                     caption=latest["Frame ID"], width="stretch")
        with b:
            render_latest_details(latest)
            st.metric("Processing time",
                      f"{st.session_state.latest_processing_time:.3f} seconds")
            st.metric("Approximate FPS", f"{processing_fps:.1f}")
    else:
        st.info("No image has been processed in this session yet.")
    a, b, c = st.columns(3)
    a.metric("Records in database", count_all_inspections())
    b.metric("Findings in session", defect_total)
    c.metric("Mapped session records", mapped_count)

# ============================================================
# PAGE 3 — GALLERY
# ============================================================
elif page == "Inspection Gallery":
    section("Inspection gallery", "WORKSPACE / FRAME ARCHIVE")
    gallery = st.session_state.gallery
    if gallery:
        labels = [f'{i+1:03d} · {x["Frame ID"]} · {x["Type"]}' for i,
                  x in enumerate(gallery)]
        selected = st.selectbox("Choose a frame", labels)
        item = gallery[labels.index(selected)]
        a, b = st.columns([1.7, 1])
        with a:
            st.image(item["Image"], caption=item["Frame ID"],
                     width="stretch")
            buffer = io.BytesIO()
            Image.fromarray(item["Image"]).save(buffer, format="PNG")
            st.download_button("Download annotated PNG", buffer.getvalue(
            ), file_name=f'annotated_{Path(item["Frame ID"]).stem}.png', mime="image/png")
        with b:
            render_latest_details(item)
        st.divider()
        section("All processed frames", "VISUAL ARCHIVE")
        for start in range(0, len(gallery), 3):
            cols = st.columns(3)
            for col, frame in zip(cols, gallery[start:start+3]):
                with col:
                    st.image(frame["Image"], width="stretch")
                    st.caption(
                        f'{frame["Frame ID"]} · {frame["Type"]} · KM {frame["KM"]}')
    else:
        st.info("Annotated image previews are available for images processed in this session. Historical metadata is available under Inspection History.")

# ============================================================
# PAGE 4 — MAP
# ============================================================
elif page == "Map View":
    section("Track intelligence map", "WORKSPACE / GEOSPATIAL")
    st.markdown("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:14px 0">
      <span class="pill pill-red">● Cracks</span>
      <span class="pill pill-orange">● Spalling</span>
      <span class="pill pill-yellow">● Corrosion</span>
      <span class="pill pill-purple">● RCF</span>
      <span class="pill pill-normal">● No Problem</span>
    </div>
    """, unsafe_allow_html=True)
    try:
        history = load_inspection_history()
        if TRACK_POINTS:
            render_map(build_map(records=history),
                       height=700, key="full_route_map")
        else:
            st.warning("No valid route coordinates. Check track_locations.csv.")
    except Exception as exc:
        st.error(f"Unable to load map records: {exc}")
    st.caption(
        "Only matched, valid coordinates are plotted. Verify prototype coordinates before real-world use.")

# ============================================================
# PAGE 5 — DETECTION LOG
# ============================================================
elif page == "Detections":
    section("Detection register", "WORKSPACE / EVENT LOG")
    search_text = st.text_input(
        "Search by filename or defect class", placeholder="track_001.jpg or Cracks")
    try:
        log_df = pd.DataFrame(load_inspection_history(search_text=search_text))
        if log_df.empty:
            st.info("No matching inspection records.")
        else:
            st.dataframe(log_df, width="stretch", hide_index=True)
            st.download_button(
                "Download detection log (CSV)",
                log_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ITMS_inspection_records.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.error(f"Unable to load detection log: {exc}")

# ============================================================
# PAGE 6 — ANALYTICS
# ============================================================
elif page == "Analytics":
    section("Inspection analytics", "WORKSPACE / REPORTING")
    history = load_inspection_history()
    hist_df = pd.DataFrame(history)
    if hist_df.empty:
        st.info("No saved records to analyse.")
    else:
        saved_counts = hist_df["Type"].value_counts()
        class_counts = {name: int(saved_counts.get(name, 0))
                        for name in CLASS_LIST}
        normal = class_counts["No Problem"]
        total = len(hist_df)
        a, b, c, d = st.columns(4)
        a.metric("Saved records", total)
        b.metric("Defect classifications", total - normal)
        c.metric("Normal frames", normal)
        d.metric("GPS-mapped records",
                 int(hist_df["Latitude"].notna().sum() & hist_df["Longitude"].notna().sum()))

        left, right = st.columns(2)
        with left:
            section("Class distribution", "DETECTION MIX")
            st.bar_chart(pd.DataFrame(
                {"Count": [class_counts[n] for n in CLASS_LIST]}, index=CLASS_LIST), height=320)
        with right:
            section("Priority summary", "PROTOTYPE TRIAGE")
            priority_counts = hist_df["Priority"].value_counts()
            st.bar_chart(pd.DataFrame({"Count": [int(priority_counts.get(n, 0)) for n in [
                         "CRITICAL", "HIGH", "MODERATE", "NORMAL"]]}, index=["CRITICAL", "HIGH", "MODERATE", "NORMAL"]), height=320)

        summary_df = pd.DataFrame([
            {"Defect Class": name,
                "Frames Classified": class_counts[name], "Display Priority": priority_for(name)}
            for name in CLASS_LIST
        ])
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.download_button("Download analytics summary (CSV)", summary_df.to_csv(
            index=False).encode("utf-8-sig"), file_name="ITMS_analytics_summary.csv", mime="text/csv")
        st.caption(
            "Priority labels are prototype categories, not official railway maintenance standards.")

# ============================================================
# PAGE 7 — INSPECTION REPORTS
# ============================================================
elif page == "Reports":
    section("Inspection reports", "WORKSPACE / REPORT CENTRE")
    history = load_inspection_history()
    if not history:
        st.info("No inspection records are saved yet. Run an inspection first.")
    else:
        report_df = pd.DataFrame(history)
        normal = int((report_df["Type"] == "No Problem").sum())
        total = len(report_df)
        defects = total - normal
        mapped = int((report_df["Latitude"].notna() &
                     report_df["Longitude"].notna()).sum())
        class_counts = {
            name: int((report_df["Type"] == name).sum()) for name in CLASS_LIST}

        a, b, c, d = st.columns(4)
        a.metric("Images inspected", total)
        b.metric("Defect classifications", defects)
        c.metric("Normal classifications", normal)
        d.metric("GPS-mapped images", mapped)

        section("Defect summary", "CLASSIFICATION BREAKDOWN")
        summary_df = pd.DataFrame([
            {"Defect Class": name,
                "Images": class_counts[name], "Display Priority": priority_for(name)}
            for name in CLASS_LIST
        ])
        st.dataframe(summary_df, width="stretch", hide_index=True)

        section("Inspection records", "SAVED IMAGE-LEVEL DETAILS")
        columns = ["Time", "Frame ID", "Type", "Priority",
                   "Latitude", "Longitude", "KM", "Sequence"]
        st.dataframe(report_df[columns],
                     width="stretch", hide_index=True)

        a, b, c = st.columns(3)
        with a:
            st.download_button("Download detailed CSV", report_df[columns].to_csv(index=False).encode(
                "utf-8-sig"), file_name="ITMS_inspection_records.csv", mime="text/csv", width="stretch")
        with b:
            st.download_button("Download defect summary", summary_df.to_csv(index=False).encode(
                "utf-8-sig"), file_name="ITMS_defect_summary.csv", mime="text/csv", width="stretch")
        with c:
            try:
                pdf_bytes = create_inspection_pdf(history, class_counts)
                st.download_button("Download PDF report", pdf_bytes, file_name="ITMS_inspection_report.pdf",
                                   mime="application/pdf", width="stretch")
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
        st.caption(
            "Reports use persistent SQLite history. GPS is shown only when a valid filename match exists in track_locations.csv.")

# ============================================================
# PAGE 8 — INSPECTION HISTORY
# ============================================================
elif page == "History":
    section("Inspection history", "DATABASE / PERSISTENT ARCHIVE")
    st.caption(f"Database file: {DB_PATH}")
    search_text = st.text_input(
        "Search history", placeholder="filename, class, or priority")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From date", value=None)
    with col2:
        end_date = st.date_input("To date", value=None)

    if start_date and end_date and start_date > end_date:
        st.error("The start date must be before or equal to the end date.")
    else:
        try:
            history = load_inspection_history(
                search_text, start_date, end_date)
            hist_df = pd.DataFrame(history)
            st.metric("Matching saved records", len(hist_df))
            if hist_df.empty:
                st.info("No records match these filters.")
            else:
                st.dataframe(hist_df, width="stretch",
                             hide_index=True)
                st.download_button(
                    "Export filtered history (CSV)",
                    hist_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ITMS_filtered_history.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Unable to read history: {exc}")

    with st.expander("Danger zone — delete saved history"):
        st.warning("This permanently deletes every inspection row from the SQLite database. It does not delete the model, dataset, or CSV coordinate file.")
        confirm_delete = st.checkbox(
            "I understand that this permanently deletes all saved inspection records.")
        if st.button("Delete all saved history", type="secondary", disabled=not confirm_delete):
            delete_all_inspection_history()
            sync_session_from_database()
            st.session_state.gallery = []
            st.session_state.latest_detection = None
            st.session_state.latest_image = None
            st.success("Saved inspection history deleted.")
            st.rerun()

# ============================================================
# PAGE 9 — MAINTENANCE MANAGEMENT
# ============================================================
elif page == "Maintenance":
    section("Maintenance management", "WORKSPACE / DEFECT FOLLOW-UP")
    st.caption(
        "Create a maintenance case from saved inspection findings, assign personnel, "
        "track repair progress, and retain updates in SQLite."
    )

    inspection_history = load_inspection_history()
    eligible = [x for x in inspection_history if x.get("Type") != "No Problem"]

    with st.expander("＋ Create a maintenance case", expanded=not load_maintenance_cases()):
        if not eligible:
            st.info(
                "No defect findings are available yet. Run an inspection that identifies a defect first.")
        else:
            options = {
                f'#{x.get("Database ID")} · {x.get("Frame ID")} · {x.get("Type")} · {x.get("Priority")}': x
                for x in eligible
            }
            selected_label = st.selectbox("Select a detected defect", list(
                options.keys()), key="maintenance_source")
            selected_inspection = options[selected_label]
            st.write(
                f"**Track KM:** {selected_inspection.get('KM', 'Unknown')}  ·  "
                f"**Detected:** {selected_inspection.get('Time', 'Unknown')}"
            )
            already_open = any(
                c.get("inspection_id") == selected_inspection.get("Database ID")
                and c.get("status") != "Repair Completed"
                for c in load_maintenance_cases()
            )
            if already_open:
                st.warning(
                    "There is already an active maintenance case for this inspection record.")
            if st.button("Create maintenance case", key="create_maintenance_case", disabled=already_open):
                case_id = create_maintenance_case(selected_inspection)
                st.success(f"Maintenance case #{case_id} created and saved.")
                st.rerun()

    cases = load_maintenance_cases()
    total_cases = len(cases)
    open_cases = sum(c["status"] == "Open" for c in cases)
    under_inspection = sum(c["status"] == "Under Inspection" for c in cases)
    completed_cases = sum(c["status"] == "Repair Completed" for c in cases)

    a, b, c, d = st.columns(4)
    a.metric("Total cases", total_cases)
    b.metric("Open", open_cases)
    c.metric("Under inspection", under_inspection)
    d.metric("Repair completed", completed_cases)

    st.divider()
    section("Maintenance case register", "CASE TRACKING")
    maintenance_search = st.text_input(
        "Search cases", placeholder="filename, defect, status, assignee, or track KM",
        key="maintenance_search"
    )
    filtered_cases = load_maintenance_cases(maintenance_search)
    if not filtered_cases:
        st.info("No maintenance cases match your search.")
    else:
        register_columns = [
            "id", "created_at", "image_name", "defect_type", "priority",
            "status", "assigned_to", "track_km", "updated_at"
        ]
        register_df = pd.DataFrame(filtered_cases)
        st.dataframe(
            register_df[register_columns].rename(columns={
                "id": "Case ID", "created_at": "Created", "image_name": "Image",
                "defect_type": "Defect", "priority": "Priority", "status": "Status",
                "assigned_to": "Assigned To", "track_km": "Track KM", "updated_at": "Last Updated"
            }),
            width="stretch", hide_index=True
        )

        case_labels = {
            f'#{c["id"]} · {c["image_name"]} · {c["defect_type"]} · {c["status"]}': c
            for c in filtered_cases
        }
        selected_case_label = st.selectbox(
            "Select a case to update", list(case_labels.keys()), key="maintenance_edit_select"
        )
        selected_case = case_labels[selected_case_label]

        st.markdown("#### Update selected case")
        st.write(
            f"**Defect:** {selected_case['defect_type']}  ·  "
            f"**Priority:** {selected_case['priority']}  ·  "
            f"**Track KM:** {selected_case.get('track_km') or 'Unknown'}"
        )
        with st.form(f"maintenance_update_form_{selected_case['id']}"):
            status_options = ["Open", "Under Inspection", "Repair Completed"]
            status_index = status_options.index(
                selected_case["status"]) if selected_case["status"] in status_options else 0
            updated_status = st.selectbox(
                "Status", status_options, index=status_index)
            assigned_to = st.text_input(
                "Assigned personnel / team", value=selected_case.get("assigned_to") or "")
            notes = st.text_area("Maintenance notes",
                                 value=selected_case.get("notes") or "")
            repair_notes = st.text_area(
                "Repair completion notes", value=selected_case.get("repair_notes") or "")
            save_case = st.form_submit_button("Save case updates")
        if save_case:
            update_maintenance_case(
                selected_case["id"], updated_status, assigned_to, notes, repair_notes
            )
            st.success(f"Maintenance case #{selected_case['id']} updated.")
            st.rerun()

        with st.expander("Delete a maintenance case"):
            st.warning(
                "Deleting a case removes only its maintenance record. The original inspection record remains in the database.")
            confirm_case_delete = st.checkbox(
                f"Confirm deletion of maintenance case #{selected_case['id']}",
                key=f"confirm_delete_case_{selected_case['id']}"
            )
            if st.button(
                "Delete selected maintenance case",
                key=f"delete_case_{selected_case['id']}",
                disabled=not confirm_case_delete
            ):
                delete_maintenance_case(selected_case["id"])
                st.success("Maintenance case deleted.")
                st.rerun()

        st.divider()
        section("Export maintenance records", "REPORT CENTRE")
        export_df = pd.DataFrame(filtered_cases)
        csv_columns = [
            "id", "created_at", "updated_at", "inspection_id", "image_name",
            "defect_type", "priority", "latitude", "longitude", "track_km",
            "status", "assigned_to", "notes", "repair_notes"
        ]
        export_df = export_df[csv_columns]
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download maintenance CSV",
                export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="ITMS_maintenance_cases.csv",
                mime="text/csv",
                width="stretch"
            )
        with right:
            try:
                pdf_bytes = create_maintenance_pdf(filtered_cases)
                st.download_button(
                    "Download maintenance PDF",
                    pdf_bytes,
                    file_name="ITMS_maintenance_report.pdf",
                    mime="application/pdf",
                    width="stretch"
                )
            except Exception as exc:
                st.error(f"Maintenance PDF generation failed: {exc}")

    st.caption(
        "Maintenance cases are stored separately from inspection history in the same SQLite database. "
        "Changing or deleting a maintenance case does not alter the original YOLO inspection record."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid #1e3548;padding-top:14px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">
  <span class="small-note">ITMS · Vision-Based Contactless Track Crack &amp; Gauge Detection System</span>
  <span class="small-note">YOLO Model-2 · SQLite persistent inspection history</span>
</div>
""", unsafe_allow_html=True)
