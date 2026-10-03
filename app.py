"""
Golden Steps CBO – GBV Case Management Dashboard
================================================
Streamlit dashboard fed live from KoboToolbox (API v2).

Covers the form's components:
  C1  Cumulative outreach data (SRHR, pad distribution, GBV enrolments)
  C2  GBV survivor case records (coded / de-identified)
  C3  Accompaniment & follow-up
  C4  Multi-agency meetings
  C5  Police station / CPU support
  C6  GBV help-desk

SETUP (3 values to edit – see CONFIG below, or use .streamlit/secrets.toml):
  KOBO_SERVER     e.g. https://kf.kobotoolbox.org  or  https://eu.kobotoolbox.org
  KOBO_API_TOKEN  Kobo > Account Settings > Security > API key
  KOBO_ASSET_UID  the form ID (the string after /forms/ in the project URL)

RUN:  streamlit run app.py

PRIVACY: this dashboard shows CODED data only. Set APP_PASSWORD in secrets,
deploy privately, and never put names / phone numbers in the Kobo form.
"""
import hmac
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import base64
import io
from PIL import Image

# =============================================================================
# CONFIG  –  EDIT HERE (or set the same keys in .streamlit/secrets.toml)
# =============================================================================


# Private repos / local use ONLY: you may paste the token between the quotes as a last resort.
# Never do this if the code is in a public GitHub repo.
KOBO_API_TOKEN_FALLBACK = ""


def _all_secrets():
    """Flatten st.secrets (incl. one level of [sections]) into {lowercase_key: value}."""
    flat = {}
    try:
        for k, v in st.secrets.items():
            if hasattr(v, "items"):
                for k2, v2 in v.items():
                    flat[str(k2).lower()] = v2
            else:
                flat[str(k).lower()] = v
    except Exception:
        pass
    return flat


def _secret(key, default):
    val = _all_secrets().get(key.lower())
    if val in (None, ""):
        val = os.environ.get(key, default)
    return val.strip().strip("\"'") if isinstance(val, str) else val


KOBO_SERVER = _secret("KOBO_SERVER", "https://kf.kobotoolbox.org")  # <-- your Kobo server
KOBO_API_TOKEN = _secret("KOBO_API_TOKEN", KOBO_API_TOKEN_FALLBACK)  # API key: put in .streamlit/secrets.toml, NOT in this file
KOBO_ASSET_UID = _secret("KOBO_ASSET_UID", "aJ5SsJRgzQw6UtpNtH63V2")  # form ID (not secret)
KOBO_FALLBACK_SERVERS = ["https://kf.kobotoolbox.org", "https://eu.kobotoolbox.org",
                         "https://kobo.humanitarianresponse.info"]  # tried automatically if the first fails
APP_PASSWORD_FALLBACK = ""  # access code used if no APP_PASSWORD secret is set (change it!)
APP_PASSWORD = _secret("APP_PASSWORD", APP_PASSWORD_FALLBACK)  # optional access code (recommended)

CASE_ID_PATTERN = r"^NYA-[A-Z]-\d{3}$"  # coded Case ID format: NYA-[Village initial]-[###]
TZ = "Africa/Nairobi"
CACHE_SECONDS = 300  # how long API data is cached before re-pulling

# Optional manual overrides if automatic mapping misses a field.
# Format: {"canonical_key": "raw_kobo_field_path"}, e.g. {"risk": "grp_c2/risk_level"}
API_NAME_OVERRIDES = {}

# Canonical field -> (question label in the Kobo form, occurrence among identical labels)
# Labels must match the form exactly (including the "Numebr" typo in the form).
FIELDS = {
    # ---- C1: cumulative outreach ----
    "c1_activity": ("Activity type", 0),
    "c1_date": ("Date", 0),
    "c1_location": ("Indicate location of activity", 0),
    "c1_seen": ("Numebr of individuals seen", 0),
    "c1_srh_counseled": ("Number Counseled on SRH", 0),
    "c1_new": ("Number of New GBV cases received", 0),
    "c1_followup": ("Number of Follow-up cases seen", 0),
    "c1_referrals": ("Number of Referrals made this day", 0),
    "c1_services": ("SRH services offered", 0),
    "c1_pads": ("Number of Sanitary pads distributed", 0),
    "c1_notes": ("Notes", 0),
    # ---- C2: survivor case records ----
    "case_id": ("Case ID (coded)", 0),
    "intake_date": ("Intake date", 0),
    "village": ("Village Unit", 0),
    "age": ("Age of survivor", 0),
    "age_bracket": ("Age bracket", 0),
    "sex": ("Sex", 0),
    "violence": ("Violence type (broad category)", 0),
    "risk": ("Risk level", 0),
    "consent": ("Informed consent obtained?", 0),
    "ref_health": ("Referred – Health", 0),
    "ref_health_date": ("Health referral date", 0),
    "ref_police": ("Referred – Police", 0),
    "ref_police_date": ("Police referral date", 0),
    "ref_legal": ("Referred – Legal Aid", 0),
    "ref_legal_date": ("Legal referral date", 0),
    "ref_psy": ("Referred – Psychosocial", 0),
    "ref_psy_date": ("Psychosocial referral date", 0),
    "ref_cps": ("Referred – Children's Protection Services", 0),
    "ref_cps_date": ("CPS referral date", 0),
    "ref_status": ("Referral completion status", 0),
    "officer": ("Case officer", 0),
    # ---- C3: accompaniment & follow-up ----
    "acc_case_id": ("Case ID (coded)", 1),
    "acc_date": ("Accompaniment date", 0),
    "acc_type": ("Accompaniment type", 0),
    "acc_by": ("Accompanied by", 0),
    "next_fu": ("Next follow-up date", 0),
    # ---- C4: multi-agency meetings ----
    "mtg_no": ("Meeting #", 0),
    "mtg_date": ("Date", 1),
    "mtg_agencies": ("Attendees / agencies represented", 0),
    "mtg_cases": ("# Cases reviewed", 0),
    "mtg_decisions": ("Key decisions", 0),
    "mtg_actions": ("Action tracker (owner + due date)", 0),
    "mtg_next": ("Next meeting date", 0),
    # ---- C5: police station / CPU ----
    "pol_station": ("Police station / CPU", 0),
    "pol_date": ("Activity date", 0),
    "pol_type": ("Activity type", 1),
    "pol_trained": ("# Officers trained", 0),
    "pol_equipment": ("Equipment / support provided", 0),
    "pol_base": ("Functionality status (baseline)", 0),
    "pol_curr": ("Functionality status (current)", 0),
    # ---- C6: help desk ----
    "desk_loc": ("Desk location", 0),
    "desk_officer": ("Case officer on duty", 0),
    "desk_hours": ("Hours open", 0),
}

DATE_KEYS = ["c1_date", "intake_date", "ref_health_date", "ref_police_date", "ref_legal_date",
             "ref_psy_date", "ref_cps_date", "acc_date", "next_fu", "mtg_date", "mtg_next", "pol_date"]
NUM_KEYS = ["c1_seen", "c1_srh_counseled", "c1_new", "c1_followup", "c1_referrals", "c1_pads",
            "age", "mtg_no", "mtg_cases", "pol_trained"]
YN_KEYS = ["consent", "ref_health", "ref_police", "ref_legal", "ref_psy", "ref_cps"]
LONG_TEXT = ["c1_notes", "mtg_decisions", "mtg_actions", "mtg_agencies", "pol_equipment"]
META_KEYS = ["_id", "_submission_time", "_submitted_by", "_validation_status"]

PATHWAYS = [
    ("Health", "ref_health", "ref_health_date"),
    ("Police", "ref_police", "ref_police_date"),
    ("Legal aid", "ref_legal", "ref_legal_date"),
    ("Psychosocial", "ref_psy", "ref_psy_date"),
    ("Children's Protection (CPS)", "ref_cps", "ref_cps_date"),
]
SRH_SERVICES = [
    ("Family planning counselling", r"family|fp"),
    ("STI screening / treatment", r"\bsti\b"),
    ("Pregnancy testing", r"pregnan"),
    ("HIV testing", r"hiv"),
    ("PEP", r"\bpep\b|post"),
    ("Emergency contraception", r"emergency"),
    ("Referral to health facility", r"referral"),
    ("Other", r"\bother\b"),
]

PALETTE = ["#B68334", "#4B2E83", "#1F8A8A", "#D1495B", "#7A9E3B", "#8E7DBE", "#E08E45", "#5B7DB1"]
RISK_COLORS = {"High": "#D1495B", "Medium": "#C99A2E", "Low": "#1F8A8A", "Not recorded": "#9AA0A6"}
px.defaults.color_discrete_sequence = PALETTE

# Golden Steps CBO logo (embedded so the app is a single self-contained file)
LOGO_B64 = "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"
LOGO_IMG = Image.open(io.BytesIO(base64.b64decode(LOGO_B64)))

# =============================================================================
# PAGE + STYLE
# =============================================================================
st.set_page_config(page_title="Golden Steps CBO · GBV Case Management", page_icon=LOGO_IMG, layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], .stDeployButton, [class*="viewerBadge"], [class*="_profileContainer"],
a[href*="github.com"], a[href*="streamlit.io"] {display: none !important; visibility: hidden !important;}
.kpi {border-left: 5px solid #B68334; background: rgba(127,127,127,0.09);
      border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; min-height: 96px;}
.kpi-l {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .75;}
.kpi-v {font-size: 1.9rem; font-weight: 700; line-height: 1.25; color: #4B2E83;}
.kpi-s {font-size: 0.78rem; opacity: .7;}
@media (prefers-color-scheme: dark) {.kpi-v {color: #E8B766;}}
.small-note {font-size: 0.82rem; opacity: .75;}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# ACCESS CONTROL
# =============================================================================
def gate():
    if not APP_PASSWORD:
        return
    if st.session_state.get("auth"):
        if st.sidebar.button("🔒 Log out"):
            st.session_state["auth"] = False
            st.rerun()
        return
    _l, _m, _r = st.columns([1, 1, 1])
    _m.image(LOGO_IMG, width=170)
    st.title("Golden Steps CBO")
    st.caption("Restricted – authorised case officers and supervisors only.")
    if st.session_state.get("fails", 0) >= 5:
        st.error("Too many incorrect attempts. Close this tab and try again later.")
        st.stop()
    pw = st.text_input("Access code", type="password")
    if pw:
        if hmac.compare_digest(str(pw).encode(), str(APP_PASSWORD).encode()):
            st.session_state["auth"] = True
            st.session_state["fails"] = 0
            st.rerun()
        else:
            st.session_state["fails"] = st.session_state.get("fails", 0) + 1
            st.error("Incorrect access code.")
    st.stop()


gate()


# =============================================================================
# DATA LAYER
# =============================================================================
def _label_of(row, tr_idx):
    lab = row.get("label")
    if isinstance(lab, list):
        pick = lab[tr_idx] if len(lab) > tr_idx and lab[tr_idx] else next((x for x in lab if x), None)
        lab = pick
    return lab.strip() if isinstance(lab, str) else None


def schema_from_asset(asset):
    """Pull question paths/labels/choices out of the Kobo asset so labels map to API field names."""
    content = asset.get("content", {}) or {}
    tr_idx = 0
    for i, t in enumerate(content.get("translations") or []):
        if t and "english" in str(t).lower():
            tr_idx = i
            break
    stack, questions = [], []
    for row in content.get("survey", []):
        t = str(row.get("type", ""))
        name = row.get("name") or row.get("$autoname")
        if t in ("begin_group", "begin_repeat"):
            stack.append(name)
            continue
        if t in ("end_group", "end_repeat"):
            if stack:
                stack.pop()
            continue
        if not name:
            continue
        qtype, list_name = t, row.get("select_from_list_name")
        if t.startswith(("select_one", "select_multiple")):
            parts = t.split()
            qtype = "select_multiple" if t.startswith("select_multiple") else "select_one"
            list_name = list_name or (parts[1] if len(parts) > 1 else None)
        questions.append({"path": "/".join([s for s in stack if s] + [name]),
                          "label": _label_of(row, tr_idx), "type": qtype, "list": list_name})
    choices = {}
    for c in content.get("choices", []):
        nm = c.get("name") or c.get("$autovalue")
        choices.setdefault(c.get("list_name"), {})[nm] = _label_of(c, tr_idx) or nm
    return {"questions": questions, "choices": choices}


@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Pulling latest data from KoboToolbox…")
def fetch_kobo(servers, token, uid):
    """Try each server in turn. Returns (data, schema, server_used, schema_error)."""
    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    errors = []
    for base in servers:
        base = base.rstrip("/")
        try:
            schema, schema_err = None, None
            try:
                a = requests.get(f"{base}/api/v2/assets/{uid}/", params={"format": "json"}, headers=headers, timeout=60)
                a.raise_for_status()
                schema = schema_from_asset(a.json())
            except Exception as e:  # noqa: BLE001  (form structure is nice-to-have; data is essential)
                schema_err = f"{type(e).__name__}: {str(e)[:200]}"
            rows, url, params = [], f"{base}/api/v2/assets/{uid}/data/", {"format": "json", "limit": 3000}
            while url:
                r = requests.get(url, headers=headers, params=params, timeout=120)
                r.raise_for_status()
                j = r.json()
                rows.extend(j.get("results", []))
                url, params = j.get("next"), None
            return pd.DataFrame(rows), schema, base, schema_err
        except requests.HTTPError as e:
            body = (e.response.text or "")[:200].replace("\n", " ") if e.response is not None else ""
            code = e.response.status_code if e.response is not None else "?"
            errors.append(f"{base}  ->  HTTP {code} {body}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{base}  ->  {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("\n".join(errors))


def canonicalize(raw, schema=None):
    """Map the raw Kobo columns to canonical field names (labels -> API paths via schema, or header text)."""
    cols = list(raw.columns)
    mapping, unmapped, qinfo = {}, [], {}
    for key, (label, occ) in FIELDS.items():
        col = API_NAME_OVERRIDES.get(key)
        if col is None and schema:
            qs = [q for q in schema["questions"] if q["label"] and q["label"].strip() == label]
            if len(qs) > occ:
                col, qinfo[key] = qs[occ]["path"], qs[occ]
        if col is None or col not in raw.columns:
            pat = re.compile(rf"(^|/){re.escape(label)}(\.\d+)?$")
            hits = [c for c in cols if pat.search(str(c).strip())]
            col = hits[occ] if len(hits) > occ else None
        if col is not None and col in raw.columns:
            mapping[key] = col
        else:
            unmapped.append(key)

    out = pd.DataFrame(index=raw.index)
    for key in FIELDS:
        out[key] = raw[mapping[key]] if key in mapping else np.nan
    # API returns choice *names*; translate to labels using the form schema
    for key, q in qinfo.items():
        lk = schema["choices"].get(q["list"], {}) if schema else {}
        if not lk:
            continue
        if q["type"] == "select_one":
            out[key] = out[key].map(lambda v, lk=lk: lk.get(v, v) if isinstance(v, str) else v)
        elif q["type"] == "select_multiple":
            out[key] = out[key].map(
                lambda v, lk=lk: " | ".join(lk.get(t, t) for t in v.split()) if isinstance(v, str) else v)
    for m in META_KEYS:
        out[m] = raw[m] if m in raw.columns else np.nan
    return out, unmapped


def norm_yn(s):
    low = s.astype("object").map(lambda v: str(v).strip().lower() if pd.notna(v) else "")
    out = pd.Series(np.nan, index=s.index, dtype="object")
    out[low.isin(["yes", "y", "1", "true"])] = "Yes"
    out[low.isin(["no", "n", "0", "false"])] = "No"
    return out


def prepare(df):
    df = df.copy()
    for k in DATE_KEYS:
        df[k] = pd.to_datetime(df[k], errors="coerce").dt.normalize()
    for k in NUM_KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    for k in YN_KEYS:
        df[k] = norm_yn(df[k])
    skip = set(DATE_KEYS + NUM_KEYS + YN_KEYS + META_KEYS)
    for k in FIELDS:
        if k in skip:
            continue
        df[k] = df[k].astype("object").map(lambda v: v.strip() if isinstance(v, str) else v).replace("", np.nan)
        if k not in LONG_TEXT:
            df[k] = df[k].map(lambda v: re.sub(r"\s+", " ", v) if isinstance(v, str) else v)
        df[k] = df[k].astype("object")  # keep text type even when a column is entirely empty
    df["risk"] = df["risk"].map(lambda v: v.title() if isinstance(v, str) else v).astype("object")
    df["cid"] = df["case_id"].map(lambda v: v.upper() if isinstance(v, str) else v).astype("object")
    df["acc_cid"] = df["acc_case_id"].map(lambda v: v.upper() if isinstance(v, str) else v).astype("object")
    sub = pd.to_datetime(df["_submission_time"], errors="coerce", utc=True)
    df["sub_date"] = sub.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    return df


def split_sections(df):
    any_ = lambda cols: df[cols].notna().any(axis=1)
    m1 = any_(["c1_date", "c1_seen", "c1_srh_counseled", "c1_new", "c1_followup", "c1_referrals", "c1_pads", "c1_location"])
    m2 = any_(["intake_date", "violence", "risk", "age", "consent"])
    m3 = any_(["acc_case_id", "acc_date", "acc_type", "acc_by"])
    m4 = any_(["mtg_no", "mtg_date", "mtg_cases", "mtg_agencies"])
    m5 = any_(["pol_station", "pol_date", "pol_trained"])
    m6 = any_(["desk_loc", "desk_officer", "desk_hours"])
    S = {}
    for name, m, dcol in [("c1", m1, "c1_date"), ("c2", m2, "intake_date"), ("c3", m3, "acc_date"),
                          ("c4", m4, "mtg_date"), ("c5", m5, "pol_date"), ("c6", m6, None)]:
        d = df[m].copy()
        d["dt"] = d[dcol].fillna(d["sub_date"]) if dcol else d["sub_date"]
        S[name] = d
    S["blank"] = df[~(m1 | m2 | m3 | m4 | m5 | m6)].copy()
    return S


# =============================================================================
# HELPERS
# =============================================================================
_k = {"n": 0}


def plot(fig, h=340):
    _k["n"] += 1
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=45, b=10), title_font_size=15,
                      legend=dict(orientation="h", y=-0.22, title_text=""),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, key=f"chart{_k['n']}")


def show_df(d, **kw):
    try:
        st.dataframe(d, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(d, use_container_width=True, hide_index=True, **kw)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="kpi-l">{label}</div><div class="kpi-v">{value}</div>'
                 f'<div class="kpi-s">{sub}</div></div>', unsafe_allow_html=True)


def n(x):
    return "0" if pd.isna(x) else f"{int(x):,}"


def pct(a, b):
    return "—" if not b else f"{100 * a / b:.0f}%"


def cat(s):
    return s.fillna("Not recorded")


def vc(s, name="Count"):
    return cat(s).value_counts().rename_axis("label").reset_index(name=name)


def bucket(s, gran):
    return s.dt.to_period("M").dt.to_timestamp() if gran == "Month" else s.dt.to_period("W-SUN").dt.start_time


def age_sort_key(v):
    m = re.search(r"\d+", str(v))
    return (0 if str(v).lower().startswith("under") else 1, int(m.group()) if m else 999, str(v))


def is_complete(s):
    return s.astype("object").fillna("").astype(str).str.lower().str.startswith("complete")


def func_rank(v):
    if not isinstance(v, str):
        return np.nan
    x = v.lower()
    if "non" in x or x.startswith("not"):
        return 0
    if "partial" in x:
        return 1
    if "full" in x or "functional" in x:
        return 2
    return np.nan


def empty(d, msg="No records for the selected filters yet."):
    if len(d) == 0:
        st.info(msg)
        return True
    return False


def followup_table(c2, c3_all, today):
    cols = ["cid", "village", "risk", "violence", "ref_status", "officer", "intake_date"]
    base = c2[cols].dropna(subset=["cid"]).drop_duplicates("cid")
    last = (c3_all.dropna(subset=["acc_cid"]).sort_values("dt").groupby("acc_cid").tail(1)
            [["acc_cid", "dt", "acc_type", "next_fu"]].rename(columns={"acc_cid": "cid", "dt": "last_acc"}))
    t = base.merge(last, on="cid", how="left")

    def status(r):
        if pd.isna(r["last_acc"]):
            return "No follow-up logged"
        if pd.isna(r["next_fu"]):
            return "No next date set"
        return "Overdue" if r["next_fu"] < today else "Scheduled"

    t["fu_status"] = t.apply(status, axis=1) if len(t) else pd.Series(dtype=object)
    t["days_overdue"] = np.where(t["fu_status"] == "Overdue", (today - t["next_fu"]).dt.days, np.nan)
    return t


# =============================================================================
# SIDEBAR – DATA SOURCE
# =============================================================================
st.sidebar.image(LOGO_IMG, width=130)
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

if not KOBO_API_TOKEN or not KOBO_ASSET_UID or "YOUR_" in str(KOBO_ASSET_UID):
    st.error("**KoboToolbox is not configured** – the app cannot see `KOBO_API_TOKEN`.")
    try:
        _err = None
        _names = sorted(_all_secrets().keys())
    except Exception as _e:  # noqa: BLE001
        _err, _names = str(_e), []
    st.markdown("**Diagnostics (names only, no values):**")
    st.code(f"secret keys the app can see : {_names or 'NONE'}\n"
            f"KOBO_API_TOKEN found        : {bool(KOBO_API_TOKEN)}\n"
            f"KOBO_ASSET_UID found        : {bool(KOBO_ASSET_UID)}\n"
            f"server                      : {KOBO_SERVER}")
    st.markdown(
        "- **NONE** → the secrets file is not being read. On Streamlit Cloud paste it into *Manage app → Settings → Secrets* "
        "(the local `.streamlit/secrets.toml` is not uploaded). Locally, run `streamlit run app.py` from the folder that contains `.streamlit`.\n"
        "- Keys listed but not `kobo_api_token` → the name is misspelt.\n"
        "- Every value must be in quotes, e.g. `KOBO_API_TOKEN = \"abc123\"`; an unquoted value makes the whole secrets file fail to load.")
    st.stop()

servers = tuple(dict.fromkeys([str(KOBO_SERVER).rstrip("/")] + KOBO_FALLBACK_SERVERS))
try:
    raw, schema, used_server, schema_err = fetch_kobo(servers, str(KOBO_API_TOKEN).strip(), str(KOBO_ASSET_UID).strip())
except Exception as e:  # noqa: BLE001
    st.error("Could not load data from KoboToolbox. Details per server:")
    st.code(str(e))
    st.caption("401/403 = wrong or revoked API token, or the token's account has no access to this form · "
               "404 = wrong form ID or wrong server · other = network problem.")
    st.stop()
if schema_err:
    st.warning("Could not read the form structure from Kobo, so some fields may not map correctly "
               f"({schema_err}).")

if raw is None or raw.empty:
    st.info("The form has no submissions yet.")
    st.stop()

df, unmapped = canonicalize(raw, schema)
df = prepare(df)
S = split_sections(df)
today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()

with st.sidebar.expander("Field mapping / troubleshooting"):
    st.caption(f"Server: {used_server}")
    st.caption(f"{len(raw):,} submissions loaded · {len(FIELDS) - len(unmapped)}/{len(FIELDS)} fields mapped.")
    if unmapped:
        st.caption("No matching column for (not in form, or no data yet): " + ", ".join(unmapped))
    st.caption("If a field maps wrongly, add it to `API_NAME_OVERRIDES` in app.py.")
    if st.checkbox("Show raw column names"):
        st.write(list(raw.columns))

# =============================================================================
# SIDEBAR – FILTERS
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
alld = pd.concat([S[k]["dt"] for k in ("c1", "c2", "c3", "c4", "c5")]).dropna()
dmin = alld.min() if len(alld) else today - pd.Timedelta(days=90)
dmax = max(alld.max(), today) if len(alld) else today
period = st.sidebar.selectbox("Period", ["All time", "Last 30 days", "Last 90 days", "This year", "Custom"])
if period == "Last 30 days":
    start, end = today - pd.Timedelta(days=30), dmax
elif period == "Last 90 days":
    start, end = today - pd.Timedelta(days=90), dmax
elif period == "This year":
    start, end = pd.Timestamp(today.year, 1, 1), dmax
elif period == "Custom":
    dr = st.sidebar.date_input("Dates", value=(dmin.date(), dmax.date()))
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start, end = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
    else:
        start, end = dmin, dmax
else:
    start, end = dmin, dmax
gran = st.sidebar.radio("Trend granularity", ["Month", "Week"], horizontal=True)

c2_all = S["c2"]
st.sidebar.caption("Case filters (apply to survivor / follow-up views)")
f_vil = st.sidebar.multiselect("Village unit", sorted(c2_all["village"].dropna().unique()))
f_off = st.sidebar.multiselect("Case officer", sorted(c2_all["officer"].dropna().unique()))
f_risk = st.sidebar.multiselect("Risk level", sorted(c2_all["risk"].dropna().unique()))
case_filters_on = bool(f_vil or f_off or f_risk)


def in_period(d):
    return d[(d["dt"] >= start) & (d["dt"] <= end)]


c1 = in_period(S["c1"])
c2 = in_period(S["c2"])
if f_vil:
    c2 = c2[c2["village"].isin(f_vil)]
if f_off:
    c2 = c2[c2["officer"].isin(f_off)]
if f_risk:
    c2 = c2[c2["risk"].isin(f_risk)]
c3_all = S["c3"]
c3 = in_period(c3_all)
if case_filters_on:
    c3 = c3[c3["acc_cid"].isin(c2["cid"])]
c4, c5, c6 = in_period(S["c4"]), in_period(S["c5"]), S["c6"]

tracker = followup_table(c2, c3_all, today)
needs_fu = tracker["fu_status"].isin(["No follow-up logged", "Overdue", "No next date set"]) if len(tracker) else pd.Series(dtype=bool)

# =============================================================================
# HEADER
# =============================================================================
_hl, _hr = st.columns([1, 8], vertical_alignment="center")
_hl.image(LOGO_IMG, width=95)
_hr.title("GBV Case Management Dashboard")
st.caption(f"Golden Steps CBO · Component C – Case Management & Referral · "
           f"{start:%d %b %Y} → {end:%d %b %Y} · refreshed {datetime.now():%d %b %Y %H:%M}")
if not APP_PASSWORD:
    st.sidebar.warning("No APP_PASSWORD set – anyone with the link can view this dashboard.")

tabs = st.tabs(["📊 Overview", "🌱 Outreach & SRHR", "🛡️ Survivor cases", "🤝 Follow-up & accompaniment",
                "🏛️ Coordination & systems", "✅ Data quality", "🗂️ Data"])

# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
with tabs[0]:
    r1 = st.columns(4)
    kpi(r1[0], "People reached (outreach)", n(c1["c1_seen"].sum()), f"{len(c1)} outreach sessions")
    kpi(r1[1], "Counselled on SRH", n(c1["c1_srh_counseled"].sum()), pct(c1["c1_srh_counseled"].sum(), c1["c1_seen"].sum()) + " of reached")
    kpi(r1[2], "Sanitary pads distributed", n(c1["c1_pads"].sum()))
    kpi(r1[3], "Survivors enrolled (case records)", n(len(c2)), f"{n(c1['c1_new'].sum())} new cases reported in outreach logs")
    r2 = st.columns(4)
    hi = (c2["risk"] == "High").sum()
    kpi(r2[0], "High-risk cases", n(hi), pct(hi, len(c2)) + " of enrolled")
    kpi(r2[1], "Informed consent obtained", pct((c2["consent"] == "Yes").sum(), c2["consent"].notna().sum()), "of cases with consent recorded")
    kpi(r2[2], "Referrals fully completed", pct(is_complete(c2["ref_status"]).sum(), len(c2)), "of enrolled cases")
    kpi(r2[3], "Cases needing follow-up", n(needs_fu.sum()) if len(tracker) else "0", "overdue, none logged, or no next date")

    a, b = st.columns(2)
    with a:
        if not empty(c1):
            t = c1.assign(period=bucket(c1["dt"], gran)).groupby("period")[["c1_seen", "c1_srh_counseled"]].sum().reset_index()
            t = t.rename(columns={"c1_seen": "Reached", "c1_srh_counseled": "Counselled on SRH"}).melt("period")
            plot(px.bar(t, x="period", y="value", color="variable", barmode="group",
                        title=f"Outreach reach by {gran.lower()}", labels={"period": "", "value": "People", "variable": ""}))
    with b:
        if not empty(c2):
            t = c2.assign(period=bucket(c2["dt"], gran), risk=cat(c2["risk"])).groupby(["period", "risk"]).size().reset_index(name="Cases")
            plot(px.bar(t, x="period", y="Cases", color="risk", color_discrete_map=RISK_COLORS,
                        title=f"New survivor enrolments by {gran.lower()} and risk", labels={"period": "", "risk": ""}))

    st.subheader("🚩 Priority watchlist – high-risk cases needing action")
    if len(tracker):
        wl = tracker[(tracker["risk"] == "High") & (needs_fu | ~is_complete(tracker["ref_status"]))].copy()
        if len(wl):
            wl["Issue"] = np.where(needs_fu.loc[wl.index], wl["fu_status"], "Referrals incomplete")
            wl = wl.sort_values("days_overdue", ascending=False)
            show_df(wl.rename(columns={"cid": "Case ID", "village": "Village", "violence": "Violence type",
                                       "ref_status": "Referral status", "officer": "Case officer",
                                       "next_fu": "Next follow-up", "days_overdue": "Days overdue"})
                    [["Case ID", "Village", "Violence type", "Referral status", "Issue", "Next follow-up", "Days overdue", "Case officer"]])
        else:
            st.success("No high-risk cases are flagged.")
    else:
        st.info("No survivor case records yet.")

# -----------------------------------------------------------------------------
# OUTREACH & SRHR
# -----------------------------------------------------------------------------
with tabs[1]:
    if not empty(c1, "No outreach sessions in the selected period."):
        k = st.columns(5)
        kpi(k[0], "Sessions", n(len(c1)))
        kpi(k[1], "People reached", n(c1["c1_seen"].sum()))
        kpi(k[2], "New GBV cases (reported)", n(c1["c1_new"].sum()))
        kpi(k[3], "Follow-up cases seen", n(c1["c1_followup"].sum()))
        kpi(k[4], "Referrals made", n(c1["c1_referrals"].sum()))

        a, b = st.columns(2)
        with a:
            t = c1.assign(period=bucket(c1["dt"], gran)).groupby("period")[["c1_new", "c1_followup", "c1_referrals"]].sum().reset_index()
            t = t.rename(columns={"c1_new": "New GBV cases", "c1_followup": "Follow-ups", "c1_referrals": "Referrals"}).melt("period")
            plot(px.line(t, x="period", y="value", color="variable", markers=True,
                         title="GBV activity reported at outreach", labels={"period": "", "value": "Count", "variable": ""}))
        with b:
            t = c1.assign(period=bucket(c1["dt"], gran)).groupby("period")["c1_pads"].sum().reset_index()
            plot(px.bar(t, x="period", y="c1_pads", title="Sanitary pads distributed", labels={"period": "", "c1_pads": "Pads"},
                        color_discrete_sequence=[PALETTE[1]]))
        a, b = st.columns(2)
        with a:
            loc = c1.assign(loc=cat(c1["c1_location"])).groupby("loc")["c1_seen"].sum().sort_values().tail(15).reset_index()
            plot(px.bar(loc, x="c1_seen", y="loc", orientation="h", title="People reached by location",
                        labels={"c1_seen": "People", "loc": ""}))
        with b:
            txt = c1["c1_services"].fillna("").str.lower()
            sv = pd.DataFrame({"Service": [s for s, _ in SRH_SERVICES],
                               "Sessions": [int(txt.str.contains(p, regex=True).sum()) for _, p in SRH_SERVICES]})
            plot(px.bar(sv.sort_values("Sessions"), x="Sessions", y="Service", orientation="h",
                        title="SRH services offered (sessions)", labels={"Service": ""}, color_discrete_sequence=[PALETTE[2]]))

        st.subheader("Outreach log")
        show_notes = st.checkbox("Show narrative session notes (must remain de-identified)", value=False)
        log = c1.sort_values("dt", ascending=False)
        cols = {"dt": "Date", "c1_location": "Location", "c1_seen": "Seen", "c1_srh_counseled": "SRH counselled",
                "c1_new": "New GBV", "c1_followup": "Follow-up", "c1_referrals": "Referrals", "c1_pads": "Pads",
                "c1_services": "Services"}
        if show_notes:
            cols["c1_notes"] = "Notes"
        show_df(log[list(cols)].rename(columns=cols))

# -----------------------------------------------------------------------------
# SURVIVOR CASES
# -----------------------------------------------------------------------------
with tabs[2]:
    if not empty(c2, "No survivor case records for the selected filters."):
        k = st.columns(5)
        kpi(k[0], "Cases enrolled", n(len(c2)))
        kpi(k[1], "Female survivors", pct((c2["sex"].fillna("").str.lower() == "female").sum(), c2["sex"].notna().sum()))
        under18 = c2["age_bracket"].fillna("").str.lower().str.startswith("under").sum()
        kpi(k[2], "Children (under 18)", n(under18), pct(under18, len(c2)) + " of cases")
        anyref = c2[[p[1] for p in PATHWAYS]].eq("Yes").any(axis=1).sum()
        kpi(k[3], "Cases with ≥1 referral", pct(anyref, len(c2)))
        kpi(k[4], "Consent not obtained", n((c2["consent"] == "No").sum()), "safety check only")

        a, b = st.columns(2)
        with a:
            t = c2.assign(period=bucket(c2["dt"], gran), v=cat(c2["violence"])).groupby(["period", "v"]).size().reset_index(name="Cases")
            plot(px.bar(t, x="period", y="Cases", color="v", title=f"Cases by violence type per {gran.lower()}", labels={"period": "", "v": ""}))
        with b:
            v = vc(c2["violence"]).sort_values("Count")
            plot(px.bar(v, x="Count", y="label", orientation="h", title="Violence type (broad category)", labels={"label": ""}))

        a, b = st.columns(2)
        with a:
            t = c2.assign(ab=cat(c2["age_bracket"]), sx=cat(c2["sex"])).groupby(["ab", "sx"]).size().reset_index(name="Cases")
            order = sorted(t["ab"].unique(), key=age_sort_key)
            plot(px.bar(t, x="ab", y="Cases", color="sx", category_orders={"ab": order},
                        title="Age bracket and sex", labels={"ab": "", "sx": ""}))
        with b:
            t = c2.assign(vil=cat(c2["village"]), rk=cat(c2["risk"])).groupby(["vil", "rk"]).size().reset_index(name="Cases")
            plot(px.bar(t, x="vil", y="Cases", color="rk", color_discrete_map=RISK_COLORS,
                        title="Risk level by village unit", labels={"vil": "", "rk": ""}))

        st.subheader("Referral pathways")
        rows = []
        for label, rk, dk in PATHWAYS:
            ref = c2[c2[rk] == "Yes"]
            days = (ref[dk] - ref["intake_date"]).dt.days.dropna()
            rows.append({"Pathway": label, "Referred": len(ref), "% of cases": round(100 * len(ref) / len(c2), 1),
                         "Median days intake → referral": (float(days.median()) if len(days) else np.nan),
                         "Missing referral date": int(ref[dk].isna().sum())})
        pw = pd.DataFrame(rows)
        a, b = st.columns([3, 2])
        with a:
            plot(px.bar(pw.sort_values("Referred"), x="Referred", y="Pathway", orientation="h",
                        title="Cases referred by pathway", labels={"Pathway": ""}, color_discrete_sequence=[PALETTE[0]]), h=300)
        with b:
            plot(px.pie(vc(c2["ref_status"]), names="label", values="Count", hole=0.5, title="Referral completion status"), h=300)
        show_df(pw)

        a, b = st.columns(2)
        with a:
            t = c2.assign(off=cat(c2["officer"]), rk=cat(c2["risk"])).groupby(["off", "rk"]).size().reset_index(name="Cases")
            plot(px.bar(t, x="off", y="Cases", color="rk", color_discrete_map=RISK_COLORS,
                        title="Caseload by case officer", labels={"off": "", "rk": ""}))
        with b:
            if c2["violence"].notna().any() and c2["age_bracket"].notna().any():
                hm = pd.crosstab(cat(c2["violence"]), cat(c2["age_bracket"]))
                hm = hm[sorted(hm.columns, key=age_sort_key)]
                fig = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale=["#F4EFFB", "#4B2E83"],
                                title="Violence type × age bracket", labels={"x": "", "y": "", "color": "Cases"})
                plot(fig)

# -----------------------------------------------------------------------------
# FOLLOW-UP & ACCOMPANIMENT
# -----------------------------------------------------------------------------
with tabs[3]:
    k = st.columns(4)
    kpi(k[0], "Accompaniments logged", n(len(c3)))
    kpi(k[1], "Cases accompanied", n(c3["acc_cid"].nunique()))
    enrolled = tracker["cid"].nunique() if len(tracker) else 0
    kpi(k[2], "Enrolled cases with ≥1 follow-up", pct((tracker["fu_status"] != "No follow-up logged").sum() if len(tracker) else 0, enrolled))
    kpi(k[3], "Overdue follow-ups", n((tracker["fu_status"] == "Overdue").sum()) if len(tracker) else "0")

    if len(c3):
        a, b, c = st.columns(3)
        with a:
            plot(px.bar(vc(c3["acc_type"]).sort_values("Count"), x="Count", y="label", orientation="h",
                        title="Accompaniment type", labels={"label": ""}), h=300)
        with b:
            plot(px.bar(vc(c3["acc_by"]).sort_values("Count"), x="Count", y="label", orientation="h",
                        title="Accompanied by", labels={"label": ""}, color_discrete_sequence=[PALETTE[1]]), h=300)
        with c:
            t = c3.assign(period=bucket(c3["dt"], gran)).groupby("period").size().reset_index(name="Accompaniments")
            plot(px.line(t, x="period", y="Accompaniments", markers=True, title="Accompaniments over time", labels={"period": ""}), h=300)
    else:
        st.info("No accompaniment records in the selected period.")

    st.subheader("Follow-up tracker")
    if len(tracker):
        opts = ["Overdue", "No follow-up logged", "No next date set", "Scheduled"]
        sel = st.multiselect("Show status", opts, default=opts[:3])
        tt = tracker[tracker["fu_status"].isin(sel)].sort_values(["days_overdue"], ascending=False)
        show_df(tt.rename(columns={"cid": "Case ID", "village": "Village", "risk": "Risk", "officer": "Case officer",
                                   "last_acc": "Last accompaniment", "acc_type": "Last type", "next_fu": "Next follow-up",
                                   "fu_status": "Status", "days_overdue": "Days overdue"})
                [["Case ID", "Village", "Risk", "Case officer", "Last accompaniment", "Last type", "Next follow-up", "Status", "Days overdue"]])
    else:
        st.info("No survivor case records to track yet.")

# -----------------------------------------------------------------------------
# COORDINATION & SYSTEMS
# -----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Multi-agency case conferences")
    if not empty(c4, "No multi-agency meetings in the selected period."):
        k = st.columns(4)
        kpi(k[0], "Meetings held", n(len(c4)))
        kpi(k[1], "Cases reviewed", n(c4["mtg_cases"].sum()))
        nxt = c4["mtg_next"].dropna()
        up_next = nxt[nxt >= today].min() if (nxt >= today).any() else None
        kpi(k[2], "Next meeting", f"{up_next:%d %b %Y}" if up_next is not None else "—")
        kpi(k[3], "Latest meeting", f"{c4['dt'].max():%d %b %Y}")
        t = c4.sort_values("dt")
        plot(px.bar(t, x="dt", y="mtg_cases", title="Cases reviewed per meeting", labels={"dt": "", "mtg_cases": "Cases"},
                    color_discrete_sequence=[PALETTE[2]]), h=280)
        with st.expander("Meeting register (decisions & action tracker)"):
            show_df(t.sort_values("dt", ascending=False)[["mtg_no", "dt", "mtg_agencies", "mtg_cases", "mtg_decisions", "mtg_actions", "mtg_next"]]
                    .rename(columns={"mtg_no": "#", "dt": "Date", "mtg_agencies": "Agencies", "mtg_cases": "Cases reviewed",
                                     "mtg_decisions": "Key decisions", "mtg_actions": "Action tracker", "mtg_next": "Next meeting"}))

    st.subheader("Police stations / Child Protection Units")
    if not empty(c5, "No police / CPU activities in the selected period."):
        k = st.columns(3)
        kpi(k[0], "Officers trained", n(c5["pol_trained"].sum()))
        kpi(k[1], "Stations / CPUs supported", n(c5["pol_station"].nunique()))
        kpi(k[2], "Activities", n(len(c5)))
        a, b = st.columns(2)
        with a:
            t = c5.assign(st_=cat(c5["pol_station"])).groupby("st_")["pol_trained"].sum().sort_values().reset_index()
            plot(px.bar(t, x="pol_trained", y="st_", orientation="h", title="Officers trained by station",
                        labels={"pol_trained": "Officers", "st_": ""}), h=300)
        with b:
            fs = c5.sort_values("dt").groupby("pol_station").tail(1).copy()
            fs["rb"], fs["rc"] = fs["pol_base"].map(func_rank), fs["pol_curr"].map(func_rank)
            if fs[["rb", "rc"]].notna().all(axis=1).any():
                imp = (fs["rc"] > fs["rb"]).sum()
                kpi(st.container(), "Stations with improved functionality", n(imp), f"of {len(fs)} stations")
        show_df(c5.sort_values("dt", ascending=False)[["dt", "pol_station", "pol_type", "pol_trained", "pol_equipment", "pol_base", "pol_curr"]]
                .rename(columns={"dt": "Date", "pol_station": "Station / CPU", "pol_type": "Activity", "pol_trained": "Officers trained",
                                 "pol_equipment": "Support provided", "pol_base": "Baseline functionality", "pol_curr": "Current functionality"}))

    st.subheader("GBV help-desk")
    if not empty(c6, "No help-desk records yet."):
        show_df(c6[["desk_loc", "desk_officer", "desk_hours"]].drop_duplicates()
                .rename(columns={"desk_loc": "Desk location", "desk_officer": "Case officer on duty", "desk_hours": "Hours open"}))

# -----------------------------------------------------------------------------
# DATA QUALITY
# -----------------------------------------------------------------------------
with tabs[5]:
    st.caption("Checks run on ALL records (ignores the period / case filters).")
    issues = []

    def add(sec, rec, issue, sev):
        issues.append({"Section": sec, "Record": rec, "Issue": issue, "Severity": sev})

    rec_id = lambda r: r["cid"] if isinstance(r.get("cid"), str) else f"Kobo _id {r['_id']}"
    for _, r in S["blank"].iterrows():
        add("All", f"Kobo _id {r['_id']}", "Empty submission (no section data)", "Low")
    for _, r in S["c1"].iterrows():
        rid = f"Kobo _id {r['_id']}"
        if pd.isna(r["c1_date"]):
            add("C1 Outreach", rid, "Activity date missing (submission date used)", "Medium")
        if pd.isna(r["c1_location"]):
            add("C1 Outreach", rid, "Location missing", "Low")
        if pd.notna(r["c1_srh_counseled"]) and pd.notna(r["c1_seen"]) and r["c1_srh_counseled"] > r["c1_seen"]:
            add("C1 Outreach", rid, "Counselled on SRH exceeds individuals seen", "Medium")
        if pd.notna(r["c1_seen"]) and (r["c1_new"] or 0) + (r["c1_followup"] or 0) > r["c1_seen"]:
            add("C1 Outreach", rid, "New + follow-up GBV cases exceed individuals seen", "Medium")
    seen_ids = S["c2"]["cid"].dropna()
    dup = set(seen_ids[seen_ids.duplicated(keep=False)])
    for _, r in S["c2"].iterrows():
        rid = rec_id(r)
        if pd.isna(r["cid"]):
            add("C2 Cases", rid, "Case ID missing", "High")
        elif not re.match(CASE_ID_PATTERN, r["cid"]):
            add("C2 Cases", rid, "Case ID not in coded format NYA-[Village initial]-[###]", "High")
        if r["cid"] in dup:
            add("C2 Cases", rid, "Duplicate Case ID", "High")
        if pd.isna(r["consent"]):
            add("C2 Cases", rid, "Informed consent not recorded", "High")
        refs = [r[p[1]] for p in PATHWAYS]
        if r["consent"] == "No" and any(x == "Yes" for x in refs):
            add("C2 Cases", rid, "Referrals recorded although consent = No", "High")
        if pd.notna(r["age"]) and isinstance(r["age_bracket"], str):
            if (r["age"] < 18) != r["age_bracket"].lower().startswith("under"):
                add("C2 Cases", rid, "Age and age bracket do not agree", "Medium")
        if pd.notna(r["intake_date"]) and r["intake_date"] > today:
            add("C2 Cases", rid, "Intake date is in the future", "Medium")
        if pd.isna(r["ref_status"]):
            add("C2 Cases", rid, "Referral completion status missing", "Low")
        for label, rk, dk in PATHWAYS:
            if r[rk] == "Yes" and pd.isna(r[dk]):
                add("C2 Cases", rid, f"{label}: referred but no referral date", "Medium")
            if r[rk] != "Yes" and pd.notna(r[dk]):
                add("C2 Cases", rid, f"{label}: referral date given but not marked as referred", "Medium")
            if pd.notna(r[dk]) and pd.notna(r["intake_date"]) and r[dk] < r["intake_date"]:
                add("C2 Cases", rid, f"{label}: referral date is before intake date", "Medium")
    known = set(seen_ids)
    for _, r in S["c3"].iterrows():
        rid = r["acc_cid"] if isinstance(r["acc_cid"], str) else f"Kobo _id {r['_id']}"
        if pd.isna(r["acc_cid"]):
            add("C3 Follow-up", rid, "Case ID missing", "High")
        elif r["acc_cid"] not in known:
            add("C3 Follow-up", rid, "Case ID not found in survivor records (C2)", "Medium")
        if pd.notna(r["next_fu"]) and pd.notna(r["acc_date"]) and r["next_fu"] < r["acc_date"]:
            add("C3 Follow-up", rid, "Next follow-up date is before the accompaniment date", "Medium")
    iss = pd.DataFrame(issues, columns=["Section", "Record", "Issue", "Severity"])
    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    if len(iss):
        iss = iss.sort_values("Severity", key=lambda s: s.map(sev_rank))
    k = st.columns(4)
    kpi(k[0], "Issues found", n(len(iss)))
    kpi(k[1], "High severity", n((iss["Severity"] == "High").sum()) if len(iss) else "0")
    kpi(k[2], "Medium", n((iss["Severity"] == "Medium").sum()) if len(iss) else "0")
    kpi(k[3], "Records checked", n(len(df)))
    if len(iss):
        sevs = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        view = iss[iss["Severity"].isin(sevs)]
        show_df(view)
        st.download_button("⬇️ Download issues (CSV)", view.to_csv(index=False).encode(), "data_quality_issues.csv", "text/csv")
    else:
        st.success("No data-quality issues detected.")

    st.subheader("Reconciliation: GBV cases reported at outreach (C1) vs case records (C2)")
    a_ = S["c1"].assign(m=S["c1"]["dt"].dt.to_period("M").dt.to_timestamp()).groupby("m")["c1_new"].sum()
    b_ = S["c2"].assign(m=S["c2"]["dt"].dt.to_period("M").dt.to_timestamp()).groupby("m").size()
    rec = pd.concat([a_.rename("Reported in C1 (new GBV cases)"), b_.rename("Enrolled in C2 (case records)")], axis=1, sort=True).fillna(0).astype(int)
    if len(rec):
        rec["Gap (C1 − C2)"] = rec.iloc[:, 0] - rec.iloc[:, 1]
        rec.index = rec.index.strftime("%b %Y")
        show_df(rec.reset_index().rename(columns={"index": "Month", "m": "Month"}))
        st.caption("A persistent positive gap means cases were counted at outreach but not yet opened as coded case records.")

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
with tabs[6]:
    st.caption("Survivor case records (coded, filtered). Narrative notes are never exported from this view.")
    cols = ["cid", "intake_date", "village", "age", "age_bracket", "sex", "violence", "risk", "consent", "ref_health", "ref_police",
            "ref_legal", "ref_psy", "ref_cps", "ref_status", "officer"]
    if len(c2):
        out = c2[cols].rename(columns={"cid": "Case ID"})
        show_df(out)
        st.download_button("⬇️ Download case table (CSV)", out.to_csv(index=False).encode(), "gbv_cases_coded.csv", "text/csv")
    else:
        st.info("No case records for the selected filters.")

st.markdown("---")
st.caption("🔒 CONFIDENTIAL – coded, de-identified data only. Never enter names, phone numbers or exact addresses in Kobo. "
           "Keep Kobo project encryption on and restrict access to named case officers and their supervisor.")
