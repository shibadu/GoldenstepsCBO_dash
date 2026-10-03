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

# =============================================================================
# CONFIG  –  EDIT HERE (or set the same keys in .streamlit/secrets.toml)
# =============================================================================


def _secret(key, default):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


KOBO_SERVER = _secret("KOBO_SERVER", "https://kf.kobotoolbox.org")  # <-- your Kobo server
KOBO_API_TOKEN = _secret("KOBO_API_TOKEN", "")  # API key: put in .streamlit/secrets.toml, NOT in this file
KOBO_ASSET_UID = _secret("KOBO_ASSET_UID", "aJ5SsJRgzQw6UtpNtH63V2")  # form ID (not secret)
KOBO_FALLBACK_SERVERS = ["https://kf.kobotoolbox.org", "https://eu.kobotoolbox.org",
                         "https://kobo.humanitarianresponse.info"]  # tried automatically if the first fails
APP_PASSWORD = _secret("APP_PASSWORD", "")  # optional access code (recommended)

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

PALETTE = ["#4B2E83", "#C99A2E", "#1F8A8A", "#D1495B", "#7A9E3B", "#8E7DBE", "#E08E45", "#5B7DB1"]
RISK_COLORS = {"High": "#D1495B", "Medium": "#C99A2E", "Low": "#1F8A8A", "Not recorded": "#9AA0A6"}
px.defaults.color_discrete_sequence = PALETTE

# =============================================================================
# PAGE + STYLE
# =============================================================================
st.set_page_config(page_title="Golden Steps CBO · GBV Case Management", page_icon="👣", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"], .stDeployButton, [class*="viewerBadge"], [class*="_profileContainer"],
a[href*="github.com"], a[href*="streamlit.io"] {display: none !important; visibility: hidden !important;}
.kpi {border-left: 5px solid #C99A2E; background: rgba(127,127,127,0.09);
      border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; min-height: 96px;}
.kpi-l {font-size: 0.78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .75;}
.kpi-v {font-size: 1.9rem; font-weight: 700; line-height: 1.25; color: #4B2E83;}
.kpi-s {font-size: 0.78rem; opacity: .7;}
@media (prefers-color-scheme: dark) {.kpi-v {color: #CDB8FF;}}
.small-note {font-size: 0.82rem; opacity: .75;}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# ACCESS CONTROL
# =============================================================================
def gate():
    if not APP_PASSWORD or st.session_state.get("auth"):
        return
    st.title("👣 Golden Steps CBO")
    st.caption("Restricted – authorised case officers and supervisors only.")
    pw = st.text_input("Access code", type="password")
    if pw:
        if hmac.compare_digest(str(pw), str(APP_PASSWORD)):
            st.session_state["auth"] = True
            st.rerun()
        else:
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
    df["risk"] = df["risk"].map(lambda v: v.title() if isinstance(v, str) else v)
    df["cid"] = df["case_id"].map(lambda v: v.upper() if isinstance(v, str) else v)
    df["acc_cid"] = df["acc_case_id"].map(lambda v: v.upper() if isinstance(v, str) else v)
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
    return s.fillna("").str.lower().str.startswith("complete")


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
st.sidebar.title("👣 Golden Steps CBO")
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

if not KOBO_API_TOKEN or not KOBO_ASSET_UID or "YOUR_" in str(KOBO_ASSET_UID):
    st.error("**KoboToolbox is not configured.** Add `KOBO_API_TOKEN` (and optionally `KOBO_ASSET_UID`, "
             "`KOBO_SERVER`) to `.streamlit/secrets.toml` locally, or to *Settings → Secrets* on Streamlit Cloud.")
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
st.title("GBV Case Management Dashboard")
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
