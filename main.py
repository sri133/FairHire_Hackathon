"""FairForge: fair, explained resume ranking.   Run with:  streamlit run main.py

THIS FILE IS THE USER INTERFACE ONLY. All scoring, hiding of identity, SHAP and Twin Test logic lives in core.py
and is called exactly as before (engine.analyze). Nothing in the maths was changed here.

Design idea ("the Forge")
  * Dark, high-contrast theme: every text colour is forced, so text is readable in Streamlit's light OR dark mode.
  * No sidebar. The score recipe lives in a "Forge console" at the top of the page (four colour-coded faders).
  * Navigation is a floating glass DOCK at the bottom of the screen (the Streamlit tab bar, restyled).
  * The hero is a Three.js particle sculpture that morphs through the FairForge story:
        two twin candidates -> identity dissolves -> only skills remain -> balanced scales of justice.
  * Results are a podium, a live leaderboard, a Twin-Test balance scale (tilts with the score gap),
    a redaction view of the resume, and dark, glowing Plotly charts.

Libraries used for the look: Google Fonts (Bricolage Grotesque, Instrument Serif, JetBrains Mono) and Three.js from a CDN.
Without internet the app falls back to normal fonts and a plain gradient banner, and everything stays readable.

Sections of this file
  1. page setup and colour palette
  2. demo data and static text
  3. CSS (the whole look and feel)
  4. hero banner (Three.js)
  5. small HTML builders (cards, chips, tables, scales ...)
  6. Plotly chart builders
  7. app start, forge console, ranking run
  8. result tabs
"""
# Imports: standard library first, then third-party libraries, then our own scoring module.
import html
import io
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA

# core.py holds all the scoring logic. This file only draws the interface around it.
from core import FairForge, TWIN_A, TWIN_B

# ================================================================================================================
# 1. PAGE SETUP AND COLOUR PALETTE
# ================================================================================================================
# The sidebar is collapsed and also hidden by CSS: we use a console at the top of the page instead.
st.set_page_config(page_title="FairForge | blind, glass-box resume ranking", page_icon="⚖️", layout="wide",
                   initial_sidebar_state="collapsed")

# One palette used by the CSS, the charts and the hero. (If you change a colour, change it in all three places.)
INK_DEEP, INK_BASE, INK_RAISED, INK_LIFT = "#06080F", "#0B1020", "#121A30", "#1B2644"      # backgrounds, darkest -> lightest
TEXT_BRIGHT, TEXT_DIM, TEXT_FAINT = "#F4F1EA", "#B9C1D4", "#8F99B0"                         # text, all readable on the backgrounds
EMBER, TEAL, GOLD, VIOLET, SKY = "#FF6A3D", "#2EE6C5", "#FFC24B", "#9B7BFF", "#4DA3FF"    # accent colours
GRID_LINE = "rgba(255,255,255,0.10)"                                                       # faint lines inside charts
CANDIDATE_COLORS = [EMBER, TEAL, GOLD, VIOLET, SKY]                                        # one colour per candidate in charts

# ================================================================================================================
# 2. DEMO DATA AND STATIC TEXT (unchanged from the original app)
# ================================================================================================================
# Notice shown at the top of the page and in the About tab.
DISCLAIMER = ("FairForge is decision support, not a decision maker. It hides identity details and explains its scores, "
              "but it cannot promise zero bias. A person must make the final hiring decision. All identities used in the "
              "fairness tests are synthetic (made up).")

# Sample job description pre-filled in the text box so the app can be tried without typing anything.
DEMO_JD = ("Data Analyst. We need an analyst with strong SQL, Python and Excel skills to build dashboards and reports. "
           "You will clean data, run statistical analysis, share insights with managers and support budget planning. "
           "Experience with Tableau or Power BI is a plus.")

# Three sample resumes, used when no files are uploaded or the demo checkbox is ticked.
DEMO_RESUMES = {
    "demo_priya.txt": """Priya Nair
priya.nair@example.com | +91 98765 43210 | Chennai
Summary
Data analyst with 4 years of experience. She builds dashboards and reports for business teams.
Skills
SQL, Python, pandas, Excel, Tableau, statistics, data cleaning
Experience
Data Analyst, ABC Retail 2019 to 2023
Built Tableau dashboards used by 40 managers and cut weekly report time by 30%.
Cleaned and analysed sales data in SQL and Python to support budget planning.
Education
B.Sc Statistics, Anna University, 2018""",
    "demo_rahul.txt": """Rahul Sharma
rahul.sharma@example.com | Mumbai
Summary
HR manager focused on hiring and employee relations. He leads a team of 6 recruiters.
Skills
Recruitment, onboarding, payroll, employee engagement, MS Excel
Experience
HR Manager, XYZ Services 2015 to 2023
Managed hiring for 200 roles per year and reduced time to hire by 20%.
Responsible for payroll reports in Excel and monthly headcount analysis.
Education
MBA Human Resources, Delhi University, 2014""",
    "demo_emily.txt": """Emily Carter
emily.carter@example.com | Austin, TX
Summary
Business analyst who turns data into clear recommendations. She enjoys SQL and storytelling with data.
Skills
SQL, Power BI, Excel, Python, data analysis, reporting, budgeting
Experience
Business Analyst, Northwind Co 2017 to 2024
Created Power BI reports for finance leaders and improved forecast accuracy by 15%.
Wrote SQL queries to clean data and present insights to managers.
Education
B.A. Economics, Texas State University, 2016""",
}

# The four signals that make up the score (names must match the keys that core.py returns).
PART_NAMES = ["Meaning match", "Job fit (DNN)", "Career path (BiLSTM)", "Skill keywords"]

# How each signal looks in the console and in the charts: colour, the CSS class that tints its panel, the slider
# label (same wording as before), the default weight (same as before) and a one-line plain-English explanation.
SIGNAL_STYLES = {
    "Meaning match": {"color": TEAL, "accent_class": "ff-accent-teal", "slider_label": "Meaning match (MiniLM)", "default_weight": 30,
                      "tagline": "Does the resume mean the same thing as the job?"},
    "Job fit (DNN)": {"color": EMBER, "accent_class": "ff-accent-ember", "slider_label": "Job fit (DNN)", "default_weight": 30,
                      "tagline": "How likely is this resume to belong to the job family?"},
    "Career path (BiLSTM)": {"color": GOLD, "accent_class": "ff-accent-gold", "slider_label": "Career path (BiLSTM)", "default_weight": 15,
                             "tagline": "Does the order of past jobs fit the role?"},
    "Skill keywords": {"color": VIOLET, "accent_class": "ff-accent-violet", "slider_label": "Skill keywords", "default_weight": 25,
                       "tagline": "How many of the job's key skills appear?"},
}

# The five steps of the pipeline, shown on the start page and in the About tab.
PIPELINE_STEPS = [
    ("🫥", "Hide identity", "spaCy + rules remove names, places, colleges, pronouns, contacts and graduation years.", EMBER),
    ("🧩", "Sort sections", "A CNN labels every line: Summary, Skills, Experience, Education or Other.", GOLD),
    ("🧠", "Read the career", "A BiLSTM reads the job history in order, forwards and backwards.", VIOLET),
    ("🎯", "Judge the fit", "MiniLM turns text into meaning; a 3-layer DNN gives a job-fit probability.", TEAL),
    ("🔍", "Explain and test", "SHAP shows the words that mattered. The Twin Test proves identity did not.", SKY),
]

# ================================================================================================================
# 3. CSS: THE WHOLE LOOK AND FEEL
# ================================================================================================================
# @import must be the very first rule of a stylesheet, so the font import is kept apart from the rest.
FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800"
               "&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;600&display=swap');")

# Palette as CSS variables, built from the Python constants above so both always agree.
CSS_VARIABLES = (f":root{{--ink-deep:{INK_DEEP};--ink-base:{INK_BASE};--ink-raised:{INK_RAISED};--ink-lift:{INK_LIFT};"
                 f"--text:{TEXT_BRIGHT};--text-dim:{TEXT_DIM};--text-faint:{TEXT_FAINT};"
                 f"--ember:{EMBER};--teal:{TEAL};--gold:{GOLD};--violet:{VIOLET};--sky:{SKY};}}")

# All styling of the page, written as plain CSS inside a Python string. The comments inside mark each area (background,
# inputs, buttons, dock, podium, tables, animations).
CSS_RULES = """
/* ---------- animated numbers: these registered properties let CSS count up and fill rings ---------- */
@property --n { syntax: '<integer>'; initial-value: 0; inherits: true; }
@property --score { syntax: '<number>'; initial-value: 0; inherits: true; }

/* ---------- page background: deep ink + two coloured glows + a drifting aurora + a faint grid ---------- */
html, body, .stApp { background: var(--ink-base); color: var(--text); font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; }
.stApp { background:
    radial-gradient(1100px 600px at 10% -8%, rgba(255,106,61,.17), transparent 60%),
    radial-gradient(900px 600px at 96% 8%, rgba(46,230,197,.14), transparent 60%),
    linear-gradient(180deg, var(--ink-deep), var(--ink-base) 45%, #0A0F1E); background-attachment: fixed; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, .main { background: transparent !important; }
.stApp::before { content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
    background-size: 58px 58px; -webkit-mask-image: radial-gradient(ellipse at 50% 20%, #000 10%, transparent 75%); mask-image: radial-gradient(ellipse at 50% 20%, #000 10%, transparent 75%); }
.stApp::after { content: ""; position: fixed; width: 60vmax; height: 60vmax; left: -15vmax; bottom: -30vmax; z-index: 0; pointer-events: none; border-radius: 50%;
    background: radial-gradient(circle, rgba(155,123,255,.22), transparent 62%); filter: blur(40px); animation: aurora 22s ease-in-out infinite alternate; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"], .block-container { position: relative; z-index: 1; }
.block-container { max-width: 1280px; padding: 1rem 2rem 170px 2rem; }   /* extra bottom room so the dock never covers content */
::selection { background: var(--ember); color: #14100c; }

/* ---------- things we hide: no sidebar, no menu, no footer ---------- */
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* ---------- text: every native Streamlit text element gets an explicit readable colour ---------- */
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: 'Bricolage Grotesque', sans-serif !important; color: var(--text) !important; letter-spacing: -.015em; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: var(--text-dim) !important; font-size: .86rem; }
.stApp [data-testid="stWidgetLabel"], .stApp [data-testid="stWidgetLabel"] *, .stApp label p { color: var(--text-dim) !important; font-weight: 600; letter-spacing: .01em; }
.stApp [data-testid="stCheckbox"] *, .stApp [data-testid="stToggle"] * { color: var(--text) !important; }
.stApp a { color: var(--teal) !important; }
.stApp code { background: rgba(255,255,255,.08); color: var(--gold) !important; border-radius: 6px; }

/* ---------- inputs: dark glass boxes with light text ---------- */
.stApp textarea, .stApp input[type="text"], .stApp input[type="number"] {
    background: rgba(255,255,255,.045) !important; color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
    border-radius: 16px !important; font-family: 'Bricolage Grotesque', sans-serif !important; caret-color: var(--teal); }
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="base-input"], .stApp [data-baseweb="input"] {
    background: rgba(255,255,255,.045) !important; border: 1px solid rgba(255,255,255,.14) !important; border-radius: 16px !important; }
.stApp [data-baseweb="textarea"]:focus-within, .stApp [data-baseweb="base-input"]:focus-within { border-color: var(--teal) !important; box-shadow: 0 0 0 3px rgba(46,230,197,.2); }
.stApp [data-baseweb="select"] > div { background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.16) !important; border-radius: 14px !important; color: var(--text) !important; }
.stApp [data-baseweb="select"] * { color: var(--text) !important; }
.stApp [data-baseweb="select"] svg { fill: var(--text) !important; }
div[data-baseweb="popover"], div[data-baseweb="popover"] ul, div[data-baseweb="menu"] { background: var(--ink-raised) !important; color: var(--text) !important; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * { color: var(--text) !important; background: transparent; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background: var(--ink-lift) !important; }
.stApp [data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.04) !important; border: 2px dashed rgba(255,255,255,.28) !important; border-radius: 18px !important; transition: border-color .2s, background .2s; }
.stApp [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--teal) !important; background: rgba(46,230,197,.07) !important; }
.stApp [data-testid="stFileUploaderDropzone"] *, .stApp [data-testid="stFileUploaderFile"] * { color: var(--text) !important; }
.stApp [data-testid="stFileUploaderDropzone"] small { color: var(--text-dim) !important; }
.stApp [data-testid="stFileUploaderDropzone"] button { background: transparent !important; border: 1px solid var(--teal) !important; color: var(--teal) !important; border-radius: 999px !important; }
.stApp [data-testid="stFileUploaderDropzone"] button * { color: var(--teal) !important; }

/* ---------- sliders: thumb, value and track follow the accent colour of their panel ---------- */
.stApp [data-testid="stSlider"] *, .stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stTickBarMin"], .stApp [data-testid="stTickBarMax"] { color: var(--text-dim) !important; }
.stApp [data-testid="stSliderThumbValue"] { color: var(--text) !important; font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.stApp [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; box-shadow: 0 0 0 5px rgba(255,255,255,.08), 0 0 18px var(--accent) !important; border: 2px solid #fff !important; }
.stApp [data-baseweb="slider"] div[style*="linear-gradient"] { filter: hue-rotate(var(--accent-hue)) saturate(1.25); }

/* ---------- checkbox, toggle, progress: replace Streamlit's default red with our accents ---------- */
.stApp [data-testid="stCheckbox"] label:has(input:checked) > span:first-of-type { background: var(--teal) !important; border-color: var(--teal) !important; }
.stApp [data-testid="stCheckbox"] label:has(input:checked) > span:first-of-type svg { color: #0B1020 !important; fill: #0B1020 !important; }
.stApp [data-testid="stToggle"] label:has(input:checked) > div:first-of-type, .stApp [data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type { background: var(--teal) !important; }
.stApp [data-testid="stProgress"] div[role="progressbar"] { background: rgba(255,255,255,.1) !important; border-radius: 99px; }
.stApp [data-testid="stProgress"] div[role="progressbar"] > div > div { background: linear-gradient(90deg, var(--ember), var(--gold), var(--teal)) !important; }
.stApp [data-testid="stProgress"] * { color: var(--text-dim) !important; }

/* ---------- buttons: the primary button is the glowing "Ignite" button ---------- */
.stApp button[data-testid="stBaseButton-primary"], .stApp .stButton > button[kind="primary"] {
    background: linear-gradient(100deg, var(--ember), var(--gold), var(--ember)); background-size: 220% 100%; border: none; border-radius: 999px;
    padding: .8rem 2.3rem; box-shadow: 0 0 0 1px rgba(255,255,255,.25) inset, 0 14px 40px rgba(255,106,61,.45), 0 0 60px rgba(255,194,75,.25);
    animation: shimmer 4s linear infinite, breathe 2.8s ease-in-out infinite; transition: transform .2s ease; width: 100%; }
.stApp button[data-testid="stBaseButton-primary"] *, .stApp .stButton > button[kind="primary"] * { color: #1A0E08 !important; font-weight: 800; font-size: 1.05rem; letter-spacing: .02em; }
.stApp button[data-testid="stBaseButton-primary"]:hover, .stApp .stButton > button[kind="primary"]:hover { transform: translateY(-3px) scale(1.03); }

/* ---------- alerts, expanders, charts ---------- */
.stApp [data-testid="stAlert"] { background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.16); border-radius: 16px; }
.stApp [data-testid="stAlert"] * { color: var(--text) !important; }
.stApp [data-testid="stExpander"] details { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.12) !important; border-radius: 18px; }
.stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * { color: var(--text) !important; font-weight: 600; }
.stApp [data-testid="stPlotlyChart"] { background: linear-gradient(160deg, rgba(255,255,255,.06), rgba(255,255,255,.02)); border: 1px solid rgba(255,255,255,.1);
    border-radius: 24px; padding: 10px; box-shadow: 0 18px 50px rgba(0,0,0,.35); animation: rise .7s ease both; }
.stApp iframe { border-radius: 30px; box-shadow: 0 30px 90px rgba(0,0,0,.55), 0 0 0 1px rgba(255,255,255,.12); }

/* ---------- console panels: a Streamlit column that contains .ff-panel-marker becomes a glass card ---------- */
[data-testid="stHorizontalBlock"]:has(.ff-panel-marker) { align-items: stretch !important; }
[data-testid="stColumn"]:has(.ff-panel-marker), [data-testid="column"]:has(.ff-panel-marker) {
    --accent: var(--ember); --accent-hue: 14deg;
    background: linear-gradient(165deg, rgba(255,255,255,.075), rgba(255,255,255,.025)); border: 1px solid rgba(255,255,255,.12); border-top: 3px solid var(--accent);
    border-radius: 24px; padding: 20px 22px 14px 22px; box-shadow: 0 20px 60px rgba(0,0,0,.4); backdrop-filter: blur(14px); animation: rise .7s ease both; }
[data-testid="stColumn"]:has(.ff-accent-teal), [data-testid="column"]:has(.ff-accent-teal) { --accent: var(--teal); --accent-hue: 170deg; }
[data-testid="stColumn"]:has(.ff-accent-ember), [data-testid="column"]:has(.ff-accent-ember) { --accent: var(--ember); --accent-hue: 14deg; }
[data-testid="stColumn"]:has(.ff-accent-gold), [data-testid="column"]:has(.ff-accent-gold) { --accent: var(--gold); --accent-hue: 41deg; }
[data-testid="stColumn"]:has(.ff-accent-violet), [data-testid="column"]:has(.ff-accent-violet) { --accent: var(--violet); --accent-hue: 255deg; }
.ff-step { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
.ff-step-no { font: 600 13px 'JetBrains Mono', monospace; color: #14100c; background: var(--accent, var(--ember)); padding: 3px 9px; border-radius: 8px; }
.ff-step-title { font: 800 21px 'Bricolage Grotesque', sans-serif; color: var(--text); letter-spacing: -.01em; }
.ff-step-hint { font-size: 14px; color: var(--text-dim); margin: 0 0 8px 0; }

/* ---------- the bottom DOCK: Streamlit's tab bar, floated at the bottom of the screen like a glass app dock ---------- */
.stTabs [data-baseweb="tab-list"] { position: fixed; left: 50%; bottom: 20px; transform: translateX(-50%); z-index: 1000; display: flex; gap: 6px; padding: 8px;
    width: auto; max-width: calc(100vw - 20px); overflow-x: auto; background: rgba(12,17,34,.78); backdrop-filter: blur(20px) saturate(1.5);
    border: 1px solid rgba(255,255,255,.18); border-radius: 28px; box-shadow: 0 24px 70px rgba(0,0,0,.65), inset 0 1px 0 rgba(255,255,255,.14), 0 0 40px rgba(46,230,197,.12); }
.stTabs [data-baseweb="tab"] { height: auto; padding: 10px 18px; border-radius: 20px; background: transparent; white-space: nowrap;
    transition: transform .3s cubic-bezier(.3,1.7,.5,1), background .2s; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] * { color: var(--text-dim) !important; font-weight: 700; font-size: .95rem; }
.stTabs [data-baseweb="tab"]:hover { transform: translateY(-9px) scale(1.14); background: rgba(255,255,255,.1); }
.stTabs [data-baseweb="tab"]:hover * { color: #fff !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, var(--ember), var(--gold)); box-shadow: 0 8px 24px rgba(255,106,61,.45); }
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] * { color: #1A0E08 !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; animation: rise .55s ease both; }

/* ---------- marquee ribbon under the hero ---------- */
.ff-marquee { overflow: hidden; margin: 18px 0 6px; padding: 10px 0; border-top: 1px solid rgba(255,255,255,.12); border-bottom: 1px solid rgba(255,255,255,.12);
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
.ff-marquee-track { display: flex; width: max-content; animation: marquee 32s linear infinite; }
.ff-marquee-track span { font: 600 13px 'JetBrains Mono', monospace; letter-spacing: .22em; color: var(--text-dim); padding: 0 26px; white-space: nowrap; }
.ff-marquee-track span:nth-child(4n+1) { color: var(--gold); } .ff-marquee-track span:nth-child(4n+3) { color: var(--teal); }

/* ---------- notices and section titles ---------- */
.ff-notice { margin: 14px 0 18px; padding: 12px 18px; border-radius: 16px; background: rgba(255,194,75,.09); border: 1px solid rgba(255,194,75,.35);
    color: var(--text); font-size: 14px; line-height: 1.5; }
.ff-notice b { color: var(--gold); }
.ff-section { display: flex; align-items: baseline; gap: 16px; margin: 34px 0 14px; }
.ff-section-no { font: 800 54px/1 'Bricolage Grotesque', sans-serif; color: transparent; -webkit-text-stroke: 1.5px var(--ember); opacity: .9; }
.ff-section-title { font: italic 400 38px/1.05 'Instrument Serif', Georgia, serif; color: var(--text); }
.ff-section-sub { font-size: 14px; color: var(--text-dim); margin-top: 4px; }
.ff-dock-hint { display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; border-radius: 999px; background: rgba(46,230,197,.1); border: 1px solid rgba(46,230,197,.4);
    color: var(--teal); font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .08em; }
.ff-dock-hint i { display: inline-block; animation: bounce 1.2s ease-in-out infinite; font-style: normal; }

/* ---------- recipe bar and status LEDs ---------- */
.ff-recipe { display: flex; height: 18px; border-radius: 99px; overflow: hidden; background: rgba(255,255,255,.08); box-shadow: inset 0 2px 6px rgba(0,0,0,.4); margin: 14px 0 8px; }
.ff-recipe div { height: 100%; transition: width .5s ease; }
.ff-recipe-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font: 600 12px 'JetBrains Mono', monospace; color: var(--text-dim); }
.ff-recipe-legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; background: var(--dot); box-shadow: 0 0 8px var(--dot); }
.ff-leds { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.ff-led { display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border-radius: 999px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.12);
    font: 600 12px 'JetBrains Mono', monospace; color: var(--text); }
.ff-led::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--led); box-shadow: 0 0 10px var(--led); animation: pulse 2s ease-in-out infinite; }

/* ---------- podium ---------- */
.ff-podium { display: flex; align-items: flex-end; justify-content: center; gap: 22px; margin: 10px 0 8px; perspective: 1100px; }
.ff-podium-slot { flex: 1 1 0; max-width: 300px; display: flex; flex-direction: column; align-items: stretch; }
.ff-podium-card { position: relative; text-align: center; padding: 22px 16px 18px; border-radius: 26px; color: var(--text);
    background: linear-gradient(165deg, rgba(255,255,255,.1), rgba(255,255,255,.03)); border: 1px solid rgba(255,255,255,.16); box-shadow: 0 22px 60px rgba(0,0,0,.45), 0 0 50px color-mix(in srgb, var(--ring-color) 22%, transparent);
    animation: rise .8s cubic-bezier(.2,.9,.3,1.2) both, grow-score 1.8s ease-out both; transition: transform .35s ease; transform-style: preserve-3d; }
.ff-podium-card:hover { transform: translateY(-10px) rotateX(6deg) rotateY(-4deg); }
.ff-crown { position: absolute; top: -26px; left: 50%; transform: translateX(-50%); font-size: 30px; filter: drop-shadow(0 0 12px var(--gold)); animation: bounce 2.4s ease-in-out infinite; }
.ff-rank-tag { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .2em; color: var(--ring-color); }
.ff-ring { width: 132px; height: 132px; margin: 12px auto 12px; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--ring-color) calc(var(--score) * 1%), rgba(255,255,255,.1) 0); box-shadow: 0 0 30px color-mix(in srgb, var(--ring-color) 40%, transparent); }
.ff-ring-core { width: 100px; height: 100px; border-radius: 50%; background: var(--ink-raised); display: grid; place-items: center; align-content: center; }
.ff-count { counter-reset: n var(--n); font: 800 38px/1 'Bricolage Grotesque', sans-serif; color: var(--text); }
.ff-count::after { content: counter(n); }
.ff-ring-core small { font: 600 11px 'JetBrains Mono', monospace; color: var(--text-dim); margin-top: -2px; }
.ff-card-name { font: 800 22px 'Bricolage Grotesque', sans-serif; color: var(--text); margin-bottom: 8px; }
.ff-move { display: inline-block; padding: 4px 12px; border-radius: 999px; font: 600 12px 'JetBrains Mono', monospace; }
.ff-move-up { background: rgba(46,230,197,.16); color: var(--teal); border: 1px solid rgba(46,230,197,.5); }
.ff-move-down { background: rgba(255,106,61,.16); color: #FF9B7A; border: 1px solid rgba(255,106,61,.55); }
.ff-move-same { background: rgba(255,255,255,.08); color: var(--text-dim); border: 1px solid rgba(255,255,255,.2); }
.ff-card-sub { font: 500 12px 'JetBrains Mono', monospace; color: var(--text-dim); margin-top: 8px; }
.ff-pedestal { margin-top: 10px; border-radius: 16px 16px 6px 6px; display: grid; place-items: center; font: 800 54px 'Bricolage Grotesque', sans-serif; color: rgba(255,255,255,.88);
    background: linear-gradient(180deg, color-mix(in srgb, var(--ring-color) 55%, #0B1020), rgba(255,255,255,.03)); border: 1px solid rgba(255,255,255,.14); border-bottom: none; animation: rise .9s ease both; }

/* ---------- leaderboard rows ---------- */
.ff-board { display: grid; gap: 10px; margin: 8px 0 4px; }
.ff-board-row { display: grid; grid-template-columns: 56px 150px 1fr 74px 120px; align-items: center; gap: 14px; padding: 12px 18px; border-radius: 18px;
    background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); animation: rise .6s ease both; transition: transform .2s, background .2s; }
.ff-board-row:hover { transform: translateX(6px); background: rgba(255,255,255,.09); }
.ff-board-rank { font: 800 26px 'Bricolage Grotesque', sans-serif; color: var(--gold); }
.ff-board-name { font: 700 17px 'Bricolage Grotesque', sans-serif; color: var(--text); }
.ff-board-bar { height: 12px; border-radius: 99px; background: rgba(255,255,255,.1); overflow: hidden; }
.ff-board-bar div { height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg, var(--ember), var(--gold), var(--teal)); animation: grow-bar 1.2s ease-out both; }
.ff-board-score { font: 600 17px 'JetBrains Mono', monospace; color: var(--text); text-align: right; }

/* ---------- stat tiles, chips, tips, bars ---------- */
.ff-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 6px 0 16px; }
.ff-tile { padding: 18px 18px 16px; border-radius: 22px; background: linear-gradient(165deg, rgba(255,255,255,.08), rgba(255,255,255,.025)); border: 1px solid rgba(255,255,255,.12);
    border-top: 3px solid var(--tile-color); box-shadow: 0 16px 44px rgba(0,0,0,.35); animation: rise .6s ease both; transition: transform .25s; }
.ff-tile:hover { transform: translateY(-5px) rotate(-.6deg); }
.ff-tile-label { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .08em; color: var(--text-dim); text-transform: uppercase; }
.ff-tile-value { font: 800 40px/1.1 'Bricolage Grotesque', sans-serif; color: var(--text); margin-top: 6px; }
.ff-tile-value small { font-size: 17px; color: var(--text-dim); font-weight: 600; }
.ff-tile-note { font-size: 13px; color: var(--text-dim); margin-top: 4px; }
.ff-chip { display: inline-block; margin: 4px 6px 4px 0; padding: 5px 14px; border-radius: 999px; font: 600 13px 'JetBrains Mono', monospace; }
.ff-yes { background: rgba(46,230,197,.15); color: #6FF2DA; border: 1px solid rgba(46,230,197,.5); }
.ff-no { background: rgba(255,106,61,.15); color: #FF9B7A; border: 1px solid rgba(255,106,61,.55); }
.ff-note { font: italic 400 19px 'Instrument Serif', Georgia, serif; color: var(--text-dim); }
.ff-group-title { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .14em; color: var(--text-dim); text-transform: uppercase; margin: 12px 0 4px; }
.ff-tip { display: flex; gap: 14px; align-items: flex-start; padding: 14px 18px; margin: 8px 0; border-radius: 18px; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); color: var(--text); line-height: 1.5; }
.ff-tip-no { flex: none; width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center; font: 800 14px 'JetBrains Mono', monospace; background: var(--gold); color: #14100c; }
.ff-bar-row { display: grid; grid-template-columns: 210px 1fr 64px; gap: 14px; align-items: center; margin: 9px 0; color: var(--text); font-size: 14px; }
.ff-bar-track { position: relative; height: 14px; border-radius: 99px; background: rgba(255,255,255,.1); }
.ff-bar-track div.ff-bar-fill { height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg, var(--ember), var(--gold)); animation: grow-bar 1.2s ease-out both; }
.ff-bar-track div.ff-bar-chance { position: absolute; top: -4px; bottom: -4px; width: 2px; left: var(--chance); background: var(--teal); box-shadow: 0 0 8px var(--teal); }
.ff-bar-value { font: 600 14px 'JetBrains Mono', monospace; text-align: right; color: var(--text); }

/* ---------- tables (replace st.dataframe so colours never depend on Streamlit's theme) ---------- */
.ff-table-wrap { overflow-x: auto; border-radius: 20px; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); margin: 8px 0; }
.ff-table { width: 100%; border-collapse: collapse; font-size: 14px; color: var(--text); }
.ff-table th { text-align: left; padding: 12px 16px; font: 600 11px 'JetBrains Mono', monospace; letter-spacing: .1em; text-transform: uppercase; color: var(--gold); background: rgba(255,255,255,.06); white-space: nowrap; }
.ff-table td { padding: 11px 16px; border-top: 1px solid rgba(255,255,255,.08); color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.ff-table tr:hover td { background: rgba(255,255,255,.06); }
.ff-up { color: var(--teal) !important; font-weight: 700; } .ff-down { color: #FF9B7A !important; font-weight: 700; }

/* ---------- hidden-identity view: two paper sheets, redaction bars and section tags ---------- */
.ff-paper { padding: 20px 22px; border-radius: 20px; max-height: 460px; overflow-y: auto; line-height: 1.55; font: 400 14px/1.6 'JetBrains Mono', monospace; color: var(--text); }
.ff-paper-raw { background: #F4F1EA; border: 1px solid #fff; box-shadow: 0 18px 50px rgba(0,0,0,.45); }
.ff-paper-raw, .ff-paper-raw * { color: #1B2644 !important; }
.ff-paper-clean { background: var(--ink-raised); border: 1px solid rgba(46,230,197,.4); box-shadow: 0 0 40px rgba(46,230,197,.15); }
.ff-paper-title { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 10px; color: var(--text-dim); }
.ff-redact { display: inline-block; padding: 0 8px; margin: 0 2px; border-radius: 5px; background: #000; color: var(--gold) !important; font-weight: 700; font-size: 11px; letter-spacing: .08em; box-shadow: 0 0 0 1px var(--gold); }
.ff-line { display: flex; gap: 10px; align-items: baseline; }
.ff-sec { flex: none; width: 82px; font: 600 10px 'JetBrains Mono', monospace; letter-spacing: .08em; text-align: center; padding: 1px 0; border-radius: 6px; color: #14100c; background: var(--sec-color); }

/* ---------- Twin Test balance scales ---------- */
.ff-twin-legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 6px 0 14px; }
.ff-twin-id { flex: 1 1 260px; padding: 14px 18px; border-radius: 18px; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); color: var(--text); }
.ff-twin-id b { font: 800 18px 'Bricolage Grotesque', sans-serif; } .ff-twin-id span { display: block; color: var(--text-dim); font: 500 13px 'JetBrains Mono', monospace; margin-top: 4px; }
.ff-twin-card { padding: 18px 20px; margin: 12px 0; border-radius: 24px; background: linear-gradient(165deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border: 1px solid rgba(255,255,255,.12); animation: rise .6s ease both; }
.ff-twin-head { display: flex; justify-content: space-between; align-items: center; font: 800 20px 'Bricolage Grotesque', sans-serif; color: var(--text); }
.ff-verdict { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .12em; padding: 4px 14px; border-radius: 999px; }
.ff-pass { background: rgba(46,230,197,.16); color: var(--teal); border: 1px solid var(--teal); } .ff-check { background: rgba(255,194,75,.16); color: var(--gold); border: 1px solid var(--gold); }
.ff-twin-scales { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 8px; }
.ff-twin-scale { text-align: center; color: var(--text); }
.ff-twin-scale svg { width: 100%; max-width: 270px; height: auto; overflow: visible; }
.ff-beam { transform-box: view-box; transform-origin: 110px 40px; animation: sway 2.4s cubic-bezier(.3,1.4,.5,1) both; }
.ff-pan { transform-box: view-box; animation: sway-back 2.4s cubic-bezier(.3,1.4,.5,1) both; }
.ff-scale-cap { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--text-dim); }
.ff-scale-gap { font: 800 24px 'Bricolage Grotesque', sans-serif; margin-top: 2px; }
.ff-scale-gap small { font: 500 12px 'JetBrains Mono', monospace; color: var(--text-dim); }

/* ---------- before -> after proof tiles ---------- */
.ff-proof { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.ff-proof-before { color: #FF9B7A; font: 800 34px 'Bricolage Grotesque', sans-serif; text-decoration: line-through; text-decoration-thickness: 2px; opacity: .9; }
.ff-proof-arrow { color: var(--text-dim); font-size: 24px; }
.ff-proof-after { color: var(--teal); font: 800 44px 'Bricolage Grotesque', sans-serif; text-shadow: 0 0 24px rgba(46,230,197,.5); }

/* ---------- pipeline: five glowing stations joined by a moving dashed wire ---------- */
.ff-pipeline { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; position: relative; margin: 10px 0 8px; }
.ff-pipeline::before { content: ""; position: absolute; left: 6%; right: 6%; top: 48px; height: 2px; z-index: 0;
    background: repeating-linear-gradient(90deg, var(--teal) 0 10px, transparent 10px 22px); background-size: 44px 2px; animation: wire 1.4s linear infinite; opacity: .7; }
.ff-station { position: relative; z-index: 1; padding: 16px 14px 18px; border-radius: 22px; text-align: center; background: var(--ink-raised); border: 1px solid rgba(255,255,255,.14);
    border-bottom: 3px solid var(--station-color); box-shadow: 0 16px 44px rgba(0,0,0,.4); animation: rise .7s ease both; transition: transform .3s cubic-bezier(.3,1.6,.5,1); }
.ff-station:hover { transform: translateY(-10px) scale(1.04); box-shadow: 0 24px 60px rgba(0,0,0,.5), 0 0 40px color-mix(in srgb, var(--station-color) 40%, transparent); }
.ff-station-icon { width: 62px; height: 62px; margin: 0 auto 10px; border-radius: 50%; display: grid; place-items: center; font-size: 28px; background: color-mix(in srgb, var(--station-color) 22%, var(--ink-base)); border: 2px solid var(--station-color); }
.ff-station-no { font: 600 11px 'JetBrains Mono', monospace; letter-spacing: .16em; color: var(--station-color); }
.ff-station-title { font: 800 18px 'Bricolage Grotesque', sans-serif; color: var(--text); margin: 2px 0 6px; }
.ff-station-text { font-size: 13px; line-height: 1.45; color: var(--text-dim); }

/* ---------- animations ---------- */
@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { to { background-position: 220% 0; } }
@keyframes breathe { 0%, 100% { filter: brightness(1); } 50% { filter: brightness(1.12); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .45; } }
@keyframes aurora { from { transform: translate(0, 0) scale(1); } to { transform: translate(18vmax, -10vmax) scale(1.25); } }
@keyframes marquee { to { transform: translateX(-50%); } }
@keyframes bounce { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(5px); } }
@keyframes grow-bar { from { width: 0; } }
@keyframes grow-score { from { --score: 0; --n: 0; } }
@keyframes wire { to { background-position: 44px 0; } }
@keyframes sway { 0% { transform: rotate(0deg); } 55% { transform: rotate(calc(var(--tilt) * -.45)); } 100% { transform: rotate(var(--tilt)); } }
@keyframes sway-back { 0% { transform: rotate(0deg); } 55% { transform: rotate(calc(var(--tilt) * .45)); } 100% { transform: rotate(calc(var(--tilt) * -1)); } }
@media (prefers-reduced-motion: reduce) { * { animation-duration: .01ms !important; animation-iteration-count: 1 !important; } }

/* ---------- small screens ---------- */
@media (max-width: 900px) {
  .block-container { padding: .8rem 1rem 170px 1rem; }
  .ff-pipeline { grid-template-columns: 1fr 1fr; } .ff-pipeline::before { display: none; }
  .ff-podium { flex-direction: column; align-items: stretch; } .ff-podium-slot { max-width: none; } .ff-pedestal { display: none; }
  .ff-board-row { grid-template-columns: 40px 1fr 70px; } .ff-board-bar, .ff-board-row .ff-move { display: none; }
  .ff-twin-scales { grid-template-columns: 1fr; } .ff-bar-row { grid-template-columns: 1fr 70px; } .ff-bar-track { grid-column: 1 / -1; order: 3; }
  .ff-section-title { font-size: 30px; } .ff-section-no { font-size: 40px; }
}
"""

# Final <style> block, injected once when the app starts.
APP_CSS = "<style>" + FONT_IMPORT + CSS_VARIABLES + CSS_RULES + "</style>"

# ================================================================================================================
# 4. HERO BANNER (Three.js particle sculpture, runs inside an iframe)
# ================================================================================================================
# What it shows: 9,000 glowing particles that morph through four shapes while the caption changes:
#   TWINS  two identical busts (orange and teal)        BLIND  the particles scatter as name / age / gender tags fly away
#   SKILLS a resume sheet with the skill lines lit up   FAIR   balanced scales of justice
# The mouse tilts the sculpture. Clicking a phase pill jumps to that shape. If WebGL or the CDN is not available the
# banner still shows the title and the captions on a gradient, so nothing is ever invisible.
# Height of the hero banner iframe in pixels.
HERO_HEIGHT = 500
# The hero banner is a small stand-alone web page (HTML, CSS and JavaScript) that Streamlit shows inside an iframe. It
# draws the animated particle shapes with Three.js.
HERO_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@600&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: #070B16; }
  #hero { position: relative; width: 100%; height: 100%; overflow: hidden; color: #F4F1EA; font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif;
          background: radial-gradient(700px 420px at 78% 45%, rgba(46,230,197,.22), transparent 65%), radial-gradient(600px 400px at 8% 10%, rgba(255,106,61,.28), transparent 60%),
                      linear-gradient(120deg, #070B16, #0F1830 60%, #0B1020); }
  canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
  .copy { position: absolute; left: 5%; top: 50%; transform: translateY(-50%); max-width: 46%; z-index: 2; }
  .eyebrow { font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .22em; color: #FFC24B; text-transform: uppercase; margin-bottom: 14px; }
  h1 { margin: 0; font: 800 clamp(54px, 8.4vw, 104px)/.92 'Bricolage Grotesque', 'Segoe UI', sans-serif; letter-spacing: -.04em; color: #F4F1EA; text-shadow: 0 6px 40px rgba(0,0,0,.6); }
  h1 em { font: italic 400 1.04em 'Instrument Serif', Georgia, serif; letter-spacing: -.02em; color: #FFC24B; }
  @supports ((-webkit-background-clip: text) or (background-clip: text)) {
    h1 em { background: linear-gradient(100deg, #FF6A3D, #FFC24B 55%, #2EE6C5); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; color: transparent; }
  }
  #caption { min-height: 66px; margin: 18px 0 18px; font: 500 clamp(16px, 1.6vw, 21px)/1.4 'Bricolage Grotesque', sans-serif; color: #E7E9F0; max-width: 470px; transition: opacity .35s ease, transform .35s ease; }
  #caption.swap { opacity: 0; transform: translateY(8px); }
  .phases { display: flex; flex-wrap: wrap; gap: 8px; }
  .phase { cursor: pointer; font: 600 11px 'JetBrains Mono', monospace; letter-spacing: .16em; text-transform: uppercase; padding: 8px 14px; border-radius: 999px;
           color: #C7CEDF; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.22); transition: all .25s ease; position: relative; overflow: hidden; }
  .phase:hover { color: #fff; border-color: #2EE6C5; }
  .phase.on { color: #14100c; background: linear-gradient(100deg, #FF6A3D, #FFC24B); border-color: transparent; }
  .tag { position: absolute; z-index: 2; opacity: 0; font: 600 12px 'JetBrains Mono', monospace; letter-spacing: .14em; color: #FFC24B; padding: 5px 12px; border-radius: 8px;
         background: rgba(0,0,0,.65); border: 1px solid #FFC24B; text-decoration: line-through; text-decoration-color: #FF6A3D; text-decoration-thickness: 2px; }
  #hero[data-phase="1"] .tag { animation: fly 3.4s ease-in both; animation-delay: var(--d); }
  @keyframes fly { 0% { opacity: 0; transform: translate(0, 24px) scale(.8); } 15% { opacity: 1; transform: translate(0, 0) scale(1); } 70% { opacity: 1; } 100% { opacity: 0; transform: translate(var(--dx), -90px) rotate(var(--rot)) scale(.7); } }
  #hero.no-webgl .tag { display: none; }
  @media (max-width: 780px) { .copy { left: 6%; right: 6%; top: 24px; transform: none; max-width: none; } h1 { font-size: 54px; } #caption { max-width: none; min-height: 52px; } .tag { display: none; } }
</style>
<div id="hero" data-phase="0">
  <canvas id="stage"></canvas>
  <div class="copy">
    <div class="eyebrow">OlympAI 3.0 · FairHire · Team CODE FORGERS</div>
    <h1>Fair<em>Forge</em></h1>
    <div id="caption" aria-live="polite"></div>
    <div class="phases" id="phases"></div>
  </div>
  <span class="tag" style="left:58%; top:62%; --d:.2s; --dx:-60px; --rot:-8deg">NAME</span>
  <span class="tag" style="left:70%; top:30%; --d:.5s; --dx:50px; --rot:6deg">AGE</span>
  <span class="tag" style="left:82%; top:58%; --d:.8s; --dx:70px; --rot:10deg">GENDER</span>
  <span class="tag" style="left:64%; top:78%; --d:1.1s; --dx:-30px; --rot:-5deg">PLACE</span>
  <span class="tag" style="left:78%; top:20%; --d:1.4s; --dx:40px; --rot:-9deg">COLLEGE</span>
  <span class="tag" style="left:88%; top:76%; --d:1.7s; --dx:60px; --rot:7deg">PHOTO</span>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
  // ---- the four story phases: pill label + caption shown while that shape is on screen ----
  var PHASES = [
    { label: 'Twins',  text: 'Two identical candidates. Different names, faces and ages.' },
    { label: 'Blind',  text: 'Name, gender, age and place are hidden before any model reads a word.' },
    { label: 'Skills', text: 'Only skills and experience are left to read, and every line is explained.' },
    { label: 'Fair',   text: 'One honest score. Twins get the same number, and we prove it.' }
  ];
  var heroEl = document.getElementById('hero'), captionEl = document.getElementById('caption'), phasesEl = document.getElementById('phases');
  var currentPhase = 0;

  function showCaption(index) {                       // fade the caption out, swap the text, fade it back in
    captionEl.classList.add('swap');
    setTimeout(function () {
      captionEl.textContent = PHASES[index].text; captionEl.classList.remove('swap');
      heroEl.setAttribute('data-phase', String(index));
      Array.prototype.forEach.call(phasesEl.children, function (pill, i) { pill.classList.toggle('on', i === index); });
    }, 300);
  }
  PHASES.forEach(function (phase, i) {
    var pill = document.createElement('div'); pill.className = 'phase' + (i === 0 ? ' on' : ''); pill.textContent = phase.label;
    pill.onclick = function () { goToPhase(i); restartTimer(); }; phasesEl.appendChild(pill);
  });
  captionEl.textContent = PHASES[0].text;

  var advanceTimer = null;
  function restartTimer() { clearInterval(advanceTimer); advanceTimer = setInterval(function () { goToPhase((currentPhase + 1) % PHASES.length); }, 5600); }
  var goToPhase = function (index) { currentPhase = index; showCaption(index); };   // replaced below once WebGL is ready

  if (!window.THREE) { heroEl.classList.add('no-webgl'); restartTimer(); return; }   // no CDN: captions still cycle on the gradient
  var renderer;
  try { renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('stage'), alpha: true, antialias: true }); }
  catch (error) { heroEl.classList.add('no-webgl'); restartTimer(); return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  // ---- particle shapes: each builder returns one point {p: [x,y,z], c: [r,g,b]} for a position f in [0,1) ----
  var PARTICLE_COUNT = window.innerWidth < 700 ? 5200 : 9000;
  var C_EMBER = [1, .42, .24], C_TEAL = [.18, .9, .77], C_GOLD = [1, .76, .29], C_VIOLET = [.61, .48, 1], C_PALE = [.96, .95, .92];
  var rand = function (a, b) { return a + Math.random() * (b - a); };
  function onSphere(radius) { var u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u); return [radius * s * Math.cos(t), radius * u, radius * s * Math.sin(t)]; }
  function buildShape(builder) {
    var positions = new Float32Array(PARTICLE_COUNT * 3), colors = new Float32Array(PARTICLE_COUNT * 3);
    for (var i = 0; i < PARTICLE_COUNT; i++) { var point = builder(i / PARTICLE_COUNT); positions.set(point.p, i * 3); colors.set(point.c, i * 3); }
    return { positions: positions, colors: colors };
  }
  function twinsPoint(f) {                            // two busts: head sphere + neck + half-ellipsoid shoulders
    var isLeft = f < .5, centerX = isLeft ? -1.3 : 1.3, color = isLeft ? C_EMBER : C_TEAL, part = (f % .5) / .5, p;
    if (part < .45) { var h = onSphere(.62); p = [centerX + h[0], .78 + h[1] * 1.08, h[2] * .9]; }
    else if (part < .55) { var a = Math.random() * 6.283; p = [centerX + Math.cos(a) * .2, rand(0, .22), Math.sin(a) * .18]; }
    else { var s = onSphere(1); p = [centerX + s[0] * 1.08, -.5 + Math.abs(s[1]) * .64, s[2] * .5]; }
    return { p: p, c: color };
  }
  function dustPoint() {                              // identity details turned to dust: a wide, thin, swirling cloud
    var angle = Math.random() * 6.283, radius = Math.pow(Math.random(), .6) * 4.4, pick = Math.random();
    return { p: [Math.cos(angle) * radius, rand(-1.6, 1.6) * (1 - radius / 6), Math.sin(angle) * radius * .8], c: pick < .5 ? C_PALE : (pick < .8 ? C_GOLD : C_VIOLET) };
  }
  function resumePoint() {                            // a resume sheet: frame, header block and nine text lines (every third line is a lit-up skill line)
    var halfW = 1.15, halfH = 1.65, kind = Math.random(), p, c = C_PALE;
    if (kind < .14) { var edge = Math.floor(Math.random() * 4), t = Math.random();
      p = edge === 0 ? [-halfW + t * 2 * halfW, halfH, 0] : edge === 1 ? [-halfW + t * 2 * halfW, -halfH, 0] : edge === 2 ? [-halfW, -halfH + t * 2 * halfH, 0] : [halfW, -halfH + t * 2 * halfH, 0]; c = C_VIOLET; }
    else if (kind < .25) { p = [rand(-halfW + .15, -.1), rand(halfH - .5, halfH - .28), 0]; c = C_GOLD; }
    else { var row = Math.floor(Math.random() * 9), lengths = [1.7, 2.0, 1.4, 1.9, 1.1, 1.8, 1.5, .9, 1.6];
      p = [-halfW + .15 + Math.random() * lengths[row] * .95, halfH - .95 - row * .29, 0]; c = (row % 3 === 1) ? C_TEAL : [.7, .72, .8]; }
    p[2] += rand(-.03, .03); return { p: p, c: c };
  }
  function scalesPoint() {                            // scales of justice: pillar, base, beam, chains and two level pans
    var k = Math.random(), beamY = 1.15, side = Math.random() < .5 ? -1 : 1, panColor = side > 0 ? C_TEAL : C_EMBER, p, c = C_GOLD, a, r;
    if (k < .17) { p = [rand(-.04, .04), rand(-1.4, beamY), rand(-.04, .04)]; }
    else if (k < .27) { a = Math.random() * 6.283; r = Math.sqrt(Math.random()) * .9; p = [Math.cos(a) * r, -1.42 + rand(0, .06), Math.sin(a) * r * .6]; }
    else if (k < .39) { p = [rand(-1.55, 1.55), beamY + rand(-.03, .03), rand(-.03, .03)]; }
    else if (k < .43) { var s = onSphere(.13); p = [s[0], beamY + .2 + s[1], s[2]]; }
    else if (k < .58) { var endX = side * 1.5, rimX = endX + (Math.random() < .5 ? -.62 : .62), t2 = Math.random(); p = [endX + (rimX - endX) * t2, beamY + (.05 - beamY) * t2, 0]; c = panColor; }
    else { a = Math.random() * 6.283; r = Math.sqrt(Math.random()) * .65; p = [side * 1.5 + Math.cos(a) * r, .05 - (1 - (r / .65) * (r / .65)) * .24, Math.sin(a) * r * .5]; c = panColor; }
    return { p: p, c: c };
  }
  var shapes = [buildShape(twinsPoint), buildShape(dustPoint), buildShape(resumePoint), buildShape(scalesPoint)];

  // ---- shader: each particle flies from its start position to its end position, with a swirl in the middle of the trip ----
  var vertexShader = [
    'attribute vec3 aStart; attribute vec3 aEnd; attribute vec3 aColorStart; attribute vec3 aColorEnd; attribute float aSeed;',
    'uniform float uMix; uniform float uTime; uniform float uPixelRatio; varying vec3 vColor; varying float vAlpha;',
    'void main() {',
    '  float delay = aSeed * 0.35; float t = clamp((uMix - delay) / 0.65, 0.0, 1.0); t = t * t * (3.0 - 2.0 * t);',
    '  vec3 p = mix(aStart, aEnd, t); float swirl = sin(t * 3.14159);',
    '  p += swirl * 1.3 * vec3(sin(aSeed * 40.0 + uTime * 1.5), cos(aSeed * 55.0 + uTime * 1.2), sin(aSeed * 23.0 + uTime));',
    '  p += 0.035 * vec3(sin(uTime * 0.8 + aSeed * 30.0), cos(uTime * 0.9 + aSeed * 20.0), sin(uTime * 0.7 + aSeed * 10.0));',
    '  vec4 mv = modelViewMatrix * vec4(p, 1.0);',
    '  gl_PointSize = (2.3 + 2.2 * swirl + aSeed * 1.3) * uPixelRatio * (6.5 / -mv.z);',
    '  gl_Position = projectionMatrix * mv; vColor = mix(aColorStart, aColorEnd, t); vAlpha = 0.6 + 0.4 * swirl;',
    '}'].join('\\n');
  var fragmentShader = [
    'varying vec3 vColor; varying float vAlpha;',
    'void main() { float d = length(gl_PointCoord - 0.5); if (d > 0.5) discard; gl_FragColor = vec4(vColor, smoothstep(0.5, 0.0, d) * vAlpha); }'].join('\\n');

  var geometry = new THREE.BufferGeometry(), seeds = new Float32Array(PARTICLE_COUNT);
  for (var i = 0; i < PARTICLE_COUNT; i++) seeds[i] = Math.random();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PARTICLE_COUNT * 3), 3));   // required by three.js, real positions come from the shader
  geometry.setAttribute('aStart', new THREE.BufferAttribute(new Float32Array(shapes[0].positions), 3));
  geometry.setAttribute('aEnd', new THREE.BufferAttribute(new Float32Array(shapes[0].positions), 3));
  geometry.setAttribute('aColorStart', new THREE.BufferAttribute(new Float32Array(shapes[0].colors), 3));
  geometry.setAttribute('aColorEnd', new THREE.BufferAttribute(new Float32Array(shapes[0].colors), 3));
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(seeds, 1));
  var material = new THREE.ShaderMaterial({ uniforms: { uMix: { value: 1 }, uTime: { value: 0 }, uPixelRatio: { value: renderer.getPixelRatio() } },
    vertexShader: vertexShader, fragmentShader: fragmentShader, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
  var sculpture = new THREE.Points(geometry, material); sculpture.frustumCulled = false;

  var scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 100), group = new THREE.Group();
  camera.position.z = 7.4; group.add(sculpture); scene.add(group);
  var starCount = 450, starPositions = new Float32Array(starCount * 3);          // faint background stars
  for (var s = 0; s < starCount * 3; s++) starPositions[s] = (Math.random() - .5) * 26;
  var starGeometry = new THREE.BufferGeometry(); starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
  var stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ size: .035, color: 0xF4F1EA, transparent: true, opacity: .55 })); scene.add(stars);

  // ---- switching shapes: copy the current shape into "start", the next into "end", and restart the 0 -> 1 progress ----
  var transitionStart = performance.now() - 9999;
  goToPhase = function (nextIndex) {
    if (nextIndex === currentPhase) return;
    geometry.attributes.aStart.array.set(shapes[currentPhase].positions); geometry.attributes.aColorStart.array.set(shapes[currentPhase].colors);
    geometry.attributes.aEnd.array.set(shapes[nextIndex].positions); geometry.attributes.aColorEnd.array.set(shapes[nextIndex].colors);
    ['aStart', 'aEnd', 'aColorStart', 'aColorEnd'].forEach(function (name) { geometry.attributes[name].needsUpdate = true; });
    currentPhase = nextIndex; transitionStart = performance.now(); showCaption(nextIndex);
  };

  var pointerX = 0, pointerY = 0;
  window.addEventListener('mousemove', function (e) { pointerX = e.clientX / window.innerWidth - .5; pointerY = e.clientY / window.innerHeight - .5; });
  function resize() {                                  // keep the sculpture on the right on wide screens, below the text on phones
    var width = heroEl.clientWidth, height = heroEl.clientHeight;
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
    if (width > 780) { var visibleWidth = 2 * camera.position.z * Math.tan(25 * Math.PI / 180) * camera.aspect; group.position.set(visibleWidth * .215, -.1, 0); group.scale.setScalar(1.38); }
    else { group.position.set(0, -1.5, 0); group.scale.setScalar(.72); }
  }
  window.addEventListener('resize', resize); resize();

  (function frame(now) {
    material.uniforms.uTime.value = now / 1000;
    material.uniforms.uMix.value = Math.min(1, (now - transitionStart) / 2800);
    group.rotation.y += ((Math.sin(now / 2800) * .38 + pointerX * .9) - group.rotation.y) * .05;
    group.rotation.x += (pointerY * .45 - group.rotation.x) * .05;
    stars.rotation.y = now / 90000;
    renderer.render(scene, camera); requestAnimationFrame(frame);
  })(performance.now());
  restartTimer();
})();
</script>
"""

# ================================================================================================================
# 5. SMALL HTML BUILDERS
# ================================================================================================================
# Every custom HTML block in this file is shown through this helper.
def render_html(markup):
    """Show a block of HTML. Streamlit's Markdown treats indented lines as code, so every line is trimmed and joined first."""
    single_line_markup = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(single_line_markup, unsafe_allow_html=True)


def escape(value):
    """Make any value safe to place inside HTML (resume text can contain < > & characters)."""
    return html.escape(str(value))


def with_alpha(hex_color, alpha):
    """Turn '#RRGGBB' into 'rgba(r,g,b,alpha)' (Plotly needs rgba for transparent fills)."""
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def build_section_title(number, title, subtitle=""):
    """Big editorial heading: outlined number + serif italic title + small explanation."""
    subtitle_html = f"<div class='ff-section-sub'>{escape(subtitle)}</div>" if subtitle else ""
    number_html = f"<div class='ff-section-no'>{number}</div>" if number else ""       # sub-sections have no big number
    return (f"<div class='ff-section'>{number_html}"
            f"<div><div class='ff-section-title'>{escape(title)}</div>{subtitle_html}</div></div>")


def build_panel_heading(accent_class, step_number, title, hint):
    """Heading of a console panel. The hidden .ff-panel-marker is what makes the surrounding Streamlit column look like a glass card."""
    return (f"<div class='ff-panel-marker {accent_class}'></div><div class='ff-step'><span class='ff-step-no'>{step_number}</span>"
            f"<span class='ff-step-title'>{escape(title)}</span></div><div class='ff-step-hint'>{escape(hint)}</div>")


def build_marquee():
    """The slow ribbon of keywords under the hero (text is repeated twice so the loop has no gap)."""
    words = ["BLIND SCREENING", "GLASS-BOX SCORES", "TWIN TESTED", "SKILLS ONLY", "A HUMAN DECIDES", "EXPLAINED WORD BY WORD"]
    one_round = "".join(f"<span>{word}</span><span>✦</span>" for word in words)
    return f"<div class='ff-marquee'><div class='ff-marquee-track'>{one_round}{one_round}</div></div>"


def build_recipe_bar(score_weights):
    """Stacked colour bar showing how much each signal counts in the final score (updates live with the faders)."""
    total_weight = sum(score_weights.values()) or 1
    segments = "".join(f"<div style='width:{100 * score_weights[name] / total_weight:.1f}%; background:{SIGNAL_STYLES[name]['color']}'></div>" for name in PART_NAMES)
    legend = "".join(f"<span style='--dot:{SIGNAL_STYLES[name]['color']}'>{escape(name)} {100 * score_weights[name] / total_weight:.0f}%</span>" for name in PART_NAMES)
    return f"<div class='ff-recipe'>{segments}</div><div class='ff-recipe-legend'>{legend}</div>"


def build_status_leds(status):
    """Small glowing LEDs: green = that model part is working, amber = fallback in use."""
    leds = "".join(f"<span class='ff-led' style='--led:{TEAL if is_ok else GOLD}'>{escape(name)}</span>" for name, is_ok in status.items())
    return f"<div class='ff-leds'>{leds}</div>"


def describe_rank_move(rank_before, rank_after):
    """Return (css class, text) describing how a candidate's rank changed once identity was hidden."""
    places_moved = rank_before - rank_after                     # positive = moved up after hiding identity
    if places_moved > 0:
        return "ff-move-up", f"▲ up {places_moved} when identity hidden"
    if places_moved < 0:
        return "ff-move-down", f"▼ down {-places_moved} when identity hidden"
    return "ff-move-same", "● same rank either way"


# Builds the top-three podium cards shown at the start of the Ranking tab.
def build_podium(ordered_results):
    """Podium for the top three candidates (2nd | 1st | 3rd), each card with an animated score ring and a pedestal."""
    medal_colors = {1: GOLD, 2: TEAL, 3: EMBER}
    pedestal_heights = {1: 96, 2: 66, 3: 46}
    top_three = ordered_results[:3]
    visual_order = [top_three[1], top_three[0], top_three[2]] if len(top_three) == 3 else top_three    # 1st goes in the middle
    slots = []
    for candidate in visual_order:
        rank = candidate["rank"]
        score_percent = round(100 * candidate["a"]["hidden"]["final"])
        move_class, move_text = describe_rank_move(candidate["rank_before"], candidate["rank"])
        crown = "<div class='ff-crown'>♛</div>" if rank == 1 else ""
        slots.append(
            f"<div class='ff-podium-slot'><div class='ff-podium-card' style='--ring-color:{medal_colors[rank]}; --score:{score_percent}; --n:{score_percent}'>{crown}"
            f"<div class='ff-rank-tag'>RANK {rank}</div>"
            f"<div class='ff-ring'><div class='ff-ring-core'><span class='ff-count'></span><small>OUT OF 100</small></div></div>"
            f"<div class='ff-card-name'>{escape(candidate['label'])}</div>"
            f"<span class='ff-move {move_class}'>{move_text}</span>"
            f"<div class='ff-card-sub'>BEFORE HIDING: #{candidate['rank_before']}</div></div>"
            f"<div class='ff-pedestal' style='height:{pedestal_heights[rank]}px; --ring-color:{medal_colors[rank]}'>{rank}</div></div>")
    return f"<div class='ff-podium'>{''.join(slots)}</div>"


# Builds the list with one row per candidate (rank, score bar, rank movement).
def build_leaderboard(ordered_results):
    """One row per candidate: rank, name, score bar, score and rank movement."""
    rows = []
    for candidate in ordered_results:
        score_percent = 100 * candidate["a"]["hidden"]["final"]
        move_class, move_text = describe_rank_move(candidate["rank_before"], candidate["rank"])
        rows.append(f"<div class='ff-board-row' style='--w:{score_percent:.1f}%'><div class='ff-board-rank'>#{candidate['rank']}</div>"
                    f"<div class='ff-board-name'>{escape(candidate['label'])}</div><div class='ff-board-bar'><div></div></div>"
                    f"<div class='ff-board-score'>{score_percent:.1f}%</div><span class='ff-move {move_class}'>{escape(move_text.split(' when')[0])}</span></div>")
    return f"<div class='ff-board'>{''.join(rows)}</div>"


def build_table(column_names, table_rows, signed_columns=()):
    """Dark HTML table. signed_columns = names of columns whose numbers are coloured green (+) or orange (-)."""
    header = "".join(f"<th>{escape(name)}</th>" for name in column_names)
    body_rows = []
    for row in table_rows:
        cells = []
        for name, value in zip(column_names, row):
            css_class = ""
            if name in signed_columns and isinstance(value, (int, float)):
                css_class = " class='ff-up'" if value > 0 else (" class='ff-down'" if value < 0 else "")
                value = f"{value:+d}" if isinstance(value, int) else f"{value:+.1f}"
            cells.append(f"<td{css_class}>{escape(value)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='ff-table-wrap'><table class='ff-table'><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def build_stat_tile(label, value_text, note, color, unit=""):
    """One big-number tile with a coloured top edge."""
    unit_html = f"<small>{escape(unit)}</small>" if unit else ""
    return (f"<div class='ff-tile' style='--tile-color:{color}'><div class='ff-tile-label'>{escape(label)}</div>"
            f"<div class='ff-tile-value'>{escape(value_text)}{unit_html}</div><div class='ff-tile-note'>{escape(note)}</div></div>")


def build_proof_tile(label, before_text, after_text, note, color=TEAL):
    """Tile that shows a bad 'before' number crossed out and the good 'after' number glowing next to it."""
    return (f"<div class='ff-tile' style='--tile-color:{color}'><div class='ff-tile-label'>{escape(label)}</div>"
            f"<div class='ff-proof'><span class='ff-proof-before'>{escape(before_text)}</span><span class='ff-proof-arrow'>→</span>"
            f"<span class='ff-proof-after'>{escape(after_text)}</span></div><div class='ff-tile-note'>{escape(note)}</div></div>")


def build_pipeline():
    """The five-step pipeline as glowing stations joined by an animated wire."""
    stations = "".join(
        f"<div class='ff-station' style='--station-color:{color}; animation-delay:{index * 0.09:.2f}s'><div class='ff-station-icon'>{icon}</div>"
        f"<div class='ff-station-no'>STEP {index + 1}</div><div class='ff-station-title'>{escape(title)}</div><div class='ff-station-text'>{escape(text)}</div></div>"
        for index, (icon, title, text, color) in enumerate(PIPELINE_STEPS))
    return f"<div class='ff-pipeline'>{stations}</div>"


def build_accuracy_bars(accuracy_groups, chance_level=0.04):
    """Bars for the model accuracies saved in metrics.json (a teal tick marks pure chance, 4% for 24 job families)."""
    entries = []
    for group_name, value in accuracy_groups.items():
        if isinstance(value, dict):
            for variant_name, variant_value in value.items():
                number = variant_value.get("test_acc") if isinstance(variant_value, dict) else variant_value
                if isinstance(number, (int, float)):
                    entries.append((f"{group_name}: {variant_name}", number))
        elif isinstance(value, (int, float)):
            entries.append((group_name, value))
    rows = []
    for name, number in entries:
        share = number if number <= 1 else number / 100
        rows.append(f"<div class='ff-bar-row'><div>{escape(name)}</div><div class='ff-bar-track' style='--w:{100 * share:.1f}%; --chance:{100 * chance_level:.0f}%'>"
                    f"<div class='ff-bar-fill'></div><div class='ff-bar-chance'></div></div><div class='ff-bar-value'>{100 * share:.1f}%</div></div>")
    return "".join(rows)


def build_balance_scale(tilt_degrees, beam_color):
    """SVG balance scale. Positive tilt turns the beam clockwise (right pan heavier). The pans stay level because they counter-rotate."""
    def pan(pivot_x, color):                                    # a hanging pan: two strings and a shallow bowl, pivoting at the beam end
        return (f"<g class='ff-pan' style='transform-origin:{pivot_x}px 40px'><path d='M{pivot_x} 40 L{pivot_x - 24} 92 M{pivot_x} 40 L{pivot_x + 24} 92' stroke='{TEXT_DIM}' stroke-width='1.5' fill='none'/>"
                f"<path d='M{pivot_x - 30} 92 Q{pivot_x} 116 {pivot_x + 30} 92 Z' fill='{color}' fill-opacity='.85' stroke='{color}' stroke-width='2'/></g>")
    return (f"<svg viewBox='0 0 220 150' role='img' aria-label='balance scale'><g style='--tilt:{tilt_degrees:.2f}deg'>"
            f"<path d='M110 40 L110 128' stroke='{GOLD}' stroke-width='5' stroke-linecap='round'/><ellipse cx='110' cy='132' rx='40' ry='6' fill='{GOLD}' fill-opacity='.7'/>"
            f"<g class='ff-beam'><line x1='28' y1='40' x2='192' y2='40' stroke='{beam_color}' stroke-width='6' stroke-linecap='round'/><circle cx='110' cy='40' r='8' fill='{GOLD}'/>"
            f"{pan(28, EMBER)}{pan(192, TEAL)}</g></g></svg>")


# Builds the Twin Test card for one candidate, using two balance scales.
def build_twin_card(candidate_label, twin_result):
    """Twin Test card: two balance scales. 'Identity visible' leans with the score gap, 'identity hidden' should sit level."""
    points_per_degree = 4                                       # exaggeration: every score point tilts the beam 4 degrees (max 22)

    def tilt_for(twin_scores):
        return max(-22.0, min(22.0, 100 * (twin_scores["twin B"] - twin_scores["twin A"]) * points_per_degree))

    visible, hidden = twin_result["identity visible"], twin_result["identity hidden"]
    verdict_class, verdict_text = ("ff-pass", "PASS") if twin_result["pass"] else ("ff-check", "CHECK")
    return (f"<div class='ff-twin-card'><div class='ff-twin-head'><span>{escape(candidate_label)}</span><span class='ff-verdict {verdict_class}'>{verdict_text}</span></div>"
            f"<div class='ff-twin-scales'>"
            f"<div class='ff-twin-scale'>{build_balance_scale(tilt_for(visible), EMBER)}<div class='ff-scale-cap'>Identity visible</div>"
            f"<div class='ff-scale-gap' style='color:{EMBER}'>{100 * visible['gap']:.2f}<small> pts gap</small></div></div>"
            f"<div class='ff-twin-scale'>{build_balance_scale(tilt_for(hidden), TEAL)}<div class='ff-scale-cap'>Identity hidden</div>"
            f"<div class='ff-scale-gap' style='color:{TEAL}'>{100 * hidden['gap']:.2f}<small> pts gap</small></div></div></div></div>")


IDENTITY_TAG_PATTERN = re.compile(r"\[(NAME|PLACE|EMAIL|PHONE|URL|COLLEGE|PRONOUN|TITLE|YEAR)\]")
# Colour of the label that shows which section the CNN put a line into.
SECTION_COLORS = {"HEADER": TEXT_FAINT, "SUMMARY": VIOLET, "SKILLS": TEAL, "EXPERIENCE": GOLD, "EDUCATION": SKY, "OTHER": "#C9CFDD"}


# Left sheet in the Hidden identity tab: the resume as the original text.
def build_raw_paper(raw_text, character_limit=3500):
    """The resume as a person would see it (light paper, identity visible)."""
    safe_text = escape(raw_text[:character_limit]).replace("\n", "<br>")
    return f"<div class='ff-paper ff-paper-raw'><div class='ff-paper-title'>What a person would see</div>{safe_text}</div>"


# Right sheet in the Hidden identity tab: the resume as the models see it, with section labels.
def build_clean_paper(hidden_lines, section_labels, character_limit=3500):
    """The resume as the models see it: identity replaced by black redaction bars, every line tagged with its CNN section."""
    html_lines, used_characters = [], 0
    for line, section in zip(hidden_lines, section_labels):
        used_characters += len(line)
        if used_characters > character_limit:
            break
        marked_line = IDENTITY_TAG_PATTERN.sub(lambda match: f"<span class='ff-redact'>{match.group(1)}</span>", escape(line))
        html_lines.append(f"<div class='ff-line'><span class='ff-sec' style='--sec-color:{SECTION_COLORS.get(section, TEXT_FAINT)}'>{section}</span><span>{marked_line}</span></div>")
    return f"<div class='ff-paper ff-paper-clean'><div class='ff-paper-title'>What the scoring models see</div>{''.join(html_lines)}</div>"


# ================================================================================================================
# 6. PLOTLY CHART BUILDERS (all share one dark, glowing look)
# ================================================================================================================
# Every Plotly chart passes through this function so they all share the same dark theme.
def style_chart(figure, height=380, title=None):
    """Shared chart look: transparent background, light text, faint grid lines, neon colour cycle."""
    figure.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         font=dict(family="Bricolage Grotesque, Segoe UI, sans-serif", color=TEXT_BRIGHT, size=13),
                         margin=dict(l=14, r=14, t=56 if title else 20, b=14),
                         title=dict(text=title, font=dict(family="Bricolage Grotesque, sans-serif", size=20, color=TEXT_BRIGHT)) if title else None,
                         colorway=CANDIDATE_COLORS, legend=dict(font=dict(color=TEXT_BRIGHT)),
                         hoverlabel=dict(bgcolor=INK_RAISED, font=dict(color=TEXT_BRIGHT, family="JetBrains Mono, monospace"), bordercolor=TEAL))
    figure.update_xaxes(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE, linecolor=GRID_LINE, tickfont=dict(color=TEXT_DIM), title_font=dict(color=TEXT_DIM))
    figure.update_yaxes(gridcolor=GRID_LINE, zerolinecolor=GRID_LINE, linecolor=GRID_LINE, tickfont=dict(color=TEXT_DIM), title_font=dict(color=TEXT_DIM))
    return figure


# Streamlit re-runs this whole script on every click. Caching makes sure the models are loaded only once.
@st.cache_resource(show_spinner="Loading models (the first run downloads the MiniLM language model)...")
def load_engine():
    """Load every trained model once and keep it for the whole session."""
    return FairForge()


def extract_text(uploaded_file):
    """Read text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def build_radar_chart(ordered_results):
    """Radar ('spider') chart: the shape of each candidate's four signal scores."""
    figure = go.Figure()
    for index, candidate in enumerate(ordered_results):
        part_scores = candidate["a"]["hidden"]["parts"]
        values = [part_scores[name] for name in PART_NAMES]
        color = CANDIDATE_COLORS[index % len(CANDIDATE_COLORS)]
        # One filled shape per candidate; the four corners are the four signals.
        figure.add_trace(go.Scatterpolar(r=values + values[:1], theta=PART_NAMES + PART_NAMES[:1], fill="toself", name=candidate["label"],
                                         fillcolor=with_alpha(color, 0.22), line=dict(width=3, color=color), marker=dict(size=7, color=color)))
    figure.update_layout(polar=dict(bgcolor="rgba(255,255,255,0.04)",
                                    radialaxis=dict(range=[0, 1], tickformat=".0%", gridcolor=GRID_LINE, linecolor=GRID_LINE, tickfont=dict(color=TEXT_DIM)),
                                    angularaxis=dict(gridcolor=GRID_LINE, linecolor=GRID_LINE, tickfont=dict(color=TEXT_BRIGHT, size=12))),
                         legend=dict(orientation="h", y=-0.08))
    return style_chart(figure, 430, "The shape of each score")


def build_slope_chart(ordered_results):
    """Slope chart: how every rank moves once identity is hidden (teal = moved up, orange = moved down)."""
    figure, candidate_count = go.Figure(), len(ordered_results)
    for candidate in ordered_results:
        rank_before, rank_after = candidate["rank_before"], candidate["rank"]
        line_color = TEAL if rank_after < rank_before else (EMBER if rank_after > rank_before else TEXT_FAINT)
        figure.add_trace(go.Scatter(x=[0, 1], y=[rank_before, rank_after], mode="lines+markers+text", line=dict(color=line_color, width=5),
                                    marker=dict(size=15, color=line_color, line=dict(color="#fff", width=2)),
                                    text=[f"{candidate['label']}  #{rank_before}", f"#{rank_after}  {candidate['label']}"], textposition=["middle left", "middle right"],
                                    textfont=dict(color=TEXT_BRIGHT, size=13), hoverinfo="skip", showlegend=False))
    figure.update_xaxes(range=[-0.9, 1.9], tickvals=[0, 1], ticktext=["Identity visible", "Identity hidden"], showgrid=False, zeroline=False)
    figure.update_yaxes(autorange="reversed", dtick=1, range=[candidate_count + .5, .5], showgrid=False, zeroline=False, visible=False)
    return style_chart(figure, 120 + 70 * candidate_count, "Rank before and after hiding identity")


def build_waterfall_chart(explanation):
    """Waterfall: each word pushes the match up (teal) or down (orange)."""
    word_effects = explanation["up"] + explanation["down"]
    figure = go.Figure(go.Waterfall(x=[word for word, _ in word_effects] + ["net of shown words"], y=[effect for _, effect in word_effects] + [0],
                                    measure=["relative"] * len(word_effects) + ["total"], connector=dict(line=dict(color=TEXT_FAINT)),
                                    increasing=dict(marker=dict(color=TEAL)), decreasing=dict(marker=dict(color=EMBER)), totals=dict(marker=dict(color=GOLD))))
    figure.update_layout(showlegend=False, yaxis_title=f"Push towards '{explanation['family']}'")
    return style_chart(figure, 380, "Glass box: word by word (SHAP)")


def build_sankey_chart(hidden_counts):
    """Sankey: which kinds of identity details were found and funnelled into 'Hidden from the models'."""
    found_counts = {tag.strip("[]").title(): count for tag, count in hidden_counts.items() if count}
    if not found_counts:
        return None
    labels = list(found_counts) + ["Hidden from the models"]
    node_colors = [VIOLET, EMBER, GOLD, TEAL, SKY, "#C9CFDD", EMBER, VIOLET, GOLD][:len(labels) - 1] + [TEXT_BRIGHT]
    figure = go.Figure(go.Sankey(textfont=dict(color=TEXT_BRIGHT, size=13), node=dict(label=labels, pad=22, thickness=18, color=node_colors, line=dict(color="rgba(255,255,255,.4)", width=1)),
                                 link=dict(source=list(range(len(found_counts))), target=[len(found_counts)] * len(found_counts),
                                           value=list(found_counts.values()), color="rgba(255,106,61,.45)")))
    return style_chart(figure, 120 + 38 * len(found_counts), "What was hidden before scoring")


def build_section_sunburst(section_labels):
    """Sunburst: how the resume splits into sections (found by the CNN)."""
    line_counts = pd.Series(section_labels).value_counts()
    slice_colors = [SECTION_COLORS.get(name, TEXT_FAINT) for name in line_counts.index]
    figure = go.Figure(go.Sunburst(labels=["Resume"] + list(line_counts.index), parents=[""] + ["Resume"] * len(line_counts),
                                   values=[int(line_counts.sum())] + [int(value) for value in line_counts.values], branchvalues="total",
                                   marker=dict(colors=[INK_LIFT] + slice_colors, line=dict(color=INK_BASE, width=2)),
                                   textfont=dict(color="#0B1020", size=13), insidetextorientation="radial"))
    figure.update_traces(textfont=dict(color=[TEXT_BRIGHT] + ["#0B1020"] * len(line_counts)))
    return style_chart(figure, 380, "Resume map (lines per section)")


def build_dumbbell_chart(rows, title, threshold=None, x_range=None):
    """Dumbbell chart: two dots joined by a line. rows = [(label, value_when_identity_visible, value_when_hidden)]."""
    figure = go.Figure()
    for label, visible_value, hidden_value in rows:
        figure.add_trace(go.Scatter(x=[visible_value, hidden_value], y=[label, label], mode="lines", line=dict(color="rgba(255,255,255,.35)", width=5),
                                    hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(x=[row[1] for row in rows], y=[row[0] for row in rows], mode="markers", marker=dict(size=18, color=EMBER, line=dict(color="#fff", width=2)), name="Identity visible"))
    figure.add_trace(go.Scatter(x=[row[2] for row in rows], y=[row[0] for row in rows], mode="markers", marker=dict(size=18, color=TEAL, line=dict(color="#fff", width=2)), name="Identity hidden"))
    if threshold is not None:
        figure.add_vline(x=threshold, line=dict(color=GOLD, dash="dash", width=2), annotation_text="pass mark 0.80", annotation_position="top", annotation_font_color=GOLD)
    if x_range:
        figure.update_xaxes(range=x_range)
    figure.update_layout(legend=dict(orientation="h", y=-0.2))
    return style_chart(figure, 130 + 70 * len(rows), title)


def build_galaxy_chart(engine, ordered_results):
    """3D chart: where candidates and the job description sit among the 24 job families (PCA of MiniLM vectors)."""
    candidate_points = [(c["label"], c["a"]["hidden"]["emb"]) for c in ordered_results if c["a"]["hidden"].get("emb") is not None]
    job_embedding = ordered_results[0]["a"]["hidden"].get("emb_jd") if ordered_results else None
    if not candidate_points or job_embedding is None:
        return None
    # Squash the 384-number meaning vectors down to 3 numbers so they can be drawn in 3D.
    pca = PCA(n_components=3).fit(engine.category_centroids)
    family_xyz = pca.transform(engine.category_centroids)
    job_xyz = pca.transform(np.asarray(job_embedding)[None])[0]
    candidate_xyz = pca.transform(np.stack([embedding for _, embedding in candidate_points]))
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=family_xyz[:, 0], y=family_xyz[:, 1], z=family_xyz[:, 2], mode="markers+text", text=engine.categories, textfont=dict(size=9, color=TEXT_DIM),
                                  marker=dict(size=5, color=SKY, opacity=.75), name="Job families", hoverinfo="text"))
    for index, (label, _) in enumerate(candidate_points):       # a thin beam from the job description to every candidate: shorter = closer meaning
        figure.add_trace(go.Scatter3d(x=[job_xyz[0], candidate_xyz[index, 0]], y=[job_xyz[1], candidate_xyz[index, 1]], z=[job_xyz[2], candidate_xyz[index, 2]],
                                      mode="lines", line=dict(color=with_alpha(GOLD, .55), width=4), hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter3d(x=candidate_xyz[:, 0], y=candidate_xyz[:, 1], z=candidate_xyz[:, 2], mode="markers+text", text=[label for label, _ in candidate_points],
                                  textposition="top center", textfont=dict(color=TEXT_BRIGHT, size=12), marker=dict(size=10, color=EMBER, symbol="diamond", line=dict(color="#fff", width=1)), name="Candidates"))
    figure.add_trace(go.Scatter3d(x=[job_xyz[0]], y=[job_xyz[1]], z=[job_xyz[2]], mode="markers+text", text=["Job description"], textposition="top center",
                                  textfont=dict(color=GOLD, size=13), marker=dict(size=13, color=GOLD, symbol="cross"), name="Job description"))
    figure.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
                         legend=dict(orientation="h", y=-0.02))
    return style_chart(figure, 540, "Job-fit galaxy (drag to rotate)")


# ================================================================================================================
# 7. APP START, FORGE CONSOLE AND THE RANKING RUN
# ================================================================================================================
# Apply the CSS before anything else is drawn.
render_html(APP_CSS)

try:
    # Load the scoring engine. If the model files are missing, show an error and stop.
    engine = load_engine()
except Exception as startup_error:                                  # missing model files, broken install, ...
    st.error(f"Could not start FairForge: {startup_error}")
    st.stop()

# ---- hero banner, ribbon and the standing disclaimer ----
components.html(HERO_HTML, height=HERO_HEIGHT)
render_html(build_marquee())
render_html(f"<div class='ff-notice'><b>Please note.</b> {escape(DISCLAIMER)}</div>")

# ---- the Forge console: step 1 (job) and step 2 (resumes) as two glass panels ----
render_html(build_section_title("01", "The forge console", "Describe the job, drop in the resumes, tune the recipe, then ignite."))
# Left panel: the job description. Right panel: the resume upload.
job_panel, resume_panel = st.columns(2, gap="large")
with job_panel:
    render_html(build_panel_heading("ff-accent-ember", "STEP 1", "Describe the job", "Paste the job description you are hiring for."))
    # Job description box, pre-filled with the demo text.
    job_description = st.text_area("Job description", DEMO_JD, height=230)
with resume_panel:
    render_html(build_panel_heading("ff-accent-teal", "STEP 2", "Drop the resumes", "PDF or TXT. Names are never shown while you review."))
    # Accepts several PDF or TXT resumes at once.
    uploaded_files = st.file_uploader("Resumes (PDF or TXT, several allowed)", type=["pdf", "txt"], accept_multiple_files=True)
    # Ticked by default when nothing has been uploaded.
    use_demo_resumes = st.checkbox("Use the 3 demo resumes", value=not uploaded_files)

# ---- step 3: four colour-coded faders that replace the old sidebar sliders (same labels, same defaults) ----
render_html(build_section_title("02", "Mix the score", "Move a fader and the recipe bar changes. The recipe is fully visible, nothing is hidden."))
# One slider per signal. The values become the weights used in the final score.
fader_columns = st.columns(4, gap="medium")
score_weights = {}
for fader_column, signal_name in zip(fader_columns, PART_NAMES):
    signal_style = SIGNAL_STYLES[signal_name]
    with fader_column:
        render_html(build_panel_heading(signal_style["accent_class"], "FADER", signal_name, signal_style["tagline"]))
        score_weights[signal_name] = st.slider(signal_style["slider_label"], 0, 100, signal_style["default_weight"])
if sum(score_weights.values()) == 0:
    score_weights["Skill keywords"] = 1                              # avoid dividing by zero when every fader is at 0
render_html(build_recipe_bar(score_weights))

# ---- launch row: model status LEDs on the left, Twin Test switch and the Ignite button on the right ----
status_column, launch_column = st.columns([3, 2], gap="large")
with status_column:
    render_html(build_status_leds(engine.status))
    if not engine.status["Name detection (spaCy)"]:
        st.warning("spaCy model missing: run  python -m spacy download en_core_web_sm  (names are hidden less reliably until then).")
with launch_column:
    # The Twin Test scores every resume extra times, so it can be switched off to save time.
    run_twin_test = st.toggle("Run the Twin Test (slower)", value=True)
    # Main button. Nothing is scored until it is pressed.
    ignite_clicked = st.button("Ignite the forge  →", type="primary")

# ---- run the ranking (same flow as before: read files, analyse each resume, remember results) ----
# Runs only when the button is pressed: read every resume, analyse it with core.py, and keep the results.
if ignite_clicked:
    # Collect the resume texts (uploaded files first, then the demo resumes if requested).
    resume_sources = {}
    for uploaded_file in uploaded_files or []:
        resume_sources[uploaded_file.name] = extract_text(uploaded_file)
    if use_demo_resumes or not resume_sources:
        resume_sources.update(DEMO_RESUMES)
    # Analyse the resumes one by one. A resume that fails is skipped with a warning, so one bad file does not stop the
    # others.
    analysed_results, progress_bar = [], st.progress(0.0, text="Reading resumes...")
    for position, (file_name, resume_text) in enumerate(resume_sources.items()):
        try:
            if len(resume_text.split()) < 30:
                raise ValueError("almost no text found (a scanned PDF needs OCR first)")
            analysed_results.append({"label": f"Candidate {position + 1}", "file": file_name,
                                     "a": engine.analyze(resume_text, job_description, score_weights, run_twin_test)})
        except Exception as analysis_error:
            st.warning(f"Skipped {file_name}: {analysis_error}")
        progress_bar.progress((position + 1) / len(resume_sources), text=f"Analysed {position + 1} of {len(resume_sources)}")
    progress_bar.empty()
    # Save the results in session_state so they survive Streamlit re-runs (for example when a tab or dropdown is
    # changed).
    st.session_state["results"], st.session_state["weights"] = analysed_results, dict(score_weights)

# Results from the last ranking run (empty before the first run).
all_results = st.session_state.get("results")

# ---- start page (before any ranking): the pipeline and the proof numbers from training ----
if not all_results:
    render_html(build_section_title("03", "How FairForge thinks", "Five stations. Identity is gone before the first model ever sees the resume."))
    render_html(build_pipeline())
    training_metrics = engine.metrics or {}
    # Numbers measured during training, read from models/metrics.json.
    fairness_metrics, adversary_metrics = training_metrics.get("fairness"), training_metrics.get("adversary_accuracy")
    if fairness_metrics:
        render_html(build_section_title("04", "Proof, not promises", "Measured on synthetic identities. Crossed-out = identity visible, glowing = identity hidden."))
        proof_tiles = [build_proof_tile("Gender disparate impact", f"{fairness_metrics['gender_raw']['DI']:.2f}", f"{fairness_metrics['gender_anonymized']['DI']:.2f}", "0.80 or more passes the four-fifths rule", TEAL),
                       build_proof_tile("Gender decisions that flip", f"{100 * fairness_metrics['gender_raw']['flip']:.1f}%", f"{100 * fairness_metrics['gender_anonymized']['flip']:.1f}%", "same resume, two identities", EMBER)]
        if adversary_metrics:
            proof_tiles.append(build_proof_tile("Adversary guessing gender", f"{100 * adversary_metrics['gender_raw']:.0f}%", f"{100 * adversary_metrics['gender_anonymized']:.0f}%", "50% = no better than a coin", GOLD))
        render_html(f"<div class='ff-tiles'>{''.join(proof_tiles)}</div>")
    st.stop()

# ---- ranks before and after hiding identity (same maths as before) ----
analysis_weights = st.session_state["weights"]
# The real ranking: candidates sorted by their identity-hidden score.
for candidate, rank in zip(sorted(all_results, key=lambda item: -item["a"]["hidden"]["final"]), range(1, len(all_results) + 1)):
    candidate["rank"] = rank
# The ranking that identity-visible scores would have given. It is only used to show how ranks move.
for candidate, rank in zip(sorted(all_results, key=lambda item: -item["a"]["visible_score"]), range(1, len(all_results) + 1)):
    candidate["rank_before"] = rank
# Candidates in final rank order, used by every tab below.
ordered_results = sorted(all_results, key=lambda item: item["rank"])

render_html(build_section_title("03", "The verdict", "Switch views with the dock at the bottom of the screen."))
render_html("<div class='ff-dock-hint'><i>↓</i> USE THE DOCK AT THE BOTTOM TO SWITCH VIEWS</div>")

# ================================================================================================================
# 8. RESULT TABS (shown as a floating dock at the bottom of the screen)
# ================================================================================================================
# Five result tabs. The CSS turns the tab bar into the floating dock at the bottom of the screen.
tab_ranking, tab_why, tab_hidden, tab_fairness, tab_about = st.tabs(["🏆 Ranking", "🔬 Why this rank", "🫥 Hidden identity", "⚖️ Fairness audit", "ℹ️ About"])

# ---------------------------------------------------------------------------------------------- tab 1: ranking
# Tab 1 content: podium, leaderboard, comparison charts, score table and the 3D job-fit chart.
with tab_ranking:
    render_html(build_podium(ordered_results))
    render_html(build_leaderboard(ordered_results))
    chart_left, chart_right = st.columns(2)
    chart_left.plotly_chart(build_radar_chart(ordered_results), use_container_width=True)
    chart_right.plotly_chart(build_slope_chart(ordered_results), use_container_width=True)
    # One table row per candidate: rank, total score, the four signal scores and the rank change.
    score_table_rows = []
    for candidate in ordered_results:
        part_scores = candidate["a"]["hidden"]["parts"]
        score_table_rows.append([candidate["rank"], candidate["label"], round(100 * candidate["a"]["hidden"]["final"], 1),
                                 *[round(100 * part_scores[name], 1) for name in PART_NAMES], candidate["rank_before"], candidate["rank_before"] - candidate["rank"]])
    score_table_columns = ["Rank", "Candidate", "Score %", *PART_NAMES, "Rank before hiding identity", "Rank change"]
    render_html(build_table(score_table_columns, score_table_rows, signed_columns=("Rank change",)))
    st.caption("Rank change = rank before hiding identity minus rank after. Positive means the candidate moved up once identity was hidden.")
    # 3D chart is skipped when MiniLM vectors are not available.
    galaxy_figure = build_galaxy_chart(engine, ordered_results)
    if galaxy_figure is not None:
        st.plotly_chart(galaxy_figure, use_container_width=True)
        st.caption("Each blue dot is a job family. The closer a candidate sits to the job description, the closer the meaning of the resume.")
    with st.expander("Key: which file is which candidate (shown after ranking, to keep the review blind)"):
        render_html(build_table(["Candidate", "File"], [[candidate["label"], candidate["file"]] for candidate in all_results]))

# ---------------------------------------------------------------------------------------------- tab 2: why this rank
# Tab 2 content: score breakdown, SHAP word chart, skill gap and resume tips for the chosen candidate.
with tab_why:
    chosen_label = st.selectbox("Candidate", [candidate["label"] for candidate in ordered_results], key="why")
    chosen = next(candidate for candidate in all_results if candidate["label"] == chosen_label)
    hidden_result = chosen["a"]["hidden"]
    # Big number tiles: the final score and each of the four signals.
    tiles = [build_stat_tile("Final score", f"{100 * hidden_result['final']:.1f}", "the weighted mix of the four signals", TEXT_BRIGHT, "%")]
    tiles += [build_stat_tile(name, f"{100 * hidden_result['parts'][name]:.1f}", SIGNAL_STYLES[name]["tagline"], SIGNAL_STYLES[name]["color"], "%") for name in PART_NAMES]
    render_html(f"<div class='ff-tiles'>{''.join(tiles)}</div>")
    # Bars for the job families the job description resembles most.
    family_bars = "".join(f"<div class='ff-bar-row'><div>{escape(name)}</div><div class='ff-bar-track' style='--w:{100 * probability:.1f}%; --chance:-10%'><div class='ff-bar-fill'></div></div>"
                          f"<div class='ff-bar-value'>{probability:.0%}</div></div>" for name, probability in hidden_result["top_family"])
    render_html(f"<div class='ff-group-title'>The job description looks like these job families</div>{family_bars}")
    st.caption(f"Engine: {hidden_result['engine']}  |  job-history lines read by the BiLSTM: {hidden_result['experience_lines']}")
    # SHAP chart. It is skipped when the explanation files are missing.
    explanation = chosen["a"]["explain"]
    if explanation and (explanation["up"] or explanation["down"]):
        st.plotly_chart(build_waterfall_chart(explanation), use_container_width=True)
        st.caption("Teal words raised the match, orange words lowered it. SHAP shows how much each word changed the model's output.")
    else:
        st.info("SHAP explanation not available for this resume.")
    render_html(build_section_title("", "Skill gap", "What the resume has, and what the job asks for that it does not show."))
    # Green chips are job keywords found in the resume, orange chips are the ones missing.
    matched_chips = "".join(f"<span class='ff-chip ff-yes'>{escape(word)}</span>" for word in hidden_result["matched"]) or "<span class='ff-note'>no matches</span>"
    missing_chips = "".join(f"<span class='ff-chip ff-no'>{escape(word)}</span>" for word in hidden_result["missing"]) or "<span class='ff-note'>nothing missing</span>"
    render_html(f"<div class='ff-group-title'>Found in the resume</div>{matched_chips}<div class='ff-group-title'>Missing from the resume</div>{missing_chips}")
    render_html(build_section_title("", "Resume tips", "Only skills, clarity and structure. Never identity."))
    render_html("".join(f"<div class='ff-tip'><span class='ff-tip-no'>{number}</span><span>{escape(tip)}</span></div>" for number, tip in enumerate(chosen["a"]["tips"], start=1)))
    st.caption("Tips only cover skills, clarity and structure. They never suggest changing identity details or inventing skills.")

# ---------------------------------------------------------------------------------------------- tab 3: hidden identity
# Tab 3 content: original resume next to the identity-hidden version, plus charts of what was hidden.
with tab_hidden:
    chosen_label = st.selectbox("Candidate", [candidate["label"] for candidate in ordered_results], key="hid")
    chosen = next(candidate for candidate in all_results if candidate["label"] == chosen_label)
    hidden_result = chosen["a"]["hidden"]
    # Left: the original text. Right: the hidden version that the models actually read.
    original_column, redacted_column = st.columns(2, gap="large")
    original_column.markdown(build_raw_paper(chosen["a"]["raw_text"]).replace("\n", " "), unsafe_allow_html=True)
    redacted_column.markdown(build_clean_paper(hidden_result["anon"], hidden_result["sections"]).replace("\n", " "), unsafe_allow_html=True)
    st.caption("Right side: black bars are identity details that were replaced. The coloured label on each line is the section the CNN found.")
    detail_left, detail_right = st.columns(2)
    # Flow chart of the identity details that were found and hidden.
    sankey_figure = build_sankey_chart(hidden_result["counts"])
    if sankey_figure is not None:
        detail_left.plotly_chart(sankey_figure, use_container_width=True)
    else:
        detail_left.info("No identity details were found to hide.")
    detail_right.plotly_chart(build_section_sunburst(hidden_result["sections"]), use_container_width=True)

# ---------------------------------------------------------------------------------------------- tab 4: fairness audit
# Tab 4 content: the Twin Test on the uploaded resumes, and the bias audit measured during training.
with tab_fairness:
    render_html(build_section_title("", "Twin Test on your candidates", "Same resume, two made-up identities. If identity is truly hidden, the scale stays level."))
    render_html("<div class='ff-twin-legend'>"
                f"<div class='ff-twin-id'><b style='color:{EMBER}'>Twin A</b><span>{escape(TWIN_A[0])} · {escape(TWIN_A[1])} · graduated {TWIN_A[2]}</span></div>"
                f"<div class='ff-twin-id'><b style='color:{TEAL}'>Twin B</b><span>{escape(TWIN_B[0])} · {escape(TWIN_B[1])} · graduated {TWIN_B[2]}</span></div></div>")
    # Only candidates that went through the Twin Test (the toggle can be off).
    twin_results = [(candidate["label"], candidate["a"]["twin"]) for candidate in ordered_results if candidate["a"]["twin"]]
    if twin_results:
        render_html("".join(build_twin_card(label, twin) for label, twin in twin_results))
        st.caption("The tilt is exaggerated so you can see it: every score point of gap turns the beam 4 degrees (maximum 22).")
        # Two rows per candidate (visible and hidden). Each row joins the Twin A score and the Twin B score.
        dumbbell_rows = []
        for label, twin in twin_results:
            dumbbell_rows.append((f"{label}, identity visible", 100 * twin["identity visible"]["twin A"], 100 * twin["identity visible"]["twin B"]))
            dumbbell_rows.append((f"{label}, identity hidden", 100 * twin["identity hidden"]["twin A"], 100 * twin["identity hidden"]["twin B"]))
        # Dumbbell chart: the closer the two dots, the smaller the score gap between the twins.
        twin_figure = go.Figure()
        for row_label, score_a, score_b in dumbbell_rows:
            twin_figure.add_trace(go.Scatter(x=[score_a, score_b], y=[row_label, row_label], mode="lines+markers", line=dict(color="rgba(255,255,255,.35)", width=5),
                                             marker=dict(size=[16, 16], color=[EMBER, TEAL], line=dict(color="#fff", width=2)), showlegend=False, hovertemplate="%{x:.2f}<extra></extra>"))
        twin_figure.update_xaxes(title="Score % (orange = twin A, teal = twin B)")
        st.plotly_chart(style_chart(twin_figure, 120 + 46 * len(dumbbell_rows), "Twin scores: the closer the dots, the fairer"), use_container_width=True)
        # Summary table of the score gaps and the PASS or CHECK result.
        gap_table_rows = [[label, round(100 * twin["identity hidden"]["gap"], 2), round(100 * twin["identity visible"]["gap"], 2), "PASS" if twin["pass"] else "CHECK"] for label, twin in twin_results]
        render_html(build_table(["Candidate", "Score gap, identity hidden", "Score gap, identity visible", "Result"], gap_table_rows))
        st.caption("Gaps are in score points. PASS means the gap with identity hidden is under 0.5 points.")
    else:
        st.info("Turn on 'Run the Twin Test' in the console and rank again.")
    training_metrics = engine.metrics
    # Training-time audit: disparate impact, adversary accuracy, anonymizer recall and model accuracy.
    if training_metrics and training_metrics.get("fairness"):
        render_html(build_section_title("", "Bias audit from training", "All identities and the biased hiring history are synthetic (made up for the test)."))
        fairness_metrics = training_metrics["fairness"]
        st.plotly_chart(build_dumbbell_chart([("Gender", fairness_metrics["gender_raw"]["DI"], fairness_metrics["gender_anonymized"]["DI"]),
                                              ("Age", fairness_metrics["age_raw"]["DI"], fairness_metrics["age_anonymized"]["DI"])],
                                             "Disparate Impact: identity visible vs hidden", threshold=0.8, x_range=[0.5, 1.05]), use_container_width=True)
        adversary_metrics = training_metrics.get("adversary_accuracy", {})
        if adversary_metrics:
            st.plotly_chart(build_dumbbell_chart([("Guess gender", adversary_metrics["gender_raw"], adversary_metrics["gender_anonymized"]),
                                                  ("Guess age", adversary_metrics["age_raw"], adversary_metrics["age_anonymized"])],
                                                 "Can an adversary model still guess? (0.50 = no better than a coin)", threshold=None, x_range=[0.4, 0.8]), use_container_width=True)
        recall_metrics = training_metrics.get("anonymizer_recall", {})
        if recall_metrics:
            render_html("<div class='ff-group-title'>Anonymizer recall on injected identities</div>"
                        + "".join(f"<span class='ff-chip ff-yes'>{escape(name.replace('_', ' '))} {value:.0%}</span>" for name, value in recall_metrics.items()))
        render_html("<div class='ff-group-title'>Job-category accuracy of the models (24 categories, teal tick = chance, 4%)</div>"
                    + build_accuracy_bars({"TF-IDF + Logistic Regression": training_metrics.get("tfidf_logreg_acc"),
                                           "MiniLM + Logistic Regression": training_metrics.get("minilm_logreg_acc"),
                                           "DNN": training_metrics.get("dnn_ablation") or {}}))

# ---------------------------------------------------------------------------------------------- tab 5: about
# Tab 5 content: pipeline overview and the known limits of the system.
with tab_about:
    render_html(build_section_title("", "Pipeline", "From raw resume to an explained, fairness-tested score."))
    render_html(build_pipeline())
    render_html("<div class='ff-tip'><span class='ff-tip-no'>!</span><span><b>Limits.</b> The models were trained on about 2,500 example resumes in 24 job families, so job-fit is a rough signal "
                "and is mixed with keyword and meaning matches. Name detection is not perfect, and only the first lines are checked for unknown names. Scanned PDFs need OCR first.</span></div>")
    st.warning(DISCLAIMER)
