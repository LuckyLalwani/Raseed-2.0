# app_streamlit.py
import os
import json
import tempfile
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any

import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
from dateutil import parser

# UI & visuals
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go

# Local LLM
from langchain_community.chat_models import ChatOllama

import base64

def load_image_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_data = load_image_base64("Asset 1.png")
# ---------------------- CONFIG ----------------------
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # <- update if different
OLLAMA_MODEL = "llama3.2"  # change if needed

STORE_DIR = "data"
STORE_PATH = os.path.join(STORE_DIR, "receipts_store.jsonl")
BUDGET_PATH = os.path.join(STORE_DIR, "budget.json")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ---------------------- HELPERS ----------------------
def build_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

LOTTIE_PARSE = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_j1adxtyb.json")  # change if you want

from pdf2image import convert_from_path

POPPLER_PATH = r"C:\Users\Lenovo\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"   # CHANGE THIS to your Poppler bin path

def pdf_to_images(pdf_path: str) -> list:
    """Convert PDF pages into a list of image paths."""
    images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    img_paths = []
    for i, img in enumerate(images):
        temp_path = f"{pdf_path}_page_{i}.png"
        img.save(temp_path, "PNG")
        img_paths.append(temp_path)
    return img_paths


# ---------------------- OCR & LLM PARSING ----------------------
def ocr_image_to_text(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="eng")

def llm_parse_receipt(raw_text: str) -> dict:
    """
    LLM parse function — returns structured JSON for a receipt.
    """
    llm = build_llm()
    prompt = f"""
You are an expert at extracting structured data from store receipts.

Return ONLY valid JSON with this structure:

{{
  "vendor": string or null,
  "date": string or null,
  "items": [
    {{
      "name": string,
      "quantity": number or null,
      "unit_price": number or null,
      "total_price": number or null,
      "category": string or null
    }}
  ],
  "total": number or null,
  "main_category": string or null
}}

Rules:
- Use line-items to detect quantity/unit_price/line_total. If quantity missing, assume 1.
- Ignore taxes and round-off lines.
- Infer category from item name; if unsure set "Other".
- Return numeric values (no currency symbols in JSON).
- Answer only as JSON (no extra text).
Receipt text:
\"\"\"{raw_text}\"\"\"
"""
    response = llm.invoke(prompt)
    content = response.content
    try:
        data = json.loads(content)
    except Exception:
        # try to extract JSON block
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data = json.loads(content[start:end])
        except Exception:
            # fallback minimal structure
            st.error("LLM returned non-JSON output. See raw output below for debugging.")
            st.text(content)
            raise

    # Normalize items + fields
    items = data.get("items") if isinstance(data.get("items"), list) else []
    norm_items = []
    for it in items:
        name = (it.get("name") or "").strip()
        try:
            q = float(it.get("quantity")) if it.get("quantity") is not None else 1.0
        except Exception:
            q = 1.0
        try:
            up = float(it.get("unit_price")) if it.get("unit_price") is not None else None
        except Exception:
            up = None
        try:
            tp = float(it.get("total_price")) if it.get("total_price") is not None else None
        except Exception:
            tp = None
        if tp is None and up is not None:
            try:
                tp = up * q
            except Exception:
                tp = None
        cat = it.get("category") or "Other"
        norm_items.append({"name": name, "quantity": q, "unit_price": up, "total_price": tp, "category": cat})

    data["items"] = norm_items
    try:
        data["total"] = float(data.get("total")) if data.get("total") is not None else None
    except Exception:
        data["total"] = None
    if "main_category" not in data or not data["main_category"]:
        cats = [it.get("category", "Other") for it in norm_items]
        data["main_category"] = max(set(cats), key=cats.count) if cats else None
    return data

# ---------------------- STORAGE ----------------------
def save_receipt(structured_data: dict):
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(STORE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(structured_data, ensure_ascii=False) + "\n")

def load_receipts(store_path: str = STORE_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(store_path):
        return []
    receipts = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                receipts.append(json.loads(line))
            except Exception:
                continue
    return receipts

# ---------------------- STATS & HELPERS ----------------------
def compute_basic_stats(receipts: List[Dict[str, Any]]) -> Dict[str, Any]:
    grand_total = 0.0
    by_category = {}
    by_vendor = {}
    timeline = []
    for r in receipts:
        try:
            total = float(r.get("total") or 0.0)
        except Exception:
            total = 0.0
        grand_total += total
        cat = r.get("main_category") or r.get("category") or "Unknown"
        by_category[cat] = by_category.get(cat, 0.0) + total
        vendor = (r.get("vendor") or "Unknown").strip()
        by_vendor[vendor] = by_vendor.get(vendor, 0.0) + total
        date_str = r.get("date")
        try:
            dt = parser.parse(date_str, dayfirst=True) if date_str else None
        except Exception:
            dt = None
        timeline.append({"date": dt, "total": total})
    timeline = sorted([t for t in timeline if t["date"] is not None], key=lambda x: x["date"])
    return {"grand_total": grand_total, "by_category": by_category, "by_vendor": by_vendor, "timeline": timeline}

def receipts_to_dataframe(receipts: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in receipts:
        vendor = r.get("vendor")
        date = r.get("date")
        total = r.get("total") or 0
        main_cat = r.get("main_category", r.get("category", "Unknown"))
        item_strs = []
        for it in (r.get("items") or []):
            nm = it.get("name")
            q = it.get("quantity") or 1
            up = it.get("unit_price")
            tp = it.get("total_price")
            part = f"{nm} x{int(q) if float(q).is_integer() else q}"
            if up is not None:
                part += f" @ {up}"
            if tp is not None:
                part += f" = {tp}"
            item_strs.append(part)
        items = "; ".join(item_strs)
        rows.append({"vendor": vendor, "date": date, "total": total, "main_category": main_cat, "items": items})
    return pd.DataFrame(rows)

def detect_outliers(receipts: List[Dict[str, Any]], z_thresh: float = 2.5):
    totals = [float(r.get("total") or 0) for r in receipts]
    if len(totals) < 2:
        return []
    mean = statistics.mean(totals)
    stdev = statistics.stdev(totals)
    if stdev == 0:
        return []
    outliers = []
    for i, t in enumerate(totals):
        z = (t - mean) / stdev
        if abs(z) >= z_thresh:
            outliers.append({"index": i, "receipt": receipts[i], "z_score": z})
    return outliers

def detect_recurring(receipts: List[Dict[str, Any]], months_window: int = 6, tolerance: float = 0.10):
    now = datetime.now()
    vendor_map = defaultdict(list)
    for r in receipts:
        vendor = (r.get("vendor") or "unknown").strip().lower()
        total = float(r.get("total") or 0)
        date_str = r.get("date")
        dt = None
        if date_str:
            try:
                dt = parser.parse(date_str, dayfirst=True)
            except Exception:
                dt = None
        vendor_map[vendor].append({"total": total, "date": dt})
    recurring = []
    for v, entries in vendor_map.items():
        recent = [e for e in entries if e["date"] and (now - e["date"]).days <= months_window * 30]
        if len(recent) >= 2:
            totals = [e["total"] for e in recent]
            avg = sum(totals) / len(totals) if totals else 0
            if avg == 0:
                continue
            if all(abs(t - avg) / avg <= tolerance for t in totals):
                recurring.append({"vendor": v, "count": len(recent), "avg_amount": avg})
    return recurring

# ---------------------- BUDGETS ----------------------
def load_budgets() -> Dict[str, float]:
    if not os.path.exists(BUDGET_PATH):
        return {}
    try:
        with open(BUDGET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_budgets(budgets: Dict[str, float]):
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(BUDGET_PATH, "w", encoding="utf-8") as f:
        json.dump(budgets, f, ensure_ascii=False, indent=2)

# ---------------------- INSIGHTS PROMPT ----------------------
def build_insights_prompt(receipts: List[Dict[str, Any]], stats: Dict[str, Any], question: str) -> str:
    compact = []
    for r in receipts[-200:]:
        compact.append({
            "vendor": r.get("vendor"),
            "date": r.get("date"),
            "items": r.get("items"),
            "total": r.get("total"),
            "main_category": r.get("main_category", r.get("category"))
        })
    receipts_json = json.dumps(compact, ensure_ascii=False, indent=2, default=str)
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2, default=str)

    return f"""
You are an AI financial insights assistant. Use the provided RECEIPTS_JSON and STATS_JSON for facts.

RECEIPTS_JSON:
{receipts_json}

STATS_JSON:
{stats_json}

User question:
\"\"\"{question}\"\"\"

Instructions:
- ALWAYS use Indian Rupees (₹) for all currency values. Never use $ or other currency symbols.
- When giving amounts, prefix with ₹, like: ₹120, ₹948.50
- Use STATS_JSON for all numeric facts. Do not invent numbers.
- Provide short, actionable insights (3-6 bullet points) and one suggestion to improve spending.
- If the data is insufficient, say so.
"""

# ---------------------- UI ----------------------
st.set_page_config(page_title="Raseed 2.0", page_icon="🧾", layout="wide")
st.markdown("<style>body{background-color:#0b1220;color:#e5e7eb}</style>", unsafe_allow_html=True)

# Top-level Hero
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:10px;">
        <div style="flex:0 0 auto;">
            <img src="data:image/png;base64,{logo_data}" style="height:64px;border-radius:8px;" />
        </div>
        <div style="flex:1 1 auto;">
            <h1 style="margin:0;color:#fff;">🧾 Raseed 2.0</h1>
            <div style="color:#9CA3AF;">AI Expense Agent — receipts → structured data → insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True
)

tabs = st.tabs(["➕ Add Receipt", "📊 Dashboard", "🔎 Search", "🏷 Budgets", "🤖 Insights"])

# ---------------------- TAB: Add Receipt ----------------------
with tabs[0]:
    st.header("Upload & Process Receipt")
    uploaded_file = st.file_uploader("Upload receipt (image or PDF)", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded_file:
        suffix = uploaded_file.name.split(".")[-1].lower()

        if suffix == "pdf":
            # Save temporary PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            # Convert PDF → images
            pages = pdf_to_images(pdf_path)
            st.success(f"PDF uploaded — {len(pages)} page(s) detected.")

            # Process EACH PAGE like an image
            all_results = []
            for page_path in pages:
                st.image(page_path, caption=f"PDF Page", width=380)
                raw = ocr_image_to_text(page_path)
                parsed = llm_parse_receipt(raw)
                parsed["raw_text"] = raw
                all_results.append(parsed)

            # If PDF contains multiple pages, merge totals
            if len(all_results) == 1:
                final = all_results[0]
            else:
                final = {
                    "vendor": all_results[0].get("vendor"),
                    "date": all_results[0].get("date"),
                    "items": sum([r.get("items", []) for r in all_results], []),
                    "total": sum([(r.get("total") or 0) for r in all_results]),
                    "main_category": all_results[0].get("main_category"),
                    "raw_text": "\n\n--- PAGE BREAK ---\n\n".join([r["raw_text"] for r in all_results])
                }

            save_receipt(final)
            st.success("PDF receipt saved!")
            st.json(final)

        else:
            # EXISTING IMAGE LOGIC
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.image(tmp_path, caption="Preview", width=380)
            raw = ocr_image_to_text(tmp_path)
            parsed = llm_parse_receipt(raw)
            parsed["raw_text"] = raw
            save_receipt(parsed)
            st.success("Image receipt saved!")
            st.json(parsed)


# ---------------------- TAB: Dashboard ----------------------
with tabs[1]:
    st.header("Dashboard")
    receipts = load_receipts()
    if not receipts:
        st.info("No receipts yet. Upload from the Add Receipt tab.")
    else:
        stats = compute_basic_stats(receipts)

        # Metric cards row (hero-like)
        col1, col2, col3, col4 = st.columns([1,1,1.6,1.6])
        col1.metric("Total receipts", len(receipts))
        col2.metric("Grand total", f"₹{stats['grand_total']:.2f}")
        top_cat = max(stats["by_category"].items(), key=lambda x: x[1]) if stats["by_category"] else ("-", 0)
        col3.metric("Top category", f"{top_cat[0]} — ₹{top_cat[1]:.2f}")
        top_vendor = max(stats["by_vendor"].items(), key=lambda x: x[1]) if stats["by_vendor"] else ("-", 0)
        col4.metric("Top vendor", f"{top_vendor[0]} — ₹{top_vendor[1]:.2f}")

        st.markdown("### Spending by Category")
        by_cat = stats["by_category"]
        if by_cat:
            labels = list(by_cat.keys())
            values = [by_cat[k] for k in labels]
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#FFC107", "#795548"][:len(labels)]
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors),
                sort=False,
                hovertemplate="%{label}: ₹%{value:.2f} (%{percent})"
            )])

            # set size inside the figure layout (no deprecated streamlit kwargs)
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="v", x=1.02, y=0.5, font=dict(color="#e5e7eb")),
                width=420,
                height=300,
            )

            # use config dict only (no width/use_container_width kwargs)
            st.plotly_chart(fig, config={"displaylogo": False, "responsive": True})

        else:
            st.info("No category data.")

        st.markdown("### Spending timeline (daily)")
        timeline = stats["timeline"]
        if timeline:
            df_t = pd.DataFrame([{"date": t["date"].date(), "total": t["total"]} for t in timeline])
            df_daily = df_t.groupby("date").sum().reset_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_daily["date"], y=df_daily["total"], mode="lines+markers", line=dict(width=2)))

            # set layout and explicit height here
            fig2.update_layout(
                margin=dict(t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickangle=-35),
                height=240
            )

            st.plotly_chart(fig2, config={"displaylogo": False, "responsive": True})

        else:
            st.info("No timeline data.")

        st.markdown("### Recent receipts (cards)")
        recent = receipts[-9:][::-1]
        cols = st.columns(3)
        for i, r in enumerate(recent):
            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style='background:#071023;padding:10px;border-radius:10px;'>
                        <div style='font-weight:700;color:#fff'>{r.get('vendor','Unknown')}</div>
                        <div style='color:#9CA3AF;font-size:12px;margin-bottom:6px;'>{r.get('date','')}</div>
                        <div style='font-weight:700;color:#10B981'>₹{float(r.get('total') or 0):.2f}</div>
                        <div style='margin-top:8px;color:#D1D5DB;font-size:12px'>{", ".join([it.get("name") for it in r.get("items",[])][:2])}</div>
                        <div style="margin-top:8px;"><a href="#" style="color:#60A5FA;text-decoration:none;">View</a></div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.download_button("📥 Export receipts CSV", data=receipts_to_dataframe(receipts).to_csv(index=False).encode("utf-8"), file_name="raseed_receipts.csv", mime="text/csv")

# ---------------------- TAB: Search ----------------------
with tabs[2]:
    st.header("Search & Filter")
    receipts = load_receipts()
    if not receipts:
        st.info("No receipts available.")
    else:
        df = receipts_to_dataframe(receipts)
        q = st.text_input("Search vendor/item keyword:")
        cats = sorted(list({r.get("main_category") or r.get("category") or "Unknown" for r in receipts}))
        cat_sel = st.multiselect("Category", options=cats, default=[])
        vendor_list = sorted(list({r.get("vendor") or "Unknown" for r in receipts}))
        vendor_sel = st.multiselect("Vendor", options=vendor_list, default=[])
        col1, col2 = st.columns(2)
        with col1:
            min_amt = st.number_input("Min amount (₹)", min_value=0.0, value=0.0)
        with col2:
            max_amt = st.number_input("Max amount (₹)", min_value=0.0, value=1000000.0)
        start_date = st.date_input("Start date", value=(datetime.now() - timedelta(days=365)).date())
        end_date = st.date_input("End date", value=datetime.now().date())

        filtered = []
        for r in receipts:
            amt = float(r.get("total") or 0)
            if amt < min_amt or amt > max_amt:
                continue
            ds = r.get("date")
            try:
                dt = parser.parse(ds, dayfirst=True).date() if ds else None
            except Exception:
                dt = None
            if dt and (dt < start_date or dt > end_date):
                continue
            mcat = r.get("main_category") or r.get("category") or "Unknown"
            if cat_sel and mcat not in cat_sel:
                continue
            vendor = r.get("vendor") or "Unknown"
            if vendor_sel and vendor not in vendor_sel:
                continue
            if q:
                blob = " ".join([str(r.get("vendor") or ""), str(r.get("items") or ""), str(r.get("raw_text") or "")]).lower()
                if q.lower() not in blob:
                    continue
            filtered.append(r)

        st.write(f"Showing {len(filtered)} receipt(s).")
        if filtered:
            st.dataframe(receipts_to_dataframe(filtered).sort_values("date", ascending=False), height=320)
            st.download_button("Export filtered CSV", data=receipts_to_dataframe(filtered).to_csv(index=False).encode("utf-8"), file_name="filtered_receipts.csv", mime="text/csv")
        else:
            st.info("No matching receipts.")

# ---------------------- TAB: Budgets ----------------------
with tabs[3]:
    st.header("Budgets & Alerts")
    receipts = load_receipts()
    stats = compute_basic_stats(receipts)
    budgets = load_budgets()
    st.subheader("Set a budget per category (monthly)")
    cat_new = st.text_input("Category name")
    amt_new = st.number_input("Amount (₹)", min_value=0.0, value=0.0)
    if st.button("Save Budget"):
        if not cat_new.strip():
            st.warning("Enter category name")
        else:
            budgets[cat_new.strip()] = float(amt_new)
            save_budgets(budgets)
            st.success("Budget saved.")

    st.markdown("### Current budgets")
    if budgets:
        for cat, val in budgets.items():
            st.write(f"- **{cat}** : ₹{val:.2f}")
    else:
        st.info("No budgets set.")

    # monthly totals
    now = datetime.now()
    month_start = datetime(now.year, now.month, 1)
    monthly_totals = defaultdict(float)
    for r in receipts:
        date_str = r.get("date")
        try:
            dt = parser.parse(date_str, dayfirst=True) if date_str else None
        except Exception:
            dt = None
        if dt and dt >= month_start:
            cat = r.get("main_category") or r.get("category") or "Unknown"
            monthly_totals[cat] += float(r.get("total") or 0)
    st.markdown("### Progress this month")
    for cat, limit in budgets.items():
        used = monthly_totals.get(cat, 0.0)
        perc = min(100.0, (used / limit * 100.0) if limit > 0 else 0.0)
        st.write(f"**{cat}** — ₹{used:.2f} / ₹{limit:.2f}")
        st.progress(int(perc))
        if perc >= 100:
            st.error(f"Budget exceeded for {cat}!")
        elif perc >= 80:
            st.warning(f"Approaching budget for {cat} ({perc:.0f}% used).")

# ---------------------- TAB: Insights ----------------------
with tabs[4]:
    st.header("AI Insights")
    receipts = load_receipts()
    if not receipts:
        st.info("No receipts to analyze.")
    else:
        stats = compute_basic_stats(receipts)
        question = st.text_input("Ask something (default: 3 short insights)", value="What are 3 short insights about my spending?")
        if st.button("Get Insights"):
            with st.spinner("Generating insights..."):
                prompt = build_insights_prompt(receipts, stats, question)
                llm = build_llm()
                response = llm.invoke(prompt)
                st.subheader("Insights")
                st.markdown(response.content.replace("\n", "  \n"))

        # quick auto insights
        st.markdown("---")
        st.subheader("Quick summary")
        st.write(f"- Grand total: **₹{stats['grand_total']:.2f}**")
        top_cat = max(stats["by_category"].items(), key=lambda x: x[1]) if stats["by_category"] else ("-", 0)
        st.write(f"- Top category: **{top_cat[0]}** — ₹{top_cat[1]:.2f}")
        top_vendor = max(stats["by_vendor"].items(), key=lambda x: x[1]) if stats["by_vendor"] else ("-", 0)
        st.write(f"- Top vendor: **{top_vendor[0]}** — ₹{top_vendor[1]:.2f}")

# ---------------------- END ----------------------
