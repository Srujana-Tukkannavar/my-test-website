# app.py
# Streamlit + IBM Granite (via Hugging Face Inference API)
# Personal Finance Assistant matching the flow in your screenshots:
# Home ➜ (NLU Analysis | Q&A | Budget Summary | Spending Insights)

import os
import re
import json
import textwrap
from typing import Dict, Any, List

import requests
import streamlit as st

# ---------------------------
# Page / Theme Setup
# ---------------------------
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
)

# Background + frosted glass look (simple CSS)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    .glass {
        background: rgba(255,255,255,0.72);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: 1px solid rgba(255,255,255,0.35);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    .hero {
        background: rgba(255,255,255,0.55);
        border-radius: 22px;
        padding: 14px 18px;
        box-shadow: 0 8px 26px rgba(0,0,0,0.07);
        border: 1px solid rgba(255,255,255,0.45);
        display: inline-block;
    }
    .center { text-align: center; }
    .tiny-note { font-size: 0.85rem; opacity: 0.85; }
    .btn-row { display:flex; gap:12px; justify-content:center; flex-wrap: wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Config & Helpers
# ---------------------------
DEFAULT_MODEL = os.getenv("HF_MODEL_ID", "ibm-granite/granite-3-8b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Also support Streamlit Secrets if the user prefers that method
try:
    if not HF_TOKEN and "hf_token" in st.secrets:
        HF_TOKEN = st.secrets["hf_token"]
    if "hf_model_id" in st.secrets:
        DEFAULT_MODEL = st.secrets["hf_model_id"] or DEFAULT_MODEL
except Exception:
    pass

CURRENCY = "₹"  # You can change to "$" if you prefer


def format_money(n: float) -> str:
    try:
        return f"{CURRENCY}{n:,.0f}"
    except Exception:
        return f"{CURRENCY}{n}"


# ---------------------------
# Hugging Face Inference API (IBM Granite)
# ---------------------------
@st.cache_data(show_spinner=False)
def _warmup_model(model_id: str) -> None:
    """Make a tiny request to reduce cold-start in first user call."""
    if not HF_TOKEN:
        return
    _ = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": "hello", "parameters": {"max_new_tokens": 1}},
        timeout=30,
    )


def hf_generate(prompt: str, model_id: str = DEFAULT_MODEL, *, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "Missing HF_TOKEN. Set environment variable HF_TOKEN or add st.secrets['hf_token']."
        )
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # HF Inference API returns list[{'generated_text': '...'}]
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()
    # Some backends return dict
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"].strip()
    return str(data)


# ---------------------------
# NLU extraction using Granite via prompting
# ---------------------------
JSON_STRICT_INSTRUCTION = (
    "Return ONLY a valid JSON object with keys: 'sentiment' (one of 'positive'|'neutral'|'negative'),\n"
    "'keywords' (list of up to 5 strings), 'entities' (list of strings). No markdown, no commentary."
)


import textwrap

def build_nlu_prompt(user_text: str) -> str:
    """
    Builds a prompt for Natural Language Understanding analysis.
    """
    prompt = f"""
You are a Natural Language Understanding assistant.
Analyze the user's message and extract:
- sentiment: positive | neutral | negative
- keywords: up to 5 most important keywords
- entities: named entities (people, orgs, dates, money, etc.)

User message:
{user_text}
"""
    return textwrap.dedent(prompt)



def safe_json_loads(s: str) -> Dict[str, Any]:
    # Try to pull the first JSON object from a possibly messy string
    match = re.search(r"\{[\s\S]*\}", s)
    if match:
        s = match.group(0)
    try:
        return json.loads(s)
    except Exception:
        # try to correct single quotes
        try:
            s2 = s.replace("'", '"')
            return json.loads(s2)
        except Exception:
            return {"error": "Could not parse JSON", "raw": s}


# ---------------------------
# Q&A prompt
# ---------------------------

def build_qa_prompt(question: str, persona: str) -> str:
    return textwrap.dedent(f"""
    You are a helpful Personal Finance Assistant. Audience persona: {persona}.
    Answer the question below with a friendly, professional tone. Use concise numbered steps and short paragraphs.
    Avoid jargon; define any unavoidable terms briefly. Include 4–6 concrete actions.
    End with one motivating one-liner that starts with "Reminder:".

    Question:
    {question}

    Format strictly like this (no markdown fences):
    1) step one ...\n
    2) step two ...\n
    ...\n
    Reminder: ...
    """)


# ---------------------------
# Budget calculations (pure Python)
# ---------------------------
EXPENSE_KEYS_DEFAULT = [
    "rent", "groceries", "transport", "dining_out", "utilities", "subscriptions", "insurance", "education", "healthcare", "other"
]


def parse_user_budget(text: str) -> Dict[str, Any]:
    # accept JSON or python-ish dict with trailing commas
    try:
        return json.loads(text)
    except Exception:
        # fix common issues
        t = text.strip()
        t = re.sub(r",\s*}\s*$", "}", t)  # trailing comma before closing brace
        t = re.sub(r",\s*]\s*$", "]", t)
        try:
            t = t.replace("'", '"')
            return json.loads(t)
        except Exception:
            raise ValueError("Please provide a valid JSON object.")


def compute_budget(data: Dict[str, Any]) -> Dict[str, Any]:
    income = float(data.get("income", 0))
    if income <= 0:
        raise ValueError("'income' must be a positive number (annual).")

    expense_items = {k: float(v) for k, v in data.items() if k != "income"}
    # Ensure standard keys appear (even if zero)
    for k in EXPENSE_KEYS_DEFAULT:
        expense_items.setdefault(k, 0.0)

    total_expenses = sum(expense_items.values())
    monthly_income = income / 12.0
    monthly_expenses = total_expenses / 12.0
    disposable_annual = income - total_expenses
    disposable_monthly = monthly_income - monthly_expenses

    # Percent share of each expense
    expense_pct_of_income = {k: (v / income * 100.0 if income else 0.0) for k, v in expense_items.items()}

    # 50/30/20 rule classification (rough mapping)
    needs_keys = {"rent", "groceries", "utilities", "transport", "insurance", "healthcare", "education"}
    wants_keys = {"dining_out", "subscriptions", "shopping", "entertainment", "other"}

    needs = sum(expense_items.get(k, 0.0) for k in needs_keys)
    wants = sum(expense_items.get(k, 0.0) for k in wants_keys)
    savings = max(0.0, income - (needs + wants))

    return {
        "income": income,
        "expense_items": expense_items,
        "total_expenses": total_expenses,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "disposable_annual": disposable_annual,
        "disposable_monthly": disposable_monthly,
        "expense_pct_of_income": expense_pct_of_income,
        "needs": needs,
        "wants": wants,
        "savings": savings,
    }


def render_budget_report(b: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"* Annual Income: {format_money(b['income'])}")
    lines.append(f"  Monthly Income: {format_money(b['monthly_income'])}")
    lines.append("")
    lines.append(f"Total Annual Expenses: {format_money(b['total_expenses'])}")
    lines.append(f"Monthly Total Expenses: {format_money(b['monthly_expenses'])}")
    lines.append("")
    lines.append("Expense Breakdown:")
    for k, v in b["expense_items"].items():
        pct = b["expense_pct_of_income"][k]
        lines.append(f"- {k.title():<12} {format_money(v)}  ({pct:.1f}% of income)")
    lines.append("")
    lines.append(f"Disposable Income (Annual): {format_money(b['disposable_annual'])}")
    lines.append(f"Disposable Income (Monthly): {format_money(b['disposable_monthly'])}")
    lines.append("")
    # 50/30/20 quick compare
    inc = b['income']
    ideal_needs = 0.50 * inc
    ideal_wants = 0.30 * inc
    ideal_savings = 0.20 * inc

    lines.append("50/30/20 Snapshot (of Income):")
    lines.append(f"- Needs: {format_money(b['needs'])} (ideal ≤ {format_money(ideal_needs)})")
    lines.append(f"- Wants: {format_money(b['wants'])} (ideal ≤ {format_money(ideal_wants)})")
    lines.append(f"- Savings: {format_money(b['savings'])} (ideal ≥ {format_money(ideal_savings)})")

    return "\n".join(lines)


# ---------------------------
# UI Pages
# ---------------------------

def go(page: str):
    st.session_state["page"] = page


# ---- Home ----

def page_home():
    st.markdown("<div class='center'><div class='hero'><h2>💰 Personal Finance Chatbot: Intelligent Guidance for Savings, Taxes, and Investments</h2></div></div>", unsafe_allow_html=True)
    st.write("\n")
    st.markdown("<div class='center tiny-note'>Select one of the functions below</div>", unsafe_allow_html=True)
    st.write("\n")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔎 NLU Analysis", use_container_width=True):
            go("nlu")
    with col2:
        if st.button("📊 Budget Summary", use_container_width=True):
            go("budget")
    with col3:
        if st.button("❓ Q&A", use_container_width=True):
            go("qa")
    with col4:
        if st.button("💡 Spending Insights", use_container_width=True):
            go("insights")


# ---- NLU Analysis ----

def nlu_analyze_text(user_text: str, model_id: str) -> Dict[str, Any]:
    prompt = build_nlu_prompt(user_text)
    try:
        _warmup_model(model_id)
        out = hf_generate(prompt, model_id=model_id, max_new_tokens=200, temperature=0.0)
        return safe_json_loads(out)
    except Exception as e:
        return {"error": str(e)}


def page_nlu():
    st.markdown("<div class='hero center'><h3>🔎 NLU Analysis</h3></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("Enter either free text or JSON like {\"text\": \"...\"}")
        sample = {
            "text": "I'm finding it hard to save money each month. Can you help me manage my spending better?"
        }
        user_in = st.text_area("Enter input here:", value=json.dumps(sample, indent=2), height=150)
        model_id = st.text_input("Model (Hugging Face)", value=DEFAULT_MODEL, help="e.g., ibm-granite/granite-3-8b-instruct")
        run = st.button("Send", type="primary")
        back = st.button("⬅️ Back")

    if back:
        go("home")
    if run:
        try:
            obj = json.loads(user_in)
            text = obj.get("text", user_in)
        except Exception:
            text = user_in
        with st.spinner("Analyzing..."):
            result = nlu_analyze_text(text, model_id)
        st.markdown("**Result**")
        st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")


# ---- Q&A ----

def page_qa():
    st.markdown("<div class='hero center'><h3>❓ Q&A</h3></div>", unsafe_allow_html=True)
    with st.container(border=True):
        sample = {"question": "What percentage of my income should I save each month?", "persona": "professional"}
        user_in = st.text_area("Enter input here (JSON):", value=json.dumps(sample, indent=2), height=150)
        model_id = st.text_input("Model (Hugging Face)", value=DEFAULT_MODEL)
        run = st.button("Send", type="primary")
        back = st.button("⬅️ Back")

    if back:
        go("home")
    if run:
        try:
            obj = json.loads(user_in)
            question = obj.get("question") or ""
            persona = obj.get("persona") or "general"
            if not question:
                raise ValueError("'question' is required in the JSON input")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            return

        with st.spinner("Thinking..."):
            # Also show quick NLU on the question
            nlu = nlu_analyze_text(question, model_id)
            answer = hf_generate(build_qa_prompt(question, persona), model_id=model_id, max_new_tokens=450, temperature=0.2)

        st.markdown("**Analysis**")
        st.code(json.dumps(nlu, indent=2, ensure_ascii=False), language="json")
        st.markdown("**Answer**")
        st.code(answer, language="markdown")


# ---- Budget Summary ----

def page_budget():
    st.markdown("<div class='hero center'><h3>📊 Budget Summary</h3></div>", unsafe_allow_html=True)
    with st.container(border=True):
        sample = {
            "income": 240000,
            "rent": 96000,
            "groceries": 36000,
            "transport": 18000,
            "dining_out": 12000,
            "utilities": 15000,
            "subscriptions": 6000,
        }
        user_in = st.text_area("Enter input here (annual figures, JSON):", value=json.dumps(sample, indent=2), height=180)
        run = st.button("Send", type="primary")
        back = st.button("⬅️ Back")

    if back:
        go("home")
    if run:
        try:
            data = parse_user_budget(user_in)
            b = compute_budget(data)
        except Exception as e:
            st.error(str(e))
            return

        st.markdown("**Summary**")
        st.code(render_budget_report(b), language="text")

        # Store for insights page
        st.session_state["_last_budget"] = b


# ---- Spending Insights ----

def insights_from_budget(b: Dict[str, Any]) -> List[str]:
    tips: List[str] = []

    # Identify top 3 categories by share of income
    pct = b["expense_pct_of_income"]
    top3 = sorted(pct.items(), key=lambda kv: kv[1], reverse=True)[:3]
    if top3:
        cat_list = ", ".join([f"{k} ({v:.1f}%)" for k, v in top3])
        tips.append(f"Top expense drivers: {cat_list}. Focus here first.")

    # If wants > 30% suggest a specific target
    wants_share = b["wants"] / b["income"] * 100.0 if b["income"] else 0.0
    if wants_share > 30:
        target = max(0, wants_share - 30)
        tips.append(f"Wants are {wants_share:.1f}% of income. Aim to lower by ~{target:.1f} pp to meet the 30% guideline.")

    # If savings < 20% recommend automation
    savings_share = b["savings"] / b["income"] * 100.0 if b["income"] else 0.0
    if savings_share < 20:
        tips.append("Savings below 20%—set up an automatic transfer on payday toward an emergency fund and goals.")

    # If dining_out high
    if b["expense_pct_of_income"].get("dining_out", 0.0) >= 5.0:
        tips.append("Dining out is sizeable—try a weekly cap or meal-prep two days to trim costs.")

    # Transport nudge
    if b["expense_pct_of_income"].get("transport", 0.0) >= 8.0:
        tips.append("Transport is high—batch errands, consider public transit or shared rides where possible.")

    return tips


def page_insights():
    st.markdown("<div class='hero center'><h3>💡 Spending Insights</h3></div>", unsafe_allow_html=True)
    if "_last_budget" not in st.session_state:
        st.info("Run **Budget Summary** first, then come back here for tailored insights.")
        if st.button("Go to Budget Summary"):
            go("budget")
        return

    b = st.session_state["_last_budget"]
    st.markdown("**Quick Insights**")
    tips = insights_from_budget(b)
    if not tips:
        st.success("Your budget looks balanced. Keep it up and review monthly!")
    else:
        for t in tips:
            st.write("- ", t)

    if HF_TOKEN:
        st.write("")
        st.markdown("**LLM Suggestions (Granite)**")
        # Build a compact summary prompt for the LLM
        exp_lines = ", ".join([f"{k}:{v:.0f}" for k, v in b["expense_items"].items()])
        brief = textwrap.dedent(f"""
        User annual income: {b['income']:.0f}. Expenses: {exp_lines}.
        Give 5 practical, specific tips (1–2 lines each) to improve savings next month without severe lifestyle cuts.
        Avoid generic advice; be concrete.
        """)
        try:
            _warmup_model(DEFAULT_MODEL)
            tips_llm = hf_generate(brief, model_id=DEFAULT_MODEL, max_new_tokens=250, temperature=0.3)
            st.code(tips_llm, language="markdown")
        except Exception as e:
            st.warning(f"LLM tip generation unavailable: {e}")

    if st.button("⬅️ Back"):
        go("home")


# ---------------------------
# Router
# ---------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

page = st.session_state["page"]
if page == "home":
    page_home()
elif page == "nlu":
    page_nlu()
elif page == "qa":
    page_qa()
elif page == "budget":
    page_budget()
elif page == "insights":
    page_insights()
else:
    page_home()
