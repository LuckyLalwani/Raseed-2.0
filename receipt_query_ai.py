import os
import json
from typing import List, Dict, Any

from langchain_community.chat_models import ChatOllama


# 🧠 Same LLM setup you used before
def build_llm():
    return ChatOllama(
        model="llama3.2",  # or "llama3" if you're using that
        temperature=0
    )


def compute_basic_stats(receipts):
    """Compute grand total and totals by category/vendor."""
    grand_total = 0.0
    by_category = {}
    by_vendor = {}

    for r in receipts:
        total = r.get("total") or 0
        grand_total += total

        cat = r.get("main_category") or r.get("category") or "Unknown"
        by_category[cat] = by_category.get(cat, 0) + total

        vendor = r.get("vendor") or "Unknown"
        by_vendor[vendor] = by_vendor.get(vendor, 0) + total

    return {
        "grand_total": grand_total,
        "by_category": by_category,
        "by_vendor": by_vendor,
    }


def load_receipts(store_path: str = "data/receipts_store.jsonl") -> List[Dict[str, Any]]:
    """Load all stored receipt entries from JSONL file."""
    if not os.path.exists(store_path):
        print(f"⚠️ Store file not found: {store_path}")
        return []

    receipts = []
    with open(store_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                receipts.append(json.loads(line))
            except json.JSONDecodeError:
                print("⚠️ Skipping invalid JSON line in store file.")
    return receipts


def build_query_prompt(
    receipts: List[Dict[str, Any]],
    question: str,
    stats: Dict[str, Any],
) -> str:
    """
    Build a prompt that gives the model:
    - all receipts
    - precomputed stats (numbers calculated by Python)
    and forces it to use stats for any numeric answer.
    """
    compact_receipts = []
    for r in receipts:
        compact_receipts.append({
            "vendor": r.get("vendor"),
            "date": r.get("date"),
            "items": r.get("items"),
            "total": r.get("total"),
            "main_category": r.get("main_category", r.get("category")),
        })

    receipts_json = json.dumps(compact_receipts, ensure_ascii=False, indent=2)
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

    prompt = f"""
You are an AI financial assistant.

You are given:

1) RECEIPTS_JSON: a list of receipts with vendor, date, items (name, price, category), total, main_category.
2) STATS_JSON: precomputed totals, calculated by Python, which is ALWAYS CORRECT:
   - "grand_total": sum of all receipt totals
   - "by_category": sum of totals per category
   - "by_vendor": sum of totals per vendor

VERY IMPORTANT RULES (FOLLOW STRICTLY):

- Whenever you mention ANY numeric total, you MUST use the values from STATS_JSON.
- DO NOT recompute totals by adding numbers from RECEIPTS_JSON yourself.
- Treat STATS_JSON as the single source of truth for all sums.
- If something is not available in STATS_JSON, you can reason from RECEIPTS_JSON but only for non-numeric stuff (like vendor names, item names, categories).
- For questions like "total on Food" or "total on <category>", always answer using by_category from STATS_JSON.
- For questions like "total spent at <vendor>", always answer using by_vendor from STATS_JSON.

RECEIPTS_JSON:
{receipts_json}

STATS_JSON:
{stats_json}

User question:
\"\"\"{question}\"\"\"

Your response format:

1. First give a clear, concise natural language answer.
2. Then show a "Calculations" section where you explicitly reference the values from STATS_JSON that you used.
3. If something cannot be answered exactly from the data, say that clearly instead of guessing.
4. Always express money in Indian Rupees using the symbol ₹. Do NOT use $, USD, or any other currency symbol.

Answer now.
"""
    return prompt



def answer_question(question: str):
    receipts = load_receipts()
    if not receipts:
        print("⚠️ No receipts found yet. Run the ingestion pipeline first.")
        return

    stats = compute_basic_stats(receipts)
    llm = build_llm()
    prompt = build_query_prompt(receipts, question, stats)

    print("\n🧠 Sending question + stats to LLM...\n")
    response = llm.invoke(prompt)
    print("===== ANSWER =====\n")
    print(response.content)



if __name__ == "__main__":
    print("📊 Receipt Query Assistant")
    print("Type your question about your spending.")
    print("Example: 'How much did I spend on Food?'")
    print("Type 'exit' to quit.\n")

    while True:
        q = input("❓ Your question: ")
        if q.lower().strip() in ["exit", "quit", "q"]:
            print("👋 Bye!")
            break
        answer_question(q)
        print("\n" + "="*60 + "\n")
