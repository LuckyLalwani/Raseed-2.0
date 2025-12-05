import os
from typing import TypedDict, Optional

import pytesseract
from PIL import Image

from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama


# 🔧 Configure Tesseract path (same as your working scripts)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ^^^ change this if needed


# 🧠 Local LLM via Ollama (same model you used before)
def build_llm():
    return ChatOllama(
        model="llama3.2",  # or "llama3" if you pulled that
        temperature=0
    )


# 🧱 1. Define the state that flows through the graph
class ReceiptState(TypedDict, total=False):
    image_path: str
    raw_text: str
    structured_data: dict


# 🧩 2. Define nodes (functions that operate on the state)

def ocr_node(state: ReceiptState) -> ReceiptState:
    """Take image_path from state, run OCR, add raw_text to state."""
    image_path = state["image_path"]
    print(f"🔍 [OCR NODE] Running OCR on: {image_path}")

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng")

    new_state: ReceiptState = {
        **state,
        "raw_text": text,
    }
    return new_state


def parse_node(state: ReceiptState) -> ReceiptState:
    """
    Use LLM to turn raw_text into structured JSON-like dict.
    """
    raw_text = state["raw_text"]
    print("🧠 [PARSE NODE] Sending OCR text to LLM...")

    llm = build_llm()

    prompt = f"""
You are an expert at extracting structured data from store receipts.

Return a JSON object with the following structure:

{{
  "vendor": string or null,
  "date": string or null,
  "items": [
    {{
      "name": string,
      "price": number,
      "category": string or null
    }}
  ],
  "total": number or null,
  "main_category": string or null
}}

💡 Rules for categorization:
- Use item-based inference, not only vendor.
- If vendor is food delivery platform (Zomato/Swiggy/Uber Eats), categorize based on dishes.
- If a price looks like a service fee, categorize as "Service".
- Receipts often show "Subtotal", "GST", "CGST", "Taxes". Ignore them as items.
- Only categorize actual products.

🎯 Common category examples:
- Food / Restaurant: "Paneer Tikka", "Chicken Biryani", "Burger", "Pizza"
- Transport / Rides: "Uber Trip", "Cab Ride", "Metro Ticket"
- Groceries: "Egg", "Rice", "Milk", "Vegetables"
- Shopping / Clothing: "T-shirt", "Shirt", "Shoes"
- Bills / Utilities: "Electricity", "Internet"
- Pharmacy / Health: "Paracetamol", "Vitamin C"

📌 If unsure, use "Other" instead of null.
Do not guess the price category from total. Only infer from item names.
main_category must be inferred from the majority of items by price OR count.
Receipt text:
\"\"\"{raw_text}\"\"\"
"""


    response = llm.invoke(prompt)
    content = response.content

    # We'll be a bit forgiving: just try to eval-ish parse it
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            data = json.loads(json_str)
        except Exception:
            print("⚠️ [PARSE NODE] Could not parse JSON from LLM output:")
            print(content)
            raise

    print("✅ [PARSE NODE] Parsed structured data:")
    from pprint import pprint
    pprint(data)

    new_state: ReceiptState = {
        **state,
        "structured_data": data,
    }
    return new_state


def store_node(state: ReceiptState) -> ReceiptState:
    """
    For now, just save structured_data into a local JSONL file to simulate storage.
    Later we'll replace this with a proper DB.
    """
    data = state["structured_data"]
    print("💾 [STORE NODE] Saving structured data to receipts_store.jsonl")

    import json

    os.makedirs("data", exist_ok=True)
    store_path = os.path.join("data", "receipts_store.jsonl")

    with open(store_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")

    print(f"✅ [STORE NODE] Saved to {store_path}")

    return state  # nothing new added


# 🕸️ 3. Build the LangGraph

def build_graph():
    graph = StateGraph(ReceiptState)

    # Add nodes
    graph.add_node("ocr", ocr_node)
    graph.add_node("parse", parse_node)
    graph.add_node("store", store_node)

    # Define edges (flow):
    # START -> ocr -> parse -> store -> END
    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "parse")
    graph.add_edge("parse", "store")
    graph.add_edge("store", END)

    # Compile into an executable graph/app
    app = graph.compile()
    return app


if __name__ == "__main__":
    # 👇 Change this to your receipt image
    image_path = "ReceiptSwiss.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    app = build_graph()

    # The initial state
    initial_state: ReceiptState = {
        "image_path": image_path
    }

    print("🚀 Running LangGraph receipt pipeline...\n")

    # run the graph synchronously
    final_state = app.invoke(initial_state)

    print("\n🏁 [DONE] Final state:")
    from pprint import pprint
    pprint(final_state)
