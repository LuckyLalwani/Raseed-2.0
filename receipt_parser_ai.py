import os
import json
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# LangChain + OpenAI
from langchain_openai import ChatOpenAI  # modern import


# 1️⃣ Load environment variables from .env
load_dotenv()

# (Optional) You can check if key is loaded:
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Make sure it's set in your .env file.")


# 2️⃣ Tell pytesseract where Tesseract is installed (same as before)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ^^^ ⬆️ CHANGE THIS if your tesseract.exe path is different


def ocr_image_to_text(image_path: str) -> str:
    """Run OCR on the image and return raw text."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng")
    return text


from langchain_community.chat_models import ChatOllama

def build_llm():
    return ChatOllama(
        model="llama3.2",  # or "llama3" or "llama2:7b"
        temperature=0
    )



def llm_parse_receipt(raw_text: str) -> dict:
    """
    Use an LLM to convert messy receipt text into clean JSON.
    """

    llm = build_llm()

    # Prompt that tells the model exactly what we want
    prompt = f"""
You are an expert at reading store receipts.

Extract the following fields from the receipt text below. 
If a field is missing or unclear, use null.

Return ONLY valid JSON, no explanation, in this exact format:

{{
  "vendor": string or null,
  "date": string or null,
  "items": [
    {{"name": string, "price": number}}
  ],
  "total": number or null,
  "category": string or null
}}

Receipt text:
\"\"\"{raw_text}\"\"\"
"""

    # Call the model using LangChain's .invoke()
    response = llm.invoke(prompt)

    # Get the string content from the AI's reply
    content = response.content

    # Try to parse as JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # If the model added extra text, try to extract JSON part only
        try:
            # Find first and last curly brace
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            data = json.loads(json_str)
        except Exception:
            print("⚠️ Could not parse JSON from model output. Raw content:")
            print(content)
            raise

    return data


def parse_receipt_image(image_path: str) -> dict:
    """
    Full pipeline:
    1. OCR → raw text
    2. LLM → structured JSON
    """
    print(f"🖼️ Running OCR on: {image_path}")
    raw_text = ocr_image_to_text(image_path)
    print("\n===== OCR TEXT (preview) =====")
    print(raw_text[:500])  # show first 500 chars

    print("\n🧠 Sending to LLM for structured parsing...")
    result = llm_parse_receipt(raw_text)

    return {
        **result,
        "raw_text": raw_text  # also keep original text
    }


if __name__ == "__main__":
    # ⬇️ CHANGE THIS to your image file name or path
    image_path = "ReceiptSwiss.jpg"

    data = parse_receipt_image(image_path)

    print("\n===== STRUCTURED RECEIPT DATA =====")
    from pprint import pprint
    pprint(data)
