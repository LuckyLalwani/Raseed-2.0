import pytesseract
from PIL import Image
import re

# 👇 VERY IMPORTANT: put your actual path to tesseract.exe here
# If this line already worked in your test_ocr.py, copy the SAME path here.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_image_to_text(image_path: str) -> str:
    """Run OCR on the image and return raw text."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng")
    return text


def clean_text(text: str):
    """Split into non-empty, stripped lines."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # remove empty lines
    return lines


def extract_vendor(lines):
    """
    Very simple heuristic:
    - assume the first non-empty line is the vendor/store name
    """
    if not lines:
        return None
    return lines[0]


def extract_total(text: str):
    """
    Try to find a line like:
    TOTAL: 330.00
    Grand Total 1234
    Amount Due 999.50
    """
    pattern = r"(TOTAL|Total|Amount Due|Grand Total)[^\d]*([\d,.]+)"
    match = re.search(pattern, text)
    if not match:
        return None

    amount_str = match.group(2).replace(",", "")
    try:
        return float(amount_str)
    except ValueError:
        return None


def extract_date(text: str):
    """
    Try to find common date formats:
    - 31/12/2025
    - 31-12-25
    - 2025-12-31
    This is basic and may not work for all receipts, but it's a start.
    """
    patterns = [
        r"(\d{2}[/-]\d{2}[/-]\d{2,4})",
        r"(\d{4}[/-]\d{2}[/-]\d{2})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


def extract_items(lines):
    """
    Naive item extraction:
    Look for lines that end with a number (price), like:
    'Paneer Tikka 250'
    'Butter Naan 80'
    """
    items = []
    for line in lines:
        # Match: some text, then spaces, then a number at the end
        m = re.search(r"(.+?)\s+(\d+(\.\d+)?)$", line)
        if m:
            name = m.group(1).strip(" .:-")
            price = float(m.group(2))

            # skip lines that are probably totals
            if name.lower() in ["total", "subtotal", "grand total"]:
                continue

            items.append({"name": name, "price": price})
    return items


def categorize_expense(vendor: str):
    """
    Very simple rule-based categorization using vendor name.
    You can improve this later with AI.
    """
    if not vendor:
        return "Other"

    v = vendor.lower()

    if any(word in v for word in ["zomato", "swiggy", "domino", "pizza", "restaurant", "cafe"]):
        return "Food"

    if any(word in v for word in ["uber", "ola", "rapido", "metro", "bus", "rail", "train"]):
        return "Transport"

    if any(word in v for word in ["amazon", "flipkart", "myntra", "ajio"]):
        return "Shopping"

    return "Other"


def parse_receipt(image_path: str):
    """
    Full pipeline:
    1. OCR -> raw text
    2. Clean lines
    3. Extract vendor, date, total, items, category
    4. Return everything as a dict
    """
    raw_text = ocr_image_to_text(image_path)
    lines = clean_text(raw_text)

    vendor = extract_vendor(lines)
    total = extract_total(raw_text)
    date = extract_date(raw_text)
    items = extract_items(lines)
    category = categorize_expense(vendor)

    data = {
        "vendor": vendor,
        "date": date,
        "items": items,
        "total": total,
        "category": category,
        "raw_text": raw_text,
    }
    return data


if __name__ == "__main__":
    # 👇 Change this to your actual image file name
    image_path = "ReceiptSwiss.jpg"

    result = parse_receipt(image_path)

    # pretty-print the dict
    from pprint import pprint
    pprint(result)
