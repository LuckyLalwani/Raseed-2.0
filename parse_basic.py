import re

receipt_text = """
----- ZOMATO -----
Paneer Tikka 250
Butter Naan 80
GST#AX39939
TOTAL: 330.00
"""

# Vendor: first non-empty line
lines = [l.strip() for l in receipt_text.split("\n") if l.strip()]
vendor = lines[0]

# Total amount
match = re.search(r"(Total|TOTAL)[^\d]*([\d\.]+)", receipt_text)
total = float(match.group(2)) if match else None

print("Vendor:", vendor)
print("Total:", total)
