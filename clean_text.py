import re

raw = """
----- ZOMATO -----
Paneer Tikka 250
Butter Naan 80
GST#AX39939
TOTAL: 330.00
"""

def clean(raw):
    lines = raw.split('\n')
    cleaned = [l.strip() for l in lines if len(l.strip()) > 0]
    return cleaned

print(clean(raw))
