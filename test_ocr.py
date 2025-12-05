import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Only change this path ↓
img = Image.open("ReceiptSwiss.jpg")

text = pytesseract.image_to_string(img)
print("OCR OUTPUT:\n")
print(text)
