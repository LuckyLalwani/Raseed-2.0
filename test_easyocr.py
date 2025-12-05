import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('ReceiptSwiss.jpg', detail=0)

print("\nEASY OCR OUTPUT:")
for line in result:
    print(line)