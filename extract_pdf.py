import pdfplumber

pdf = pdfplumber.open(r'C:\Users\BROU WILLIAMS\Downloads\Archipel_Project_Complete\archipel\Archipel_Documentation_Complete.pdf')
print(f'Pages: {len(pdf.pages)}')
for i, page in enumerate(pdf.pages):
    text = page.extract_text()
    print(f'--- PAGE {i+1} ---')
    if text:
        print(text)
    else:
        print('[NO EXTRACTABLE TEXT]')
