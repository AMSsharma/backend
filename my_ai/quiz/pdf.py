def text_from_pdf(url: str) -> str:
    import requests
    from io import BytesIO
    import fitz 

    response = requests.get(url)
    pdf_data = BytesIO(response.content)
    doc = fitz.open("pdf", pdf_data)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
