import fitz  # PyMuPDF
import openai

openai.api_key = 'YOUR_OPENAI_API_KEY'

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def ask_gpt4(question, context):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150
    )
    return response.choices[0].text.strip()

pdf_path = 'pride_and_prejudice.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

question = "What is the main theme of the book?"
answer = ask_gpt4(question, pdf_text[:2000])

print(f"Question: {question}")
print(f"Answer: {answer}")
