import openai

def get_ai_response(question):
    """Отправляет запрос в OpenAI и получает ответ."""
    openai.api_key = "your-api-key"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": question}]
    )
    return response["choices"][0]["message"]["content"]
