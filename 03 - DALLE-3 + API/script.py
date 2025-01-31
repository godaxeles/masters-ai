import openai

# Установите ваш API-ключ OpenAI
openai.api_key = ""

# Список стилей для генерации изображений
styles = [
    "photorealistic",
    "cartoon",
    "oil painting",
    "watercolor",
    "pencil sketch",
    "cyberpunk",
    "abstract",
    "fantasy",
    "minimalistic"
]

# Функция для генерации изображений
def generate_images(prompt):
    images = []
    for style in styles:
        styled_prompt = f"{prompt}, style: {style}"
        try:
            # Генерация изображения
            response = openai.Image.create(
                prompt=styled_prompt,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']
            images.append((style, image_url))
        except Exception as e:
            print(f"Ошибка при генерации изображения для стиля {style}: {e}")
    return images

# Основная программа
if __name__ == "__main__":
    import sys
    # Установка кодировки UTF-8 для ввода
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Ввод текстового запроса
    user_prompt = input("Введите текстовый запрос для генерации изображений: ").strip()

    results = generate_images(user_prompt)

    print("\nСгенерированные изображения:")
    for style, url in results:
        print(f"Стиль: {style}, URL: {url}")
