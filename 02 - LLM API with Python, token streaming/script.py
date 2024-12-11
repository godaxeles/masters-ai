import openai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

def generate_blog_post(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    print("Generating blog post...")
    blog_post = ""
    for chunk in response:
        if 'choices' in chunk:
            content = chunk['choices'][0]['delta'].get('content', '')
            print(content, end='', flush=True)
            blog_post += content
    print("Blog post generation completed.")
    return blog_post

if __name__ == "__main__":
    prompt = "Create a blog post summarizing the key points of the first lecture on Generative AI."
    blog_post = generate_blog_post(prompt)

    with open("generated_blog_post.txt", "w") as file:
        file.write(blog_post)
    print("Blog post saved to 'generated_blog_post.txt'.")
