import whisper
import subprocess

def trim_audio(input_path, output_path, start_time, duration):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def transcribe_audio(file_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path)
    return result

if __name__ == "__main__":
    audio_path = "ITPU_MS_Degree_Session_5_-_Generative_AI-20241213_153714-Meeting_Recording.mp3"
    trimmed_audio_path = "trimmed_audio.mp3"
    mode = input("Выберите режим (full - полный файл, part - часть записи): ").strip().lower()

    if mode == "part":
        start_time = int(input("Введите начальное время в секундах (например, 0): ").strip())
        duration = int(input("Введите длительность в секундах (например, 10): ").strip())
        trim_audio(audio_path, trimmed_audio_path, start_time, duration)
        result = transcribe_audio(trimmed_audio_path)
    else:
        result = transcribe_audio(audio_path)

    print("\nРаспознанный текст:")
    print(result["text"])

    output_file = "transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])
    print(f"\nТранскрипция сохранена в файл {output_file}")
