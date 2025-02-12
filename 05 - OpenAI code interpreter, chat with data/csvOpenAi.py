import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'sales_data_sample.csv'

df = pd.read_csv(CSV_PATH)

numeric_columns = df.select_dtypes(include=['number']).columns
if len(numeric_columns) == 0:
    print("❌ Числовые колонки не найдены!")
else:
    column_to_plot = numeric_columns[0]

    plt.figure(figsize=(8, 5))
    plt.hist(df[column_to_plot].dropna(), bins=20, edgecolor='black')
    plt.xlabel(column_to_plot)
    plt.ylabel('Частота')
    plt.title(f'Гистограмма {column_to_plot}')

    # Сохранение изображения
    plt.savefig('histogram.png')
    plt.show()

    print(f"📊 Гистограмма сохранена как 'histogram.png'")
