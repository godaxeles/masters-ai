import sqlite3
import random
from faker import Faker

fake = Faker()

def create_database():
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    
    # Удаляем таблицу, если она существует, для корректного пересоздания
    cursor.execute("DROP TABLE IF EXISTS employees")
    
    # Создаем таблицу заново
    cursor.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            age INTEGER,
            position TEXT,
            management TEXT,
            department TEXT,
            experience INTEGER,
            passport_id TEXT UNIQUE,
            bank_card_code TEXT UNIQUE,
            driver_license_category TEXT,
            salary INTEGER,
            address TEXT
        )
    ''')
    
    positions = ["Manager", "Engineer", "Analyst", "Specialist", "Director"]
    departments = ["IT", "HR", "Finance", "Marketing", "Sales", "Operations"]
    driver_license_categories = ["B", "C", ""]
    
    employees = []
    for _ in range(100):
        first_name = fake.first_name()
        last_name = fake.last_name()
        age = random.randint(22, 65)
        position = random.choice(positions)
        management = fake.word().capitalize()
        department = random.choice(departments)
        experience = random.randint(1, age - 18)
        passport_id = str(fake.unique.random_number(digits=9))  # Гарантируем строковый формат
        bank_card_code = fake.unique.credit_card_number()
        driver_license_category = random.choice(driver_license_categories)
        salary = random.randint(30, 120) * 1000  # Окончание на "00"
        address = fake.address()
        
        employees.append((first_name, last_name, age, position, management, department, experience, passport_id, bank_card_code, driver_license_category, salary, address))
    
    cursor.executemany('''
        INSERT INTO employees (first_name, last_name, age, position, management, department, experience, passport_id, bank_card_code, driver_license_category, salary, address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', employees)
    
    conn.commit()
    conn.close()
    print("Database created successfully with 100 employees!")

if __name__ == "__main__":
    create_database()
