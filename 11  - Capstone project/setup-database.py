import sqlite3
import os
from pathlib import Path

def setup_database():
    """
    Create and populate a SQLite database for the business intelligence agent.
    """
    print("Setting up business database...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Database path
    db_path = data_dir / "business_data.db"
    
    # Connect to SQLite database (creates file if not exists)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create business_info table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS business_info (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        industry TEXT,
        revenue REAL,
        employees INTEGER,
        founded INTEGER,
        headquarters TEXT,
        customer_satisfaction REAL
    )
    """)
    
    # Create products table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        price REAL,
        stock INTEGER,
        category TEXT
    )
    """)
    
    # Create orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date TEXT,
        total_amount REAL,
        status TEXT
    )
    """)
    
    # Create order_items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        price REAL,
        FOREIGN KEY (order_id) REFERENCES orders (id),
        FOREIGN KEY (product_id) REFERENCES products (id)
    )
    """)
    
    # Check if business_info already has data
    cursor.execute("SELECT COUNT(*) FROM business_info")
    if cursor.fetchone()[0] == 0:
        # Insert sample business info
        cursor.execute("""
        INSERT INTO business_info (name, industry, revenue, employees, founded, headquarters, customer_satisfaction)
        VALUES ('Acme Technologies', 'Software Development', 5000000.00, 120, 2010, 'San Francisco, CA', 4.7)
        """)
    
    # Check if products table has data
    cursor.execute("SELECT COUNT(*) FROM products")
    if cursor.fetchone()[0] == 0:
        # Insert sample products
        products = [
            ('Business Analytics Suite', 'Advanced analytics platform for enterprise', 1999.99, 50, 'Software'),
            ('Cloud Storage Pro', 'Secure cloud storage solution', 99.99, 200, 'Services'),
            ('Smart Office Assistant', 'AI-powered productivity tool', 299.99, 75, 'Software'),
            ('Network Security Package', 'Enterprise-grade security solution', 1499.99, 30, 'Security'),
            ('Professional Development Course', 'Online training for developers', 199.99, 100, 'Training'),
            ('Technical Support Subscription', 'Annual support contract', 599.99, 150, 'Services'),
            ('Virtual Server Package', 'High-performance virtual servers', 799.99, 80, 'Infrastructure'),
            ('Mobile App Development Kit', 'Tools for mobile developers', 349.99, 45, 'Development'),
            ('Data Backup Solution', 'Automated backup services', 249.99, 120, 'Services'),
            ('Enterprise CRM', 'Customer relationship management system', 2499.99, 25, 'Software')
        ]
        
        cursor.executemany("""
        INSERT INTO products (name, description, price, stock, category)
        VALUES (?, ?, ?, ?, ?)
        """, products)
    
    # Check if orders table has data
    cursor.execute("SELECT COUNT(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        # Insert sample orders
        orders = [
            (101, '2023-01-15', 1999.99, 'completed'),
            (102, '2023-02-03', 599.99, 'completed'),
            (103, '2023-02-17', 3499.98, 'completed'),
            (104, '2023-03-05', 249.99, 'completed'),
            (105, '2023-03-22', 1499.99, 'completed'),
            (106, '2023-04-10', 299.99, 'completed'),
            (107, '2023-04-28', 2499.99, 'pending'),
            (108, '2023-05-15', 799.99, 'pending'),
            (109, '2023-05-30', 699.98, 'pending'),
            (110, '2023-06-10', 4999.97, 'pending')
        ]
        
        cursor.executemany("""
        INSERT INTO orders (customer_id, order_date, total_amount, status)
        VALUES (?, ?, ?, ?)
        """, orders)
    
    # Check if order_items table has data
    cursor.execute("SELECT COUNT(*) FROM order_items")
    if cursor.fetchone()[0] == 0:
        # Insert sample order items
        order_items = [
            (1, 1, 1, 1999.99),
            (2, 6, 1, 599.99),
            (3, 2, 3, 299.99),
            (3, 5, 1, 199.99),
            (4, 9, 1, 249.99),
            (5, 4, 1, 1499.99),
            (6, 3, 1, 299.99),
            (7, 10, 1, 2499.99),
            (8, 7, 1, 799.99),
            (9, 3, 1, 299.99),
            (9, 5, 2, 199.99),
            (10, 1, 1, 1999.99),
            (10, 2, 1, 99.99),
            (10, 4, 1, 1499.99),
            (10, 5, 7, 199.99)
        ]
        
        cursor.executemany("""
        INSERT INTO order_items (order_id, product_id, quantity, price)
        VALUES (?, ?, ?, ?)
        """, order_items)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Database created successfully at: {db_path}")
    print("Sample data loaded into tables: business_info, products, orders, order_items")

if __name__ == "__main__":
    setup_database()
