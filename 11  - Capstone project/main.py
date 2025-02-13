import smtplib
import sqlite3
import openai
import streamlit as st
import socket
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import matplotlib.pyplot as plt

# Configure SMTP for mail service
SMTP_SERVER = "smtp.mail.ru"
SMTP_PORT = 587
EMAIL_SENDER = "your-email@mail.ru"  # Your email address
EMAIL_PASSWORD = "your-app-password"  # App-specific password for 2FA
EMAIL_HR = "hr-email@example.com"

# Set up OpenAI API
openai.api_key = "your-openai-api-key"


# Send email function
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, to_email, msg.as_string())


# Function to create logs table
def create_logs_table():
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_ip TEXT,
            user_query TEXT,
            sql_query TEXT,
            result TEXT
        )
    ''')

    conn.commit()
    conn.close()


# Execute SQL query
def execute_query(query):
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    conn.close()

    return result, columns


# Handle user query and generate SQL
def handle_query(user_input, user_ip):
    sql_prompt = f"""
    You are a SQL expert. I have a table called 'employees' with the following columns: 
    id, first_name, last_name, age, position, management, department, experience, passport_id, bank_card_code, driver_license_category, salary, address.
    Based on the following query: "{user_input}", create a valid SQL query.
    Please return only the SQL query with no explanation or other text.
    """

    # Send query to OpenAI
    sql_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": sql_prompt}]
    )

    # Get SQL query
    sql_query = sql_response["choices"][0]["message"]["content"].strip()

    # Print the generated SQL query for debugging
    print(f"Generated SQL Query: {sql_query}")

    if not sql_query:
        return "", ""

    try:
        # Execute SQL query
        result, columns = execute_query(sql_query)
    except sqlite3.OperationalError as e:
        # Handle SQL errors
        print(f"SQL Error: {e}")
        return "", ""

    # Create DataFrame from query result
    df = pd.DataFrame(result, columns=columns) if result and columns else pd.DataFrame()

    # Log the query
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_query(timestamp, user_ip, user_input, sql_query, str(result))

    return df


# Log the user query and result
def log_query(timestamp, user_ip, user_query, sql_query, result):
    conn = sqlite3.connect("employees.db")
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO logs (timestamp, user_ip, user_query, sql_query, result)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, user_ip, user_query, sql_query, result))

    conn.commit()
    conn.close()


# Streamlit UI
st.title("HR Salary Query Bot")

st.sidebar.header("Filters")
department_filter = st.sidebar.selectbox("Select department", ["All", "IT", "HR", "Finance", "Sales"])
management_filter = st.sidebar.selectbox("Select management level", ["All", "Upper", "Middle", "Lower"])
position_filter = st.sidebar.selectbox("Select position", ["All", "Manager", "Engineer", "Analyst", "Developer"])
age_filter = st.sidebar.slider("Select age", 18, 65, (25, 40))

create_logs_table()

user_input = st.text_input("Enter your query:")
if st.button("Query"):
    user_ip = socket.gethostbyname(socket.gethostname())
    df = handle_query(user_input, user_ip)

    if not df.empty:
        # Display column names for debugging
        st.write(f"Columns in the result: {df.columns.tolist()}")

        # Convert result to human-readable format
        readable_result = ""

        # Check if the query result is aggregated (e.g., MAX(salary)) or detailed
        if 'MAX(salary)' in df.columns:
            readable_result = "The query returned an aggregated result. For example, the maximum salary:\n"
            readable_result += f"Max Salary: {df.iloc[0]['MAX(salary)']}\n"
        else:
            if 'first_name' in df.columns and 'last_name' in df.columns and 'position' in df.columns and 'department' in df.columns and 'salary' in df.columns:
                for index, row in df.iterrows():
                    readable_result += f"Employee {row['first_name']} {row['last_name']} works as a {row['position']} in {row['department']} department, earning {row['salary']}.\n"
            else:
                readable_result = "The expected columns were not found in the query result."

        st.subheader("Query Results")
        st.text(readable_result)  # Display the human-readable result

        # Apply filters
        if "department" in df.columns and department_filter != "All":
            df = df[df["department"] == department_filter]
        if "management" in df.columns and management_filter != "All":
            df = df[df["management"] == management_filter]
        if "position" in df.columns and position_filter != "All":
            df = df[df["position"] == position_filter]
        if "age" in df.columns:
            df = df[(df["age"] >= age_filter[0]) & (df["age"] <= age_filter[1])]

        # Create pie chart for salary distribution by department
        if "department" in df.columns and "salary" in df.columns:
            department_salary = df.groupby('department')['salary'].sum()
            fig, ax = plt.subplots()
            ax.pie(department_salary, labels=department_salary.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.subheader("Salary Distribution by Department")
            st.pyplot(fig)  # Display the pie chart

        # If a department filter is applied, show the employee salary details sorted by salary
        if department_filter != "All" and "salary" in df.columns:
            department_df = df[df['department'] == department_filter]
            sorted_df = department_df[['first_name', 'last_name', 'salary', 'age']].sort_values(by='salary',
                                                                                                ascending=False)
            st.subheader(f"Employee Salary Details in {department_filter} Department")
            st.dataframe(sorted_df)  # Display the sorted table
    else:
        st.write("No data found for your query.")
