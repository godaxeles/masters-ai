import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(recipient, message):
    """Отправляет email-сообщение."""
    SMTP_SERVER = "smtp.mail.ru"
    SMTP_PORT = 587
    EMAIL_SENDER = "your-email@mail.ru"
    EMAIL_PASSWORD = "your-app-password"
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = recipient
    msg['Subject'] = "Уведомление от HR-бота"
    msg.attach(MIMEText(message, 'plain'))
    
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
