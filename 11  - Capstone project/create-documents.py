import os
from pathlib import Path

def create_sample_documents():
    """
    Create sample document files for the document retrieval system.
    """
    print("Creating sample document files...")
    
    # Create documents directory
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample documents
    documents = {
        "company_policy.txt": """
# Acme Technologies Company Policy

## Introduction
This document outlines the official policies of Acme Technologies, effective as of January 1, 2023.
All employees are expected to be familiar with and adhere to these policies.

## Work Hours and Attendance
Standard working hours are from 9:00 AM to 5:00 PM, Monday through Friday.
Employees are expected to maintain regular attendance and be punctual.
Flexible work arrangements may be approved by department managers.

## Remote Work Policy
Acme Technologies supports remote work for eligible positions.
Employees working remotely must:
- Maintain regular communication with their team
- Be available during core hours (10:00 AM to 3:00 PM)
- Ensure they have a secure and productive work environment
- Attend virtual team meetings as required

## Information Security
All employees must comply with the company's information security guidelines:
- Use strong, unique passwords for all company accounts
- Enable two-factor authentication where available
- Never share login credentials with anyone
- Report any suspected security incidents immediately
- Keep software and systems updated

## Customer Data Protection
Protection of customer data is of utmost importance:
- Customer information should only be accessed on a need-to-know basis
- Never store customer data on personal devices
- Encrypt all files containing sensitive information
- Comply with all applicable data protection regulations
        """,
        
        "product_manual.txt": """
# Product Manual: Business Analytics Suite

## Product Overview
The Business Analytics Suite is a comprehensive enterprise solution designed to transform raw data into actionable business insights.

## Key Features
1. **Interactive Dashboards**: Customizable visual representations of key performance indicators.
2. **Automated Reporting**: Schedule and generate reports automatically at defined intervals.
3. **Data Integration**: Connect to multiple data sources including databases, spreadsheets, and cloud services.
4. **Predictive Analytics**: Leverage machine learning algorithms to forecast business trends.
5. **Mobile Compatibility**: Access insights on-the-go with our mobile application.

## System Requirements
- Operating System: Windows 10+, macOS 10.14+, or Linux
- Processor: Intel Core i5 or equivalent (i7 recommended for large datasets)
- Memory: 8GB RAM minimum (16GB recommended)
- Storage: 500MB for application, additional space needed for data
- Database: Compatible with MySQL, PostgreSQL, SQL Server, Oracle

## Installation Guide
1. Download the installer from your account portal
2. Run the installer and follow the on-screen instructions
3. Enter your license key when prompted
4. Configure database connections
5. Complete the setup wizard

## Troubleshooting
Common issues and their solutions:
- Connection errors: Check network settings and database credentials
- Slow performance: Consider upgrading hardware or optimizing queries
- Error code 5001: Restart the application and check the log files

## Support
For technical assistance, contact our support team:
- Email: support@acmetech.com
- Phone: 1-800-555-0123
- Hours: Monday-Friday, 8:00 AM - 6:00 PM EST
        """,
        
        "customer_service_guide.txt": """
# Customer Service Guidelines

## Customer Service Philosophy
At Acme Technologies, we believe exceptional customer service is the foundation of our business.
Every customer interaction is an opportunity to demonstrate our commitment to excellence.

## Communication Standards
When communicating with customers:
- Respond to all inquiries within 24 hours
- Use clear, professional language
- Address customers by name
- Listen actively and empathetically
- Avoid technical jargon unless appropriate for the customer

## Issue Resolution Process
1. **Acknowledge**: Confirm the issue and thank the customer for bringing it to our attention
2. **Understand**: Ask clarifying questions to fully understand the problem
3. **Solve**: Provide a clear solution or action plan
4. **Follow-up**: Check back with the customer to ensure satisfaction

## Escalation Procedures
Issues should be escalated when:
- The issue is beyond your authority to resolve
- The customer specifically requests escalation
- The issue is complex and requires specialized knowledge

Escalation path:
- Level 1: Customer Service Representative
- Level 2: Customer Service Team Lead
- Level 3: Customer Service Manager
- Level 4: Department Director

## Customer Feedback
Actively seek customer feedback through:
- Post-interaction surveys
- Regular check-ins with long-term customers
- Annual customer satisfaction surveys
- Product and service reviews

## Difficult Customer Situations
When dealing with upset customers:
- Remain calm and professional
- Never take comments personally
- Focus on solutions rather than problems
- Know when to involve a supervisor
- Document the interaction thoroughly
        """,
        
        "sales_strategies.txt": """
# Sales Strategy Document

## Target Market Segments
Our primary market segments include:
1. **Enterprise Corporations**: 1000+ employees, focus on comprehensive solutions
2. **Mid-Market Companies**: 100-999 employees, focus on scalability
3. **Small Businesses**: Under 100 employees, focus on cost-effectiveness

## Product Positioning
- Business Analytics Suite: Premium enterprise solution with advanced capabilities
- Cloud Storage Pro: Essential service with competitive pricing
- Smart Office Assistant: Productivity enhancement tool with AI capabilities

## Sales Process
1. **Prospecting**: Identify qualified potential customers
2. **Initial Contact**: Introduce company and solutions
3. **Needs Assessment**: Understand customer requirements
4. **Solution Presentation**: Demonstrate relevant products
5. **Handling Objections**: Address concerns and questions
6. **Closing**: Secure commitment
7. **Follow-up**: Ensure customer satisfaction and identify expansion opportunities

## Pricing Strategy
Base pricing structure with the following considerations:
- Volume discounts for multi-user licenses
- Annual subscription discounts (15% compared to monthly)
- Enterprise customization packages available
- Educational and non-profit pricing available (25% discount)

## Competitive Differentiation
Key differentiators to emphasize:
- Superior data integration capabilities
- Intuitive user experience
- Comprehensive customer support
- Regular feature updates and enhancements
- Industry-specific solutions

## Quarterly Sales Targets
Q1: $1,250,000
Q2: $1,500,000
Q3: $1,750,000
Q4: $2,000,000

## Sales Team Resources
- Product demo environments
- Case studies and testimonials
- ROI calculators
- Competitor comparison sheets
- Proposal templates
        """
    }
    
    # Write sample documents to files
    for filename, content in documents.items():
        with open(docs_dir / filename, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(documents)} sample documents in {docs_dir}")
    for filename in documents.keys():
        print(f"- {filename}")

if __name__ == "__main__":
    create_sample_documents()
