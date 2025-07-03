#!/usr/bin/env python3
"""
Mock API Server for Testing

This script creates a simple mock API server that can be used for testing
the Business Intelligence Agent. It simulates external API endpoints and
returns predefined responses.

Usage:
  python mock-api.py

The server will start on http://localhost:5000 by default.
"""

from flask import Flask, request, jsonify
import time
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock-api")

# Create Flask app
app = Flask(__name__)

# Store created tickets and other data
tickets = {}
ticket_id_counter = 1000
crm_entries = {}
crm_id_counter = 500

@app.route("/create_ticket", methods=["POST"])
def create_ticket():
    """Endpoint to create a mock support ticket"""
    global ticket_id_counter
    
    # Log the request
    logger.info(f"Received create_ticket request: {request.json}")
    
    # Extract data
    data = request.json or {}
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Create ticket ID
    ticket_id = f"TICKET-{ticket_id_counter}"
    ticket_id_counter += 1
    
    # Store ticket
    tickets[ticket_id] = {
        "id": ticket_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "open",
        **data
    }
    
    # Return response
    return jsonify({
        "status": "success",
        "message": "Ticket created successfully",
        "ticket_id": ticket_id
    })

@app.route("/update_crm", methods=["POST"])
def update_crm():
    """Endpoint to update a mock CRM record"""
    global crm_id_counter
    
    # Log the request
    logger.info(f"Received update_crm request: {request.json}")
    
    # Extract data
    data = request.json or {}
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Create CRM ID if not provided
    if "id" not in data:
        crm_id = f"CRM-{crm_id_counter}"
        crm_id_counter += 1
    else:
        crm_id = data["id"]
    
    # Store CRM entry
    crm_entries[crm_id] = {
        "id": crm_id,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **data
    }
    
    # Return response
    return jsonify({
        "status": "success",
        "message": "CRM updated successfully",
        "crm_id": crm_id
    })

@app.route("/generate_report", methods=["POST"])
def generate_report():
    """Endpoint to generate a mock report"""
    # Log the request
    logger.info(f"Received generate_report request: {request.json}")
    
    # Extract data
    data = request.json or {}
    
    # Simulate processing time
    time.sleep(1.0)
    
    # Generate mock report data
    report_data = {
        "report_id": f"REPORT-{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_records": 42,
            "processed": 42,
            "errors": 0
        },
        "details": {
            "categories": ["Sales", "Marketing", "Support"],
            "metrics": {
                "Sales": 12500.0,
                "Marketing": 5600.0,
                "Support": 8200.0
            }
        }
    }
    
    # Return response
    return jsonify({
        "status": "success",
        "message": "Report generated successfully",
        "report": report_data
    })

@app.route("/send_notification", methods=["POST"])
def send_notification():
    """Endpoint to send a mock notification"""
    # Log the request
    logger.info(f"Received send_notification request: {request.json}")
    
    # Extract data
    data = request.json or {}
    
    # Simulate processing time
    time.sleep(0.3)
    
    # Check required fields
    if "recipient" not in data:
        return jsonify({
            "status": "error",
            "message": "Recipient is required"
        }), 400
    
    if "message" not in data:
        return jsonify({
            "status": "error",
            "message": "Message is required"
        }), 400
    
    # Return response
    return jsonify({
        "status": "success",
        "message": "Notification sent successfully",
        "notification_id": f"NOTIF-{int(time.time())}"
    })

@app.route("/status", methods=["GET", "POST"])
def status():
    """Endpoint to check API status"""
    logger.info(f"Status endpoint called with method: {request.method}")
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "tickets_created": len(tickets),
        "crm_updates": len(crm_entries)
    })

def main():
    """Main function to start the API server"""
    parser = argparse.ArgumentParser(description="Mock API Server for Testing")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting mock API server on http://{args.host}:{args.port}")
    logger.info("Available endpoints:")
    logger.info("  POST /create_ticket")
    logger.info("  POST /update_crm")
    logger.info("  POST /generate_report")
    logger.info("  POST /send_notification")
    logger.info("  GET  /status")
    
    app.run(host="0.0.0.0", port=args.port, debug=True)

if __name__ == "__main__":
    main()
