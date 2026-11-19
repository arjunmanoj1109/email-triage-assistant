import os
import json
import sqlite3
from datetime import datetime
from typing import Literal
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 1. Load API key and initialize Gemini client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-3.5-flash-lite"


# --- CORE MECHANIC 1: STRICT PYDANTIC SCHEMA ---
class EmailTriage(BaseModel):
    category: Literal["Urgent Support", "Billing", "Spam", "General Inquiry"]
    urgency_score: int = Field(
        ge=1, le=5, description="Urgency from 1 (lowest/spam) to 5 (critical outage/churn risk)"
    )
    key_intent: str = Field(description="One-sentence summary of what the sender wants.")
    suggested_draft: str = Field(description="Polite, context-aware reply draft (or empty if Spam).")
    action_required: bool = Field(description="True if a human needs to reply or take action.")


# --- CORE MECHANIC 2: FEW-SHOT CONDITIONING ---
SYSTEM_PROMPT = """You are an automated customer operations triage engine.
Analyze incoming emails or messages and extract structured triage metadata.

Follow these calibration examples for edge cases:

Example 1 (High Urgency Outage):
Message: "Our entire production checkout has been throwing 500 errors for 20 minutes! We are losing thousands of dollars every minute. Fix this ASAP or we're canceling our enterprise contract."
Output:
{
  "category": "Urgent Support",
  "urgency_score": 5,
  "key_intent": "Customer is experiencing a critical production checkout outage and threatening contract cancellation.",
  "suggested_draft": "Hi there, I am escalating your checkout outage to our on-call engineering lead right now. We are investigating the 500 errors as our top priority and will update you within 15 minutes.",
  "action_required": true
}

Example 2 (Subtle Spam / Cold Outreach):
Message: "Hi! Quick question about your Q4 growth targets—I noticed your domain authority could be boosted 300% using our proprietary backlink network. Free 15-min audit?"
Output:
{
  "category": "Spam",
  "urgency_score": 1,
  "key_intent": "Unsolicited SEO cold outreach disguised as a quick question.",
  "suggested_draft": "",
  "action_required": false
}

Example 3 (Routine Billing Inquiry):
Message: "Hello, could you please send over a PDF copy of our March invoice with our updated VAT number attached? Thanks!"
Output:
{
  "category": "Billing",
  "urgency_score": 2,
  "key_intent": "Customer needs a March invoice re-issued with their updated VAT number.",
  "suggested_draft": "Hello! I'd be happy to help with that. Could you confirm the VAT number you would like listed on the March invoice so I can generate the updated PDF for you?",
  "action_required": true
}
"""


# --- DATABASE SETUP FOR ASYNCHRONOUS HUMAN REVIEW ---
def init_db():
    """Creates a local SQLite table to hold low/medium urgency drafts for human review."""
    conn = sqlite3.connect("review_queue.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triage_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT,
            raw_message TEXT,
            category TEXT,
            urgency_score INTEGER,
            key_intent TEXT,
            suggested_draft TEXT,
            action_required BOOLEAN,
            status TEXT
        )
    """)
    conn.commit()
    return conn


# --- EXTRACTION ENGINE (CONSTRAINED DECODING) ---
def analyze_message(raw_text: str) -> EmailTriage:
    """Uses Gemini Structured Outputs to guarantee a valid EmailTriage object."""
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=f"Analyze and triage this incoming message:\n\n{raw_text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=EmailTriage,
            temperature=0.1,  # Low temperature for consistent classification
        ),
    )
    # response.parsed automatically returns an instantiated EmailTriage Pydantic object
    if response.parsed:
        return response.parsed
    return EmailTriage.model_validate_json(response.text)


# --- CORE MECHANIC 3: DOWNSTREAM ROUTING ---
def route_triage_result(conn, raw_text: str, result: EmailTriage):
    """Routes high-urgency items to priority alerts and queues others for human review."""
    print("\n--- Structured Extraction Result (Valid JSON) ---")
    print(result.model_dump_json(indent=2))

    if result.urgency_score >= 4:
        print("\n[ROUTING -> PRIORITY CHANNEL] Urgency >= 4 detected!")
        alert_msg = (
            f"🚨 **PRIORITY TRIAGE ALERT (Score: {result.urgency_score}/5)**\n"
            f"**Category:** {result.category}\n"
            f"**Intent:** {result.key_intent}\n"
            f"**Suggested Draft:** {result.suggested_draft}"
        )
        print(alert_msg)

        # Optional: Push to Slack/Discord if PRIORITY_WEBHOOK_URL is set in .env
        webhook_url = os.getenv("PRIORITY_WEBHOOK_URL")
        if webhook_url:
            requests.post(webhook_url, json={"content": alert_msg})
            print("-> Dispatched live webhook notification!")
    else:
        status = "Pending Human Review" if result.action_required else "Auto-Archived (No Action)"
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO triage_queue 
            (received_at, raw_message, category, urgency_score, key_intent, suggested_draft, action_required, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(),
                raw_text,
                result.category,
                result.urgency_score,
                result.key_intent,
                result.suggested_draft,
                result.action_required,
                status,
            ),
        )
        conn.commit()
        print(f"\n[ROUTING -> ASYNC QUEUE] Saved to review_queue.db with status: '{status}'")


# --- INTERACTIVE TEST RUNNER ---
if __name__ == "__main__":
    conn = init_db()

    # Sample batch of realistic incoming messages to test all routing branches
    sample_messages = [
        "Hey team, my credit card was charged twice for my Pro subscription this morning! Can you refund the duplicate charge?",
        "URGENT: Nobody on our team can log into the dashboard—SSO authentication is completely broken right before our client demo!",
        "Win a free iPad Pro! Click here to claim your exclusive corporate reward today.",
        "Hi, do you offer annual discounts for non-profit organizations looking to buy 15 seats?",
    ]

    print("=== Running Automated Batch Triage ===")
    for i, msg in enumerate(sample_messages, 1):
        print(f"\n{'='*60}\nIncoming Message #{i}:\n\"{msg}\"")
        triage_data = analyze_message(msg)
        route_triage_result(conn, msg, triage_data)

    # Interactive mode so you can test your own custom emails/messages
    print(f"\n{'='*60}\n=== Interactive Triage Mode ===")
    print("Paste any email or customer message below (or type 'exit' to quit):")

    while True:
        try:
            user_msg = input("\nNew Message > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ["exit", "quit", "q"]:
                break
            triage_data = analyze_message(user_msg)
            route_triage_result(conn, user_msg, triage_data)
        except KeyboardInterrupt:
            break

    conn.close()
    print("\nTriage assistant closed.")