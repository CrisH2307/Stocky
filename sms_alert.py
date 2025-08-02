import os
from twilio.rest import Client
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

def send_sms(to_number: str, message_body: str) -> bool:
    """
    Sends an SMS message using Twilio.

    Args:
        to_number (str): The recipient's phone number in E.164 format (e.g., +1234567890).
        message_body (str): The content of the message to send.

    Returns:
        bool: True if the message was sent successfully, False otherwise.
    """
    # Check if Twilio credentials are configured
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        logger.error("Twilio credentials are not fully configured. Please check your .env file.")
        return False

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        
        logger.info(f"SMS sent successfully to {to_number}. SID: {message.sid}")
        return True
    except Exception as e:
        logger.error(f"Failed to send SMS to {to_number}: {e}")
        return False

# Example usage (for testing this script directly)
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual phone number to test.
    # Your number must be in E.164 format.
    test_recipient_number = os.getenv("RECIPIENT_PHONE_NUMBER", "+15558675309") 
    
    print("Sending a test SMS...")
    success = send_sms(
        to_number=test_recipient_number,
        message_body="📈 STOCK ALERT: This is a test message from your Stock Volatility Dashboard."
    )
    
    if success:
        print("✅ Test SMS sent successfully!")
    else:
        print("❌ Failed to send test SMS. Check your terminal for error logs and verify your .env settings.")