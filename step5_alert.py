def send_recovery_alert(phone_number, item_name, timestamp):
    # For the Hackathon MVP, we print this to show it works.
    # In a real build, you'd use the Twilio API here.
    message = f"ALERT: Your {item_name} was spotted in surveillance at {timestamp}s. Verification clip is ready."
    print(f"Sending SMS to {phone_number}: {message}")
    return True