import requests

ANDREAS_USER = "usozo22dr29waq7yx9f5u3ab74717x"
ANDREAS_TOKEN = "at2di7dvcc9gsds3f73cfo6rbcw77r"

CASPER_USER = "uvp7eyihevg1bvxqf1of8qdomrf4xr"
CASPER_TOKEN = "a9jdszjfcgv9f1ut6nqynqir1hx8qq"

def send_notification(title: str, message: str, send_to_andreas: bool = True, send_to_casper: bool = False, priority: int = 0):

    try:
        if send_to_andreas:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": ANDREAS_TOKEN,
                    "user": ANDREAS_USER,
                    "title": title,
                    "message": message,
                    "priority": priority,
                },
                timeout=5,
            )
        if send_to_casper:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": CASPER_TOKEN,
                    "user": CASPER_USER,
                    "title": title,
                    "message": message,
                    "priority": priority,
                },
                timeout=5,
            )
        print("Pushover notification sent successfully.")
    except Exception as e:
        print(f"Failed to send Pushover notification: {e}")
        
if __name__ == "__main__":
    send_notification("Test", "Det her er en test.")