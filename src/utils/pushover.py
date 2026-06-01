import requests

ANDREAS_USER = "usozo22dr29waq7yx9f5u3ab74717x"
ANDREAS_TOKEN = "aqy5q6psoain23br3bci8dxeund4ne"

CASPER_USER = "uvp7eyihevg1bvxqf1of8qdomrf4xr"
CASPER_TOKEN = "a9jdszjfcgv9f1ut6nqynqir1hx8qq"

def send_notification(title: str, message: str, notify_andreas: bool = True, notify_casper: bool = False, priority: int = 0):

    try:
        if notify_andreas:
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
        if notify_casper:
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
    send_notification("Test", "Det her er en test.",send_to_casper=False)