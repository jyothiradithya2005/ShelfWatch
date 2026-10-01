from datetime import datetime

from twilio.rest import Client

from config import reorder_link
from detector import display


class SmsAlerter:
    def __init__(self, sid, token, from_number, to_number, messaging_service_sid=None):
        ready = all([sid, token, to_number]) and bool(messaging_service_sid or from_number)
        self.client = Client(sid, token) if ready else None
        self.sender = {"messaging_service_sid": messaging_service_sid} if messaging_service_sid else {"from_": from_number}
        self.to_number = to_number

    @property
    def enabled(self):
        return self.client is not None

    def send(self, body):
        message = self.client.messages.create(body=body, to=self.to_number, **self.sender)
        return message.sid


class StockMonitor:
    def __init__(self):
        self.alerted = set()
        self.log = []

    def evaluate(self, counts, tracked, threshold):
        rows = []
        for item in tracked:
            count = counts.get(item, 0)
            status = "Out of stock" if count == 0 else "Low" if count <= threshold else "In stock"
            rows.append({"Item": display(item), "Count": count, "Status": status, "Reorder": reorder_link(item)})
        return rows

    def alert(self, counts, tracked, threshold, alerter, source):
        for item in tracked:
            count = counts.get(item, 0)
            if count > threshold:
                self.alerted.discard(item)
                continue
            if item in self.alerted:
                continue
            self.alerted.add(item)
            body = f"Low stock: {display(item)} at {count} units. Reorder: {reorder_link(item)}"
            try:
                result = f"Sent ({alerter.send(body)})" if alerter.enabled else "Skipped (Twilio not configured)"
            except Exception as error:
                result = f"Failed: {error}"
            self.log.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Item": display(item),
                "Count": count,
                "Source": source,
                "Result": result,
            })

    def reset(self):
        self.alerted.clear()
