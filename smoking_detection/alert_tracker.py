from collections import deque
import time

class SmokingAlertTracker:
    """Temporal sliding-window smoothing and cooldown-based alert counter."""
    def __init__(self, window_size=5, alert_cooldown_sec=2.5, threshold=0.30):
        self.window = deque(maxlen=int(window_size))
        self.last_alert_time = 0.0
        self.alert_cooldown = float(alert_cooldown_sec)
        self.threshold = float(threshold)
        self.smoking_occurrences = 0

    def update(self, smoking_count):
        smoking_count = int(smoking_count)
        self.window.append(smoking_count)
        average = sum(self.window) / max(1, len(self.window))
        status = "WARNING" if average >= self.threshold or smoking_count > 0 else "SAFE"
        now = time.time()
        new_alert = False
        if status == "WARNING" and now - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = now
            self.smoking_occurrences += 1
            new_alert = True
        return {
            "status": status,
            "average_smoking": average,
            "new_alert": new_alert,
            "smoking_occurrences": self.smoking_occurrences,
        }
