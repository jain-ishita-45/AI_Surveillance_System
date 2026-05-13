import cv2
import numpy as np
import time
import threading
import pyttsx3
import winsound
from twilio.rest import Client

# ================= GLOBAL VARIABLES =================
running = True

# ================= VOICE ENGINE =================
engine = pyttsx3.init()
voice_lock = threading.Lock()

# ================= TWILIO CONFIG =================
# Replace with your actual Twilio credentials
account_sid = "##"
auth_token = "##"

twilio_number = "+##"     # Your Twilio Number
your_number = "+##"     # Your Phone Number


# ================= TWILIO SMS FUNCTION =================
def send_sms_alert(message_text):
    try:
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body=message_text,
            from_=twilio_number,
            to=your_number
        )

        print("SMS Sent:", message.sid)

    except Exception as e:
        print("SMS failed:", e)


# ================= VOICE ALERT =================
def speak_alert(text):
    try:
        with voice_lock:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print("Voice Error:", e)


# ================= FIRE ALERT FUNCTION =================
def fire_alert():
    def run():
        threading.Thread(
            target=speak_alert,
            args=("Fire detected! Evacuate immediately!",),
            daemon=True
        ).start()

        end_time = time.time() + 5

        while time.time() < end_time:
            winsound.Beep(1500, 500)

    threading.Thread(target=run, daemon=True).start()


# ================= MAIN DETECTION FUNCTION =================
def run_detection():
    global running

    # ================= LOAD YOLO =================
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers().flatten()]

    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    animal_classes = [
        "lion", "tiger", "leopard", "cheetah",
        "horse", "bear", "elephant"
    ]

    # ================= FACE DETECTOR =================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # ================= CAMERA =================
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible")
        return

    # ================= ALERT FLAGS =================
    fire_alert_sent = False
    people_alert_sent = False
    animal_alert_sent = False

    print("System Started...")

    while running:

        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape

        # ====================================================
        # FACE DETECTION
        # ====================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        people_count = len(faces)

        cv2.putText(
            frame,
            f"People: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        # ================= PERSON SMS ALERT =================
        if people_count > 0:

            if not people_alert_sent:

                send_sms_alert(
                    f"⚠ PERSON DETECTED!\n"
                    f"Number of people detected: {people_count}"
                )

                threading.Thread(
                    target=speak_alert,
                    args=("Person detected!",),
                    daemon=True
                ).start()

                people_alert_sent = True

        else:
            people_alert_sent = False

        # ====================================================
        # FIRE DETECTION
        # ====================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_fire = np.array([5, 150, 150])
        upper_fire = np.array([15, 255, 255])

        mask = cv2.inRange(hsv, lower_fire, upper_fire)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fire_detected = any(
            cv2.contourArea(c) > 1000
            for c in contours
        )

        if fire_detected:

            cv2.putText(
                frame,
                "FIRE DETECTED!",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Blinking Alert
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(
                    frame,
                    "🔥 ALERT 🔥",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3
                )

            if not fire_alert_sent:

                fire_alert()

                send_sms_alert(
                    "🚨 FIRE ALERT!\n"
                    "Fire detected! Evacuate immediately!"
                )

                fire_alert_sent = True

        else:
            fire_alert_sent = False

        # ====================================================
        # ANIMAL DETECTION
        # ====================================================
        blob = cv2.dnn.blobFromImage(
            frame,
            0.00392,
            (416, 416),
            (0, 0, 0),
            True,
            crop=False
        )

        net.setInput(blob)

        outs = net.forward(output_layers)

        animal_found = False

        for out in outs:

            for det in out:

                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:

                    label = classes[class_id]

                    if label in animal_classes:

                        animal_found = True

                        cx = int(det[0] * width)
                        cy = int(det[1] * height)

                        w = int(det[2] * width)
                        h = int(det[3] * height)

                        x = int(cx - w / 2)
                        y = int(cy - h / 2)

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

        # ================= ANIMAL SMS ALERT =================
        if animal_found:

            cv2.putText(
                frame,
                "ANIMAL DETECTED!",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

            if not animal_alert_sent:

                send_sms_alert(
                    "⚠ ANIMAL DETECTED!\n"
                    "Wildlife/animal detected near surveillance area."
                )

                threading.Thread(
                    target=speak_alert,
                    args=("Animal detected nearby!",),
                    daemon=True
                ).start()

                animal_alert_sent = True

        else:
            animal_alert_sent = False

        # ====================================================
        # FOOTER
        # ====================================================
        cv2.putText(
            frame,
            "AI Surveillance System",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ====================================================
        # SHOW FRAME
        # ====================================================
        cv2.imshow("AI Surveillance", frame)

        # ESC KEY TO EXIT
        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.01)

    # ================= CLEANUP =================
    cap.release()
    cv2.destroyAllWindows()


# ================= RUN PROGRAM =================
if __name__ == "__main__":
    run_detection()
