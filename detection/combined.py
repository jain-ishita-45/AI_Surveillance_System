import cv2
import numpy as np
import time
import threading
import pyttsx3
import winsound
from twilio.rest import Client

running = True

# ================= VOICE ENGINE =================
engine = pyttsx3.init()
voice_lock = threading.Lock()



# 🔐 Replace with your credentials
account_sid = "###"
auth_token = "##"
twilio_number = "##"
your_number = "###"

def send_sms_alert():
    try:
        client = Client(account_sid, auth_token)

        message = client.messages.create(
            body="🚨 FIRE ALERT! There is fire detected. Evacuate immediately!",
            from_=twilio_number,
            to=your_number
        )

        print("SMS sent:", message.sid)

    except Exception as e:
        print("SMS failed:", e)

# ================= VOICE FUNCTION =================
def speak_fire_alert():
    try:
        with voice_lock:
            engine.say("There is fire, be alert! Evacuate that place immediately!")
            engine.runAndWait()
    except Exception as e:
        print("Voice error:", e)

# ================= ALERT FUNCTION =================
def alert():
    def run():
        # Run both in parallel safely
        threading.Thread(target=speak_fire_alert, daemon=True).start()

        end_time = time.time() + 5
        while time.time() < end_time:
            winsound.Beep(1500, 500)  # beep every 0.5 sec

    threading.Thread(target=run, daemon=True).start()


def run_detection():
    global running

    # ================= LOAD YOLO =================
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    animal_classes = ["dog", "cat", "bird", "cow", "elephant", "horse", "sheep"]

    # ================= FACE DETECTOR =================
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    fire_announced = False   # ✅ CORRECT PLACE

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # ================= FACE =================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        cv2.putText(frame, f"People: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # ================= FIRE =================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5,150,150]), np.array([15,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        fire_detected = any(cv2.contourArea(c) > 1000 for c in contours)

        if fire_detected:
            cv2.putText(frame, "FIRE DETECTED!", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # blinking alert
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(frame, "🔥 ALERT 🔥", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            if not fire_announced:
                alert() 
                send_sms_alert()  # ✅ CORRECT CALL
                fire_announced = True
        else:
            fire_announced = False

        # ================= ANIMAL DETECTION =================
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]

                if conf > 0.5:
                    label = classes[class_id]
                    if label in animal_classes:
                        cx = int(det[0]*width)
                        cy = int(det[1]*height)
                        w = int(det[2]*width)
                        h = int(det[3]*height)

                        x = int(cx - w/2)
                        y = int(cy - h/2)

                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.putText(frame, label, (x,y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # ================= FOOTER =================
        cv2.putText(frame, "AI Surveillance System",
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("AI Surveillance", frame)

        if cv2.waitKey(1) == 27:
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
