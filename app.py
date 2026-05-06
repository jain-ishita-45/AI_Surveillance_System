from flask import Flask, render_template, jsonify
import threading

# Import your combined detection module
from detection import combined

app = Flask(__name__)

# Thread reference
detection_thread = None


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/start')
def start():
    global detection_thread

    # Prevent multiple threads
    if detection_thread and detection_thread.is_alive():
        return jsonify({"status": "⚠️ System already running"})

    # Start detection
    combined.running = True
    detection_thread = threading.Thread(target=combined.run_detection)
    detection_thread.start()

    return jsonify({"status": "✅ Monitoring Started"})


@app.route('/stop')
def stop():
    global detection_thread

    combined.running = False
    return jsonify({"status": "🛑 Monitoring Stopped"})


if __name__ == "__main__":
    app.run(debug=True)