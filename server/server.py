from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from squat_counter import SquatCounter  
from pushup_counter import PushupCounter  # Import the new class
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

socketio = SocketIO(app, cors_allowed_origins="*")

# Global references to tracking instances
squat_counter = None
pushup_counter = None
squat_thread = None
pushup_thread = None

@app.route("/start_squat", methods=["GET"])
def start_squat():
    """Starts real-time squat tracking."""
    global squat_counter, squat_thread
    
    if squat_counter:
        squat_counter.stop()
        if squat_thread and squat_thread.is_alive():
            squat_thread.join(timeout=2.0)  
    
    squat_counter = SquatCounter(socketio)
    squat_thread = squat_counter.start()
    
    return jsonify({"message": "Squat tracking started", "status": "running"})

@app.route("/stop_squat", methods=["GET"])
def stop_squat():
    """Stops real-time squat tracking."""
    global squat_counter, squat_thread
    
    if squat_counter:
        squat_counter.stop()
        if squat_thread and squat_thread.is_alive():
            squat_thread.join(timeout=2.0)  
        squat_counter = None
        return jsonify({"message": "Squat tracking stopped", "status": "stopped"})
    
    return jsonify({"message": "No active squat tracking", "status": "inactive"})

@app.route("/get_squat_count", methods=["GET"])
def get_squat_count():
    """Gets current squat count."""
    if squat_counter:
        return jsonify({"count": squat_counter.counter, "status": "active"})
    return jsonify({"count": 0, "status": "inactive"})

# Pushup tracking APIs
@app.route("/start_pushup", methods=["GET"])
def start_pushup():
    """Starts real-time pushup tracking."""
    global pushup_counter, pushup_thread
    
    if pushup_counter:
        pushup_counter.stop()
        if pushup_thread and pushup_thread.is_alive():
            pushup_thread.join(timeout=2.0)
    
    pushup_counter = PushupCounter(socketio)
    pushup_thread = pushup_counter.start()
    
    return jsonify({"message": "Pushup tracking started", "status": "running"})

@app.route("/stop_pushup", methods=["GET"])
def stop_pushup():
    """Stops real-time pushup tracking."""
    global pushup_counter, pushup_thread
    
    if pushup_counter:
        pushup_counter.stop()
        if pushup_thread and pushup_thread.is_alive():
            pushup_thread.join(timeout=2.0)
        pushup_counter = None
        return jsonify({"message": "Pushup tracking stopped", "status": "stopped"})
    
    return jsonify({"message": "No active pushup tracking", "status": "inactive"})

@app.route("/get_pushup_count", methods=["GET"])
def get_pushup_count():
    """Gets current pushup count."""
    if pushup_counter:
        return jsonify({"count": pushup_counter.counter, "status": "active"})
    return jsonify({"count": 0, "status": "inactive"})

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    try:
        socketio.run(app, debug=True, port=3001, host="0.0.0.0", allow_unsafe_werkzeug=True)
    finally:
        if squat_counter:
            squat_counter.stop()
            if squat_thread and squat_thread.is_alive():
                squat_thread.join(timeout=2.0)
        if pushup_counter:
            pushup_counter.stop()
            if pushup_thread and pushup_thread.is_alive():
                pushup_thread.join(timeout=2.0)


