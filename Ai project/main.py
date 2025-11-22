import cv2
import sqlite3
import numpy as np
from datetime import datetime
import time

# --- Database & Recognition Constants ---
DB_NAME = 'attendance_system.db'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
RECOGNITION_THRESHOLD = 0.6  # Lower means stricter match (e.g., for dlib/face_recognition)

def load_user_data():
    """Loads all known users' IDs, Names, and Encodings from the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encoding FROM users")
    data = cursor.fetchall()
    conn.close()

    known_encodings = []
    known_ids = []
    known_names = []

    for user_id, name, encoding_str in data:
        # Convert the stored string back into a NumPy array
        try:
            encoding = np.array([float(val) for val in encoding_str.split(',')])
            known_encodings.append(encoding)
            known_ids.append(user_id)
            known_names.append(name)
        except ValueError:
            print(f"Skipping user {name} due to invalid encoding data.")

    return known_encodings, known_ids, known_names

def log_attendance(user_id):
    """Inserts a record into the attendance_log table."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if user has already logged attendance today
    today_date = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT COUNT(*) FROM attendance_log WHERE user_id=? AND time_in LIKE ?",
        (user_id, f"{today_date}%")
    )
    if cursor.fetchone()[0] > 0:
        print(f"User ID {user_id} already logged attendance today.")
        conn.close()
        return False # Already logged

    cursor.execute(
        "INSERT INTO attendance_log (user_id, time_in) VALUES (?, ?)",
        (user_id, now)
    )
    conn.commit()
    conn.close()
    return True

def run_attendance_system():
    """The main loop for video capture and face recognition."""
    
    # 1. Load Data and Face Cascade
    known_encodings, known_ids, known_names = load_user_data()
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    
    # Check if cascade loaded and data exists
    if face_cascade.empty():
        print("Error: Face cascade classifier not loaded.")
        return
    if not known_encodings:
        print("Warning: No user data loaded. Please add users first.")
        # We can still run, but no one will be recognized
    
    # 2. Start Video Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Attendance System Running... Press 'q' to quit.")
    
    # Throttle recognition to prevent excessive logging
    last_log_time = {} # {user_id: datetime_object}
    LOG_COOLDOWN_SEC = 5 # Prevent logging the same person multiple times in a short window

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale (often faster for simple face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # --- Recognition Logic (Simplified) ---
            # NOTE: For proper recognition, you would use a library like 'face_recognition' 
            # or a deep learning model (e.g., FaceNet) here to get the 128-D encoding
            # and compare it with known_encodings. 
            # Since a full recognition model is too big for this template, 
            # we'll use a DUMMY recognition for demonstration purposes:
            
            recognized_id = None
            recognized_name = "Unknown"

            # DUMMY Recognition: If we have exactly one known encoding, assume it's a match
            if len(known_encodings) == 1:
                recognized_id = known_ids[0]
                recognized_name = known_names[0]
            elif len(known_encodings) > 1:
                # In a real system: Calculate the encoding of the detected face region
                # face_encoding = get_encoding(frame[y:y+h, x:x+w]) 
                # matches = np.linalg.norm(known_encodings - face_encoding, axis=1)
                # best_match_index = np.argmin(matches)
                # if matches[best_match_index] < RECOGNITION_THRESHOLD:
                #     recognized_id = known_ids[best_match_index]
                #     recognized_name = known_names[best_match_index]
                pass # Recognition logic would go here

            # --- Log Attendance & Display ---
            if recognized_id is not None:
                # Check cooldown
                now = datetime.now()
                last_log = last_log_time.get(recognized_id)
                
                if last_log is None or (now - last_log).total_seconds() > LOG_COOLDOWN_SEC:
                    if log_attendance(recognized_id):
                        print(f"Attendance Logged for: {recognized_name} at {now.strftime('%H:%M:%S')}")
                        last_log_time[recognized_id] = now
                    else:
                        print(f"Attendance already logged today for: {recognized_name}")
                        last_log_time[recognized_id] = now # Update time to reset cooldown
                        
                # Update label text
                label = f"Match: {recognized_name}"
                color = (0, 255, 0) # Green
            else:
                label = "Unknown Face"
                color = (0, 0, 255) # Red

            # Display name/status on the frame
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


        # Display the resulting frame
        cv2.imshow('Attendance System', frame)

        # Break the loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Attendance System Shutting Down.")

if __name__ == '__main__':
    # You must run db_setup.py first to create the database and add a user
    print("Ensure 'db_setup.py' was run first.")
    # time.sleep(2) # Wait for users to read the message
    run_attendance_system()