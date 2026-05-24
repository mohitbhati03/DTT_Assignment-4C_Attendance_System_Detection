import cv2
import face_recognition
import numpy as np
import pickle
import csv
import os
import threading
from datetime import datetime

# ── CONFIG ───────────────────────────────────────────────
ATTENDANCE_FILE    = "attendance.csv"
SCREENSHOTS_FOLDER = "screenshots"
TOLERANCE          = 0.5    # lower = stricter match
FRAME_SCALE        = 0.5    # shrink for Haar (speed)
# ─────────────────────────────────────────────────────────

# ── Load Haar Cascade ─────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Load saved encodings ──────────────────────────────────
if not os.path.exists("encodings.pkl"):
    print("❌ encodings.pkl not found. Run encode_faces.py first!")
    exit()

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names     = data["names"]
unique_students = sorted(set(known_names))

print(f"✅ Loaded {len(unique_students)} student(s): {', '.join(unique_students)}")
print(f"   ({len(known_names)} total encoding(s))\n")

# ── Folders ───────────────────────────────────────────────
os.makedirs(SCREENSHOTS_FOLDER, exist_ok=True)

def get_screenshot_folder():
    folder = os.path.join(SCREENSHOTS_FOLDER, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)
    return folder

# ── CSV ───────────────────────────────────────────────────
def init_csv():
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time", "Status", "Confidence"])
        print(f"📄 Created: {ATTENDANCE_FILE}")

def already_marked(name, date):
    try:
        with open(ATTENDANCE_FILE, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row["Name"] == name and row["Date"] == date:
                    return True
    except FileNotFoundError:
        pass
    return False

def mark_attendance(name, date, time_str, status, confidence=""):
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, date, time_str, status, confidence])

def mark_absent_remaining(present_set, date):
    count = 0
    for name in unique_students:
        if name not in present_set and not already_marked(name, date):
            mark_attendance(name, date, "--:--:--", "Absent", "")
            count += 1
    if count:
        print(f"\n📋 Marked {count} student(s) as Absent.")

# ── Shared state (main ↔ recognition thread) ──────────────
lock             = threading.Lock()
current_results  = []      # list of (x1,y1,w1,h1, name, label, color)
processing       = False   # is recognition thread busy?
latest_frame     = None    # frame to process next

def recognition_worker(frame_to_process, today, present_today):
    """Runs in background thread — heavy face_recognition work here."""
    global processing, current_results

    small = cv2.resize(frame_to_process, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    scale = int(1 / FRAME_SCALE)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    results = []
    for (x, y, w, h) in faces:
        face_encs = face_recognition.face_encodings(rgb, [(y, x+w, y+h, x)])

        name  = "Unknown"
        color = (0, 0, 255)
        label = "Unknown"

        if face_encs:
            distances = face_recognition.face_distance(known_encodings, face_encs[0])
            best_idx  = np.argmin(distances)
            best_dist = distances[best_idx]
            confidence = round((1 - best_dist) * 100, 1)

            if best_dist <= TOLERANCE:
                name  = known_names[best_idx]
                color = (0, 200, 0)
                label = f"{name}  {confidence}%"

                now = datetime.now()
                with lock:
                    if name not in present_today and not already_marked(name, today):
                        mark_attendance(name, today, now.strftime("%H:%M:%S"),
                                        "Present", f"{confidence}%")
                        present_today.add(name)
                        print(f"  ✅ Present: {name} | {confidence}% | {now.strftime('%H:%M:%S')}")

                        # Auto-screenshot
                        safe    = name.replace(" ", "_")
                        ts      = now.strftime("%H%M%S")
                        ss_path = os.path.join(get_screenshot_folder(), f"{safe}_{ts}.png")
                        cv2.imwrite(ss_path, frame_to_process)
                        print(f"  📸 Screenshot: {ss_path}")
            else:
                label = f"Unknown  {confidence}%"

        x1, y1, w1, h1 = x*scale, y*scale, w*scale, h*scale
        results.append((x1, y1, w1, h1, name, label, color))

    with lock:
        current_results[:] = results
        processing = False

# ── Init ──────────────────────────────────────────────────
init_csv()
today         = datetime.now().strftime("%Y-%m-%d")
present_today = set()
frame_count   = 0

# ── Camera ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

# Reduce buffer → less lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("📷 Camera started. Smooth video, background recognition.")
print("   Press Q → quit & save attendance")
print("   Press S → manual screenshot\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ── Launch recognition thread every 15 frames if free ─
    with lock:
        busy = processing

    if not busy and frame_count % 15 == 0:
        with lock:
            processing = True
        t = threading.Thread(
            target=recognition_worker,
            args=(frame.copy(), today, present_today),
            daemon=True
        )
        t.start()

    # ── Draw last known results on EVERY frame (smooth) ──
    with lock:
        results_copy = list(current_results)

    for (x1, y1, w1, h1, name, label, color) in results_copy:
        cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), color, 2)
        cv2.rectangle(frame, (x1, y1-32), (x1+w1, y1), color, cv2.FILLED)
        cv2.putText(frame, label, (x1+5, y1-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1)

    # ── HUD ───────────────────────────────────────────────
    with lock:
        p_count = len(present_today)
    hud = f"Present: {p_count}/{len(unique_students)}  |  Q=Quit  S=Screenshot"
    cv2.rectangle(frame, (0, 0), (480, 32), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, hud, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        ts   = datetime.now().strftime("%H%M%S")
        path = os.path.join(get_screenshot_folder(), f"manual_{ts}.png")
        cv2.imwrite(path, frame)
        print(f"  📸 Manual screenshot: {path}")

# ── Session end ───────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()

mark_absent_remaining(present_today, today)

print(f"\n{'='*40}")
print(f"  Session Complete")
with lock:
    print(f"  Present : {sorted(present_today)}")
    absent = [n for n in unique_students if n not in present_today]
print(f"  Absent  : {sorted(absent)}")
print(f"  CSV     : {ATTENDANCE_FILE}")
print(f"  Photos  : {SCREENSHOTS_FOLDER}/")
print(f"{'='*40}")
