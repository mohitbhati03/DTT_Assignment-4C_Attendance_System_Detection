"""
STEP 1 — Run this ONCE (or when you add new students)
Encodes all student photos and saves to encodings.pkl

Multiple photos per student supported!
Folder structure:
  students/
    Ram.jpg          ← single photo (fine)
    Sita_1.jpg       ← multiple photos: use Name_1.jpg, Name_2.jpg
    Sita_2.jpg
    Sita_3.jpg
"""

import face_recognition
import os
import pickle

known_face_encodings = []
known_face_names     = []

path = "students"

print("=" * 45)
print("   Encoding Student Faces")
print("=" * 45)

for filename in os.listdir(path):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(path, filename)
    image      = face_recognition.load_image_file(image_path)
    encodings  = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        print(f"  ✗ No face found : {filename}")
        continue

    # Strip _1, _2, _3 suffix if present → get clean name
    # e.g. "Sita_2.jpg" → "Sita"  |  "Ram.jpg" → "Ram"
    base = os.path.splitext(filename)[0]          # remove .jpg
    name = base.rsplit("_", 1)[0] if base[-1].isdigit() and "_" in base else base
    name = name.replace("_", " ").title()

    known_face_encodings.append(encodings[0])
    known_face_names.append(name)
    print(f"  ✓ Encoded : {name}  ← {filename}")

# Unique student count
unique = sorted(set(known_face_names))
print(f"\n  Photos processed : {len(known_face_names)}")
print(f"  Unique students  : {len(unique)} → {', '.join(unique)}")

with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)

print("  Saved            : encodings.pkl")
print("=" * 45)
print("\nNow run: python main.py")
