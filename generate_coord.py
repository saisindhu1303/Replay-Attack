import cv2
import os

# Output directory and file names
OUTPUT_DIR = "captured_images"
IMAGE_OUT = os.path.join(OUTPUT_DIR, "face.jpg")
COORD_OUT = os.path.join(OUTPUT_DIR, "coord.txt")

# Load Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_and_save(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected.")
        return False

    # Save the image
    cv2.imwrite(IMAGE_OUT, frame)
    print(f"📸 Saved image as {IMAGE_OUT}")

    # Save coordinates
    with open(COORD_OUT, "w") as f:
        f.write("X,Y,W,H\n")
        for (x, y, w, h) in faces:
            f.write(f"{x},{y},{w},{h}\n")
    print(f"📂 Saved coordinates to {COORD_OUT}")
    return True

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📷 Press 'c' to capture, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Face - Press 'c' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if detect_and_save(frame):
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
