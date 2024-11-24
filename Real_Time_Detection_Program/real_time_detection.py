import cv2
import numpy as np
from tensorflow.keras.models import load_model
from dlib import get_frontal_face_detector, shape_predictor
import pygame

# Load the models
classifier = load_model('best_resnet50_eye_model.keras')
face_detector = get_frontal_face_detector()
landmark_predictor = shape_predictor('shape_predictor_68_face_landmarks.dat')

# Eye landmark indices
LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]

# Drowsiness threshold
DROWSINESS_THRESHOLD = 0.67 # to be adjusted based on model performance

# Initialize sound alert system
pygame.mixer.init()
ALERT_SOUND = 'alert_sound.mp3'

def play_alert():
    """Play an alert sound if drowsiness is detected."""
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.load(ALERT_SOUND)
            pygame.mixer.music.play()
    except pygame.error as e:
        print(f"Error playing alert sound: {e}")

def crop_eye_with_padding(frame, landmarks, eye_indices, padding=20):
    """Crop the eye region with additional padding."""
    x_min = min(landmarks.part(eye_index).x for eye_index in eye_indices)
    x_max = max(landmarks.part(eye_index).x for eye_index in eye_indices)
    y_min = min(landmarks.part(eye_index).y for eye_index in eye_indices)
    y_max = max(landmarks.part(eye_index).y for eye_index in eye_indices)

    height, width = frame.shape[:2]
    x_min = max(0, x_min - padding)
    x_max = min(width, x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(height, y_max + padding)

    # Draw rectangle for visualization on the main frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    cropped = frame[y_min:y_max, x_min:x_max]
    return cropped if cropped.size > 0 else None

def preprocess_eye(eye_region):
    """Preprocess the cropped eye region."""
    if eye_region is None or eye_region.size == 0:
        return None
    try:
        eye_resized = cv2.resize(eye_region, (64, 64))

        # Convert to RGB 
        if len(eye_resized.shape) == 2:
            eye_rgb = cv2.merge([eye_resized, eye_resized, eye_resized])
        else:
            eye_rgb = eye_resized

        normalized_eye = eye_rgb / 255.0  # Normalize pixel values
        return np.expand_dims(normalized_eye, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error in preprocess_eye: {e}")
        return None

def detect_drowsiness(classifier_model, eye_regions, threshold=0.57):
    """Detect drowsiness based on eye region predictions."""
    predictions = []
    for eye_region in eye_regions:
        preprocessed_eye = preprocess_eye(eye_region)
        if preprocessed_eye is None:
            continue
        prediction = classifier_model.predict(preprocessed_eye, verbose=0)[0][0]
        predictions.append(prediction)
    if predictions:
        print(f"Predictions: {predictions}")  # Debug: Print predictions
    return np.mean(predictions) < threshold if predictions else False

def process_frame(frame):
    """Process the frame to detect and evaluate drowsiness."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)

        # Debug: Draw landmarks for eyes
        for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
            cv2.circle(frame, (landmarks.part(idx).x, landmarks.part(idx).y), 2, (0, 255, 0), -1)

        # Crop and visualize eye regions
        left_eye = crop_eye_with_padding(frame, landmarks, LEFT_EYE_INDICES, padding=20)
        right_eye = crop_eye_with_padding(frame, landmarks, RIGHT_EYE_INDICES, padding=20)

        if detect_drowsiness(classifier, [left_eye, right_eye], threshold=DROWSINESS_THRESHOLD):
            return True
    return False

def main():
    """Run real-time drowsiness detection."""
    cap = cv2.VideoCapture(0)
    alert_played = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for drowsiness detection
        drowsy_detected = process_frame(frame)

        # Display status on the frame
        status_text = "Drowsy" if drowsy_detected else "Not Drowsy"
        color = (0, 0, 255) if drowsy_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # play alert if drowsiness detected
        if drowsy_detected:
            if not alert_played:
                play_alert()
                alert_played = True
        else:
            alert_played = False

        # Show the frame with the status overlay
        cv2.imshow("Real-Time Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
