
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You'll need to download this file

# Load pre-trained acne detection model
acne_model = load_model(r"C:\Users\USER\model.h5")  # Replace with your acne detection model file path

# Function to preprocess the input face image for acne detection
def preprocess_acne(face_image):
    input_shape = (224, 224)  # Example input size expected by the model
    resized_image = cv2.resize(face_image, input_shape)
    normalized_image = resized_image.astype(np.float32) / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Function to detect acne on a face
def detect_acne(face_image):
    preprocessed_image = preprocess_acne(face_image)
    acne_probability = acne_model.predict(preprocessed_image)
    acne_detected = acne_probability > 0.5  # Adjust threshold as needed
    return acne_detected.any()

# Function to detect pigmentation in facial images
def detect_pigmentation(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pigmented_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust minimum area threshold as needed
            pigmented_areas.append(contour)
    return pigmented_areas

# Function to detect wrinkles in facial images
def detect_wrinkles(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edges = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wrinkle_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]  # Adjust minimum area threshold as needed
    return wrinkle_contours

# Main function to process the webcam feed
def process_webcam_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # Check for acne
            acne_detected = detect_acne(frame)

            # Check for pigmentation
            pigmented_areas = detect_pigmentation(frame)

            # Check for wrinkles
            wrinkle_contours = detect_wrinkles(frame)

            # Draw landmarks on the face
            for landmark in landmarks.parts():
                cv2.circle(frame, (landmark.x, landmark.y), 1, (0, 255, 0), -1)

            # Draw acne label if detected
            if acne_detected:
                cv2.putText(frame, "Acne", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw pigmented areas
            cv2.drawContours(frame, pigmented_areas, -1, (0, 0, 255), 2)

            # Draw wrinkles
            cv2.drawContours(frame, wrinkle_contours, -1, (255, 0, 0), 2)

        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the main function to start processing the webcam feed
process_webcam_feed()
