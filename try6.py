import pickle
from pathlib import Path
import streamlit as st
import sys
import requests
import streamlit as st
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer
from transformers import BlenderbotForConditionalGeneration
import cv2
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, Image


from streamlit_lottie import st_lottie

import openai
import gradio


import os



import streamlit_authenticator as stauth

#openai.api_key = "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"
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_coding=load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_H3shI6.json")
l_c=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_EHugAD.json")

st.set_page_config(page_title="GuideMySkinBot",layout="wide")
with st.container():
    st.subheader("GuideMySkinBot helps you to find the correct skincare routine as per your skin concern.")
    st.title(" GuideMySkinBot")
    st.write(" Guide My skinbot is all about Delivering skincare routine for all ages and also including from different skin types,Cost effective andreliable solutions.")
    st.write("Also provides you with trust-worthy product recommendations")
with st.container():
    st.write("---")
    
    st.header("Services Provided")
    st.write("##")
    st.write("""1.REGULAR SKINCARE ROUTINE\n
                    Get to know the basic skin routine.\n2.CUSTOMIZED SKINCARE ROUTINE \n
                            (i).Suggests skincare routine according to user concern\n\t(ii).There is product suggestion for each skin concern.
                    
                    """)
with st.container():
    st.write("---")
    
    
    st.header("Regular Skincare Routine Suggestions")
    image_column, text_column = st.columns((2, 1))
    with image_column:
    
        st.write("##")
        st.write("[NormalSkin >](https://www.healthline.com/health/beauty-skincare/the-ultimate-skin-care-routine-for-normal-skin)")
        st.write("[Oily Skin>](https://www.healthline.com/health/beauty-skin-care/skin-care-routine-for-oily-skin)")
        st.write("[Dry skin>](https://www.healthline.com/health/beauty-skin-care/skin-care-routine-for-dry-skin)")
        st.write("[Acne-Prone Skin>](https://www.healthline.com/health/beauty-skin-care/acne-prone-skin)")
        st.write("[Combination skin >](https://www.healthline.com/health/beauty-skin-care/skin-care-routine-for-combination-skin#routine)")
        st.write("[Video recommended >](https://youtu.be/vpB0u6zze-0)")
    with text_column:
        st_lottie(lottie_coding,height=300,key="skincare") 

    
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")
with st.container():
    st.write("---")
    st.header("To get Customized Skincare routine!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/agnzreenawin6@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Send your skin concern" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
            
            
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("Customized Skincare Routine")
        st.write("##")
        st_lottie(l_c,height=300,key="skin") 

        
    with right_column:
        @st.cache_resource
        def get_models():
            # it may be necessary for other frameworks to cache the model
            # seems pytorch keeps an internal state of the conversation
            model_name = "facebook/blenderbot-400M-distill"
            tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
            model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
            return tokenizer, model


        if "history" not in st.session_state:
            st.session_state.history = []

        st.title("Welcome to GuideMySkinBot")


        def generate_answer():
            tokenizer, model = get_models()
            user_message = st.session_state.input_text
            inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
            result = model.generate(**inputs)
            message_bot = tokenizer.decode(
                result[0], skip_special_tokens=True
            )  # .replace("<s>", "").replace("</s>", "")

            st.session_state.history.append({"message": user_message, "is_user": True})
            st.session_state.history.append({"message": message_bot, "is_user": False})


        st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

        for i, chat in enumerate(st.session_state.history):
            st_message(**chat, key=str(i)) #unpacking

with st.container():
    st.write("---")
    st.header("TO FIND FLAWS")
    st.write("##")
    import streamlit as st
    import cv2
    import numpy as np
    import dlib
    
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

    # Function to update webcam feed
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
    # Streamlit UI
    st.title("Facial Flaw Detection")

    option = st.radio("Select an option:", ("Upload Image", "Use Webcam"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            st.image(image, channels="BGR", caption="Uploaded Image")

            # Process the image for facial flaws
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)

                # Check for acne
                acne_detected = detect_acne(image)

                # Check for pigmentation
                pigmented_areas = detect_pigmentation(image)

                # Check for wrinkles
                wrinkle_contours = detect_wrinkles(image)

                # Draw landmarks on the face
                for landmark in landmarks.parts():
                    cv2.circle(image, (landmark.x, landmark.y), 1, (0, 255, 0), -1)

                # Draw acne label if detected
                if acne_detected:
                    cv2.putText(image, "Acne", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw pigmented areas
                cv2.drawContours(image, pigmented_areas, -1, (0, 0, 255), 2)

                # Draw wrinkles
                cv2.drawContours(image, wrinkle_contours, -1, (255, 0, 0), 2)

            st.image(image, channels="BGR", caption="Processed Image")

    elif option == "Use Webcam":
        process_webcam_feed()
        st.write("Press 'q' to stop the webcam feed.")

        
            
    
        
        
    
    
