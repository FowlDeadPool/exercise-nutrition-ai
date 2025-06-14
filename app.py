from queue import Queue
import threading
import numpy as np
import cv2
import tempfile
import pandas as pd
import requests
import os
from transformers import SiglipImageProcessor, SiglipForImageClassification
import torch
from PIL import Image
import streamlit as st
import time



# Streamlit page configuration
st.set_page_config(page_title="Fitness & Nutrition Tracker", layout="centered")
st.title("🏋️ Fitness & 🍱 Nutrition Tracker")


# Queue and logging setup
frame_queue = Queue(maxsize=10)
prediction_log = []
running = True

# Background thread for frame classification
def classify_frames():
    while running or not frame_queue.empty():
        try:
            frame_data = frame_queue.get(timeout=1)
        except:
            continue

        frame, timestamp = frame_data
        # Convert to PIL Image and classify
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model_2(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model_2.config.id2label[predicted_class_idx]
        
        prediction_log.append({
            "timestamp": timestamp,
            "prediction": predicted_label
        })

# Start inference thread
classifier_thread = threading.Thread(target=classify_frames)
classifier_thread.start()


# Load the pre-trained model and processor
@st.cache_resource
def load_model():
    model_path = "./Gym-Workout-Classifier-SigLIP2"
    model = SiglipForImageClassification.from_pretrained(model_path)
    processor = SiglipImageProcessor.from_pretrained(model_path)
    return model, processor

model_2, processor_2 = load_model()

# Load image classification model and processor
@st.cache_resource
def load_model():
    model_path = "./Food-101-93M"
    model = SiglipForImageClassification.from_pretrained(model_path)
    processor = SiglipImageProcessor.from_pretrained(model_path)
    return model, processor


model, processor = load_model()

# API Key for API Ninjas
api_key = "s8BxGJt0mdJiezmSk8je2g==qufz4TRj4leLecTa"


# Function to get nutrition info
def get_nutrition_info(food_item):
    url = f"https://api.calorieninjas.com/v1/nutrition?query={food_item}"
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            return data["items"][0]  # Return first match
    return None


# Tabs for different classifiers
tab0, tab1, tab2, tab3 = st.tabs(
    ["🏠 Home", "🏃 Exercise Classifier", "🍔 Food & Calorie Classifier", "🔁 Exercise Repetition Counter"])


# -------------------------------
# Home Tab
# -------------------------------
with tab0:  
    st.markdown("## 👋 Welcome to the Fitness & Nutrition Tracker")

    st.image("static\healthy-lifestyle-diet-fitness-vector-sign-shape-heart-with-multiple-icons-depicting-various-sports-vegetables-cereals-seafood-meat-fruit-sleep-weight-beverages_1284-44073145.png", use_container_width=True)

    st.markdown("""
    Staying healthy doesn't have to be complicated. Whether you're training at the gym, keeping an eye on your meals, or just trying to stick to your fitness goals — this app is built to support you every step of the way.

    Here's what you can do here:
    """)

    st.markdown("###  Track Your Workouts")
    st.markdown("""
    Upload a workout video or use your webcam to automatically detect which exercise you're doing — like push-ups, squats, or pull-ups — and count how many reps you've done. No wearables or extra sensors needed.
    """)

    st.markdown("###  Track Your Meals Smarter")
    st.markdown("""
    Snap a picture of your meal, and the app will try to recognize the food and estimate its calorie content using a built-in database. It’s a fast way to stay mindful of your nutrition without manually looking things up.
    """)

    st.markdown("###  Keep Track of Progress")
    st.markdown("""
    Counting reps can be tiring — let the app handle it. Whether you're uploading videos or working out live in front of your webcam, the repetition counter gives you instant feedback and keeps track of your performance.
    """)

    st.markdown("###  How to Get Started")
    st.markdown("""
    Use the tabs at the top:
    - **Exercise Classifier** – Upload a workout video and find out what exercise you're doing.
    - **Food & Calorie Classifier** – Upload a food photo and get a quick nutrition breakdown.
    - **Exercise Repetition Counter** – Start a live or uploaded workout session and count your reps in real-time.
    """)

    st.markdown("###  Built With Simplicity in Mind")
    st.markdown("""
    This app is designed to be lightweight, beginner-friendly, and practical. Whether you're just starting out on your fitness journey or you're already deep into it, you'll find something useful here.
    """)

    st.success("Let’s get moving — choose a tab above and start tracking!")

# -------------------------------
# Exercise Classifier Tab
# -------------------------------
with tab1:
    st.header("🏃 Exercise Video Classifier")
    
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], key="video")

    if video_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name

        st.video(video_path)

        # Start button
        if st.button("Classify Exercise"):
            with st.spinner("Classifying..."):
                # Open video and process frames
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                    # Send frame to background classifier thread
                    if frame_count % 5 == 0:
                        frame_queue.put((frame.copy(), timestamp))

                    frame_count += 1

                cap.release()

                # Show predicted exercise (from log)
                if prediction_log:
                    predictions = [entry['prediction'] for entry in prediction_log]
                    most_common_prediction = max(set(predictions), key=predictions.count)
                    st.success(f"Detected Exercise: **{most_common_prediction}**")

# Clean up thread
running = False
classifier_thread.join()


# -------------------------------
# Food Classifier Tab
# -------------------------------
with tab2:
    st.header("🍔 Food & Calorie Classifier")
    image_file = st.file_uploader("Upload a food image", type=[
                                  "jpg", "jpeg", "png"], key="image")

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded Food Image",
                 use_container_width=True)

        if st.button("Classify Food"):
            with st.spinner("Classifying..."):
                # Run model prediction
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_class_idx = logits.argmax(-1).item()
                    predicted_label = model.config.id2label[predicted_class_idx]

                st.success(f"🍽️ Predicted Food: **{predicted_label}**")

                # Get nutrition info
                nutrition = get_nutrition_info(predicted_label)

                if nutrition:
                    nutrition_data = {
                        "Nutrient": [
                            "Calories (kcal)", "Serving Size (g)", "Fat (g)",
                            "Saturated Fat (g)", "Protein (g)", "Carbohydrates (g)",
                            "Sugar (g)", "Fiber (g)", "Cholesterol (mg)", "Sodium (mg)"
                        ],
                        "Value": [
                            nutrition.get("calories", "N/A"),
                            nutrition.get("serving_size_g", "N/A"),
                            nutrition.get("fat_total_g", "N/A"),
                            nutrition.get("fat_saturated_g", "N/A"),
                            nutrition.get("protein_g", "N/A"),
                            nutrition.get("carbohydrates_total_g", "N/A"),
                            nutrition.get("sugar_g", "N/A"),
                            nutrition.get("fiber_g", "N/A"),
                            nutrition.get("cholesterol_mg", "N/A"),
                            nutrition.get("sodium_mg", "N/A")
                        ]
                    }

                    df = pd.DataFrame(nutrition_data)
                    st.table(df)

                else:
                    st.warning(
                        "⚠️ Could not fetch nutrition info from the API.")

                    # Fallback: Calorie lookup
                    calorie_lookup = {
                        "pizza": 285,
                        "burger": 295,
                        "apple_pie": 320,
                        # Add more mappings as needed
                    }
                    calories = calorie_lookup.get(
                        predicted_label.lower(), "N/A")
                    st.info(f"🔢 Estimated Calories: **{calories} kcal**")


# -------------------------------
# Exercise Repetition Counter Tab
# -------------------------------
with tab3:
    st.header("🏋️ Exercise Repetition Counter")
    
    
    #     # Mode switch using horizontal buttons
    # st.markdown("### Mode Selection")
    
    # Create columns for horizontal buttons
    col1, col2 = st.columns([1, 1])  # Two equal-width columns
    
    # Place buttons in the columns
    with col1:
        live_mode_button = st.button("📷 Live Mode", key="live_mode_button")
    
    with col2:
        upload_mode_button = st.button("📁 Upload Mode", key="upload_mode_button")
    
    # Initialize session state if it doesn't exist
    if "live_mode" not in st.session_state:
        st.session_state["live_mode"] = True  # Default to Live Mode
    
    # Button click logic to toggle between modes
    if live_mode_button:
        st.session_state["live_mode"] = True
    elif upload_mode_button:
        st.session_state["live_mode"] = False
    
   


    # Exercise selection
    exercise = st.selectbox(
        "Choose an exercise to track:",
        ("Pull-ups", "Push-ups", "Squats", "Bicep Curls", "Deadlifts")
    )

    # Start and Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start", key="start_button")
    with col2:
        stop_button = st.button("⏹️ Stop", key="stop_button")

    if "stop" not in st.session_state:
        st.session_state["stop"] = False

    if stop_button:
        st.session_state["stop"] = True

    # Import counters
    from Counters.PullUpCounter import PullupCounter
    from Counters.PushUpCounter import PushupCounter
    from Counters.SquatCounter import SquatCounter
    from Counters.BicepCurlCounter import BicepCurlCounter
    from Counters.DeadliftCounter import DeadliftCounter

    exercise_classes = {
        "Pull-ups": PullupCounter,
        "Push-ups": PushupCounter,
        "Squats": SquatCounter,
        "Bicep Curls": BicepCurlCounter,
        "Deadlifts": DeadliftCounter
    }

    def run_video_counter(counter_class, video_path):
        counter = counter_class(video_path=video_path)
        st.session_state["counter_instance"] = counter  # 🔥 Track the instance
        cap = counter.cap
        stframe = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("Video ended or can't be read.")
                break

            frame = counter.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels='RGB', use_container_width=True)

            if st.session_state.get("stop", False):
                break

        cap.release()

    def run_live_counter(counter_class):
        counter = counter_class()
        st.session_state["counter_instance"] = counter  
        cap = counter.cap
        stframe = st.empty()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.warning("Failed to read from webcam.")
                break

            frame = counter.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels='RGB', use_container_width=True)

            if st.session_state.get("stop", False):
                break

        cap.release()
        

    # Handle input based on mode
    if not st.session_state["live_mode"]:
        uploaded_video = st.file_uploader(
            "Upload a workout video", type=["mp4", "mov", "avi"])
        if uploaded_video and start_button:
            st.session_state["stop"] = False
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            run_video_counter(exercise_classes[exercise], video_path)

    elif st.session_state["live_mode"] and start_button:
        st.session_state["stop"] = False
        run_live_counter(exercise_classes[exercise])
        
        
            

    
    