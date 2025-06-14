# Exercise & Nutrition AI

-----

## Overview

This lightweight Streamlit web app helps you track your gym workouts and vegetarian meals using **deep learning** and **computer vision**. It's designed for **fast inference**, **minimal dependencies**, and **real-world usability**.

-----

## Features

### Gym Exercise Classification

Upload short workout video clips to **automatically detect the type of gym exercise** (e.g., push-up, squat). The model, based on **SigLIP**, has been trained on a custom-scraped dataset for accurate classification.

### Food Recognition & Calorie Estimation

Simply upload an image of a vegetarian dish, and the app will **classify the food and fetch its calorie information**. This feature leverages a **SigLIP model fine-tuned on a vegetarian subset of Food101** for precise recognition and utilizes a cleaned dataset for nutritional data.

### Repetition Counter

Utilizing **MediaPipe Pose tracking**, this feature **counts your workout repetitions** by detecting body joints and calculating angles (e.g., elbow or knee angles). It works seamlessly with both live webcam feeds and uploaded videos, identifying motion phase changes to accurately tally reps.

-----

## Tech Stack

  * **Frontend/UI**: Streamlit
  * **Computer Vision**: OpenCV, MediaPipe
  * **ML Models**:
      * **SigLIP-based classifiers** for both food and workout recognition
      * **MediaPipe Pose** for repetition counting
      * **Custom calorie lookup** from a cleaned dataset/API

-----

## How to Run

Follow these steps to get the app running on your local machine:

```bash
git clone https://github.com/FowlDeadPool/exercise-nutrition-ai.git
cd exercise-nutrition-ai
pip install -r requirements.txt
streamlit run app.py
```

### Download Model Checkpoints (Required)

Before running the app, you need to download the pretrained models and place them in their respective folders:

  * **Food Classifier (SigLIP)**:
    [Download](https://drive.google.com/drive/folders/1i0yyg18JpPa6OiqaCaPx3mvMlBQ3k2SV?usp=drive_link) → Place in `Food-101-93M/model.safetensors`
  * **Exercise Classifier (SigLIP)**:
    [Download](https://drive.google.com/drive/folders/1MvQruDYjX5VTcsDvXRcNdL3JY4KZCw1k?usp=sharing) → Place in `Gym-Workout-Classifier-SigLIP2/model.safetensors`

*If the links open in your browser, right-click and select "Save As" or use `gdown` for direct download.*

-----

## Datasets & Model Details

### 📚 Datasets Used

| Task                    | Dataset                                                                                                               |
| :---------------------- | :-------------------------------------------------------------------------------------------------------------------- |
| Food Classification     | [Food41 (Kaggle)](https://www.kaggle.com/datasets/kmader/food41)                                                      |
| Exercise Classification | [Workout/Fitness Video Dataset (Kaggle)](https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video)         |
| Nutrition Info          | Cleaned version of [Daily Food and Nutrition Dataset](https://www.kaggle.com/datasets/ekrembayar/daily-food-consumption-and-nutrition) |

### 🧠 Model Details

  * **SigLIP**: Used for both food and exercise classification.
      * **Food Classification**: Fine-tuned on a subset of Food101 (Food41).
      * **Exercise Classification**: Fine-tuned on labeled frames from fitness videos.
  * **Repetition Counter**: Leverages MediaPipe Pose to detect 33 keypoints, calculate joint angles (e.g., elbow, knee), and count reps by detecting state transitions.

-----

## Future Improvements

  * Enhance SigLIP performance through contrastive pretraining.
  * Add support for tracking multi-exercise sessions.
  * Implement workout history logging with charts and export options.
  * Integrate voice feedback or sound cues for each repetition.

-----

## Acknowledgements

A big thank you to the creators and communities behind these incredible tools and resources:

  * Streamlit
  * SigLIP (Google)
  * MediaPipe
  * OpenCV
  * Kaggle Datasets
