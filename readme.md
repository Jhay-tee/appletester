# Fresh vs Rotten Apple Classifier

GET 324 — Cloud Computing and AI Model Deployment for Engineering Applications
Laboratory Exercise 10 (Mini-Project) | Group CV10 | Task: **Fresh Apple vs Rotten Apple**

A Streamlit web application that serves a MobileNetV2 transfer-learning model trained to classify
apple images as **Fresh** or **Rotten**.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application — loads the model and serves predictions |
| `model.keras` | Trained model file (produced by the training notebook — copy it in here) |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Files/folders excluded from version control |

## Running locally

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure model.keras (from the training notebook) is in this same folder

# 4. Run the app
streamlit run app.py
```

The app opens automatically in your browser (usually at `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. Push this project (including `model.keras`) to a GitHub repository.
2. Create an account at [streamlit.io](https://streamlit.io/) and sign in with GitHub.
3. Click **New app**, select this repository/branch, and set the main file path to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and launches the app,
   giving you a public URL you can share.

## Model details

- **Architecture:** MobileNetV2 (ImageNet weights) — frozen feature extraction, then fine-tuned
  on the top 30 layers.
- **Input size:** 224×224×3
- **Output:** single sigmoid unit (binary classification: Fresh Apple vs Rotten Apple)
- **Trained on:** [Fresh Apple Vs Rotten Apple Classification](https://www.kaggle.com/datasets/srishtisharma9977/fresh-apple-vs-rotten-apple-classification) (Kaggle)

## "Is this even an apple?" check

The fresh/rotten model was only ever trained on apple photos, so on its own it would confidently
(and wrongly) label a photo of anything — a dog, a car, a person — as "Fresh" or "Rotten". To
avoid that, `app.py` first runs the uploaded image through a general-purpose, 1000-class ImageNet
MobileNetV2 (`tf.keras.applications.MobileNetV2(weights="imagenet")`) as a gatekeeper:

- If it recognises the image as `Granny_Smith` (the closest ImageNet class to "apple") → proceeds
  normally to the fresh/rotten prediction.
- If it recognises some other fruit (fig, pomegranate, orange, etc.) → still shows the fresh/rotten
  prediction, but with a warning that it may be less reliable.
- If nothing fruit-like is recognised at all → skips the fresh/rotten prediction entirely and tells
  the user the image doesn't look like an apple, along with the model's actual best guess.

This is a simple heuristic, not a certified out-of-distribution detector, but it stops the app from
giving a confident-sounding wrong answer on completely unrelated images.