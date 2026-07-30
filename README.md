# Fresh vs Rotten Apple Classifier

GET 324 — Cloud Computing and AI Model Deployment for Engineering Applications  
Laboratory Exercise 10 (Mini-Project) | Group CV10  
Task: **Fresh Apple vs Rotten Apple**

A Streamlit web application that serves a MobileNetV2 transfer-learning model trained to classify apple images as either **Fresh** or **Rotten**.

---

## Project overview

This project demonstrates a complete mini-project workflow: dataset preparation, model training, evaluation, and a simple web deployment. The Streamlit app (`app.py`) loads a trained MobileNetV2-based binary classifier (`model.keras`) and returns predictions for uploaded images.

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application — loads the model and serves predictions |
| `model.keras` | Trained model file (produced by the training notebook — copy it in here) |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Files/folders excluded from version control |
| `members.md` | Team membership and registration numbers |

---

## Quick start — Running locally

1. Clone the repository and cd into the project directory.
2. Create and activate a virtual environment, install dependencies, and run the app:

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure model.keras (from the training notebook) is in this same folder

# 4. Run the app
streamlit run app.py
```

The app opens automatically in your browser (usually at http://localhost:8501).

---

## Deploying to Streamlit Community Cloud

1. Push this project (including `model.keras`) to a GitHub repository.  
   Note: Streamlit Cloud will download all repository files — include `model.keras` if it is small enough. For large models consider using an external model hosting service or Git LFS.
2. Create an account at https://streamlit.io/ and sign in with GitHub.
3. Click **New app**, select this repository/branch, and set the main file path to `app.py`.
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and launches the app, giving you a public URL you can share.

---

## Model details

- **Architecture:** MobileNetV2 (ImageNet weights) — frozen feature extraction, then fine-tuned on the top 30 layers.
- **Input size:** 224×224×3
- **Output:** single sigmoid unit (binary classification: Fresh Apple vs Rotten Apple)
- **Trained on:** Fresh Apple Vs Rotten Apple Classification dataset (Kaggle)

---

## "Is this even an apple?" check

The fresh/rotten model was only ever trained on apple photos, so on its own it might confidently label a photo of anything as "Fresh" or "Rotten". To help with that, `app.py` runs the uploaded image through a general-purpose, 1000-class ImageNet MobileNetV2 (`tf.keras.applications.MobileNetV2(weights="imagenet")`) alongside the main prediction:

- If it recognises the image as `Granny_Smith` (the closest ImageNet class to "apple") or another fruit, the prediction is shown as normal.
- If nothing fruit-like is recognised, a small note is shown alongside the prediction saying the image doesn't look like a typical apple photo — but the fresh/rotten prediction still displays, since a badly decayed apple can also fail this check.

This is a simple heuristic, not a certified out-of-distribution detector — it never blocks a prediction, it just adds context.

---

## Contributors — Group CV10

Akpan Jonathan Otobong coordinated the group and handled deployment. Team member contributions are listed below.

| # | Full Name | Registration Number | GitHub Username | Contribution |
|---|---|---|---|---|
| 1 | Akpan Jonathan Otobong | 23/EG/CV/064 | [Jhay-tee](https://github.com/Jhay-tee) | Group leader — coordination and deployment |
| 2 | Ndon, KukpongAbasi Ime | 23/EG/CV/024 | [Kp-Crypt](https://github.com/Kp-Crypt) | Dataset preparation and preprocessing |
| 3 | Ndon, Fidelis Silas | 23/EG/CV/074 | [Dreyquentin](https://github.com/Dreyquentin) | Dataset preparation and preprocessing |
| 4 | Egbo, Precious Sochima | 23/EG/CV/034 | [Sochi002](https://github.com/Sochi002) | Model development and training |
| 5 | Asuquo Victory Imoh | 23/CV/EG/084 | [vickeeimoh123](https://github.com/vickeeimoh123) | Model evaluation |
| 6 | Mbom, Unwana Ime | 24/EG/CV/184 | [Unmbi](https://github.com/Unmbi) | Application development |
| 7 | Sunday, Ikakke-abasi David | 23/EG/CV/044 | [ikakkesunday27-star](https://github.com/ikakkesunday27-star) | Documentation and report writing |

---

## Notes

- If your `model.keras` file is large (>100 MB) consider using Git LFS or hosting the model in cloud storage and downloading it at first run.  
- This project is for educational purposes — treat predictions as experimental and do not use in safety-critical settings.

---

If you'd like, I can also commit this README to a different branch or open a pull request instead of committing to the default branch.