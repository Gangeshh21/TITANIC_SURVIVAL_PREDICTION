# Titanic Streamlit App

A simple Streamlit web app that loads a pre-trained Titanic survival model and predicts the probability that a passenger would survive given a set of features.

Contents
- `app.py` — Streamlit front-end and small inference wrapper (minimal preprocessing).
- `titanic_model.pkl` — Pre-trained model (pickle file). Place it in the repo root.
- `requirements.txt` — Python packages required to run the app.
- `.github/copilot-instructions.md` — Instructions for AI coding agents.

Quick start (local)

1. Clone the repository (or use the folder on your machine).

```bash
# if you haven't cloned yet
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure `titanic_model.pkl` is in the repo root. If you don't have it, place a compatible model there.

5. Run the app:

```bash
streamlit run app.py
```

What the app shows

- The app provides UI controls for typical Titanic features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
- When you click `Predict` the app passes these inputs to the loaded model.
- If the model implements `predict_proba`, the app shows a survival probability (a number between 0 and 1). This represents the model's estimated probability that the passenger survives.
  - Example: `0.73` means the model estimates a 73% chance of survival for the given features.
- If the model only supports `predict`, the app shows the predicted class label (e.g., `0` or `1`).

Notes about the probability

- A probability is the model's confidence expressed as a decimal between 0 and 1. The app displays it and maps it to a simple human message:
  - >= 0.5 → "likely survive"
  - < 0.5 → "likely not survive"
- The absolute reliability of that number depends on the model and training data quality. Treat it as a predictive estimate, not ground truth.

Preparing this repo for GitHub

Below are step-by-step commands to create a new GitHub repository and push your local project.

A. Using the GitHub website

1. Go to https://github.com/new and create a new repository (public or private).
2. Copy the repo URL (HTTPS or SSH).
3. In your project folder run:

```bash
git init
git add .
git commit -m "Initial commit: Titanic Streamlit app"
# Add remote (replace <URL> with your repo URL)
git remote add origin <URL>
git branch -M main
git push -u origin main
```

B. Using the GitHub CLI (gh)

If you have `gh` installed and authenticated:

```bash
gh repo create <username>/<repo-name> --public --source=. --remote=origin --push
```

C. Important notes

- If your model file (`titanic_model.pkl`) is larger than 100 MB, GitHub will refuse the push. Use Git LFS in that case:

```bash
brew install git-lfs
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add titanic_model.pkl
git commit -m "Add model with Git LFS"
```

- Add a `.gitignore` (see example in this repo) to exclude `.venv` and other transient files.

Files to include in the repo

- `app.py`
- `requirements.txt`
- `titanic_model.pkl` (or instructions to download it elsewhere)
- `.github/copilot-instructions.md`
- `README.md`
- `.gitignore`

Security notes

- Do NOT commit secrets (API keys, service account JSON, .env files) to the repository. Use GitHub Secrets for CI/CD.

If you want, I can:
- initialize git in the folder and make the initial commit for you, and/or
- create the remote GitHub repo using `gh` and push the code.

Tell me which you prefer and I will run the commands for you.
