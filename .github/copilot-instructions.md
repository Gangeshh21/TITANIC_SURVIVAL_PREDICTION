python3 -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn# Copilot Instructions for AI Coding Agents

## Project Overview
This repository is a simple Streamlit application for predicting Titanic survival using a pre-trained model. The main files are:
- `app.py`: Streamlit app entry point. Handles UI, user input, and model inference.
- `titanic_model.pkl`: Pre-trained machine learning model (likely scikit-learn or similar).
- `Untitled0.ipynb`: Jupyter notebook, possibly used for model training or exploration.

## Architecture & Data Flow
- The Streamlit app loads the model from `titanic_model.pkl` at startup.
- User inputs are collected via Streamlit widgets in `app.py`.
- Inputs are preprocessed and passed to the loaded model for prediction.
- Prediction results are displayed in the app UI.

## Developer Workflows
- **Run the app:**
  ```bash
  streamlit run app.py
  ```
- **Model updates:**
  - Update or retrain the model in a notebook (e.g., `Untitled0.ipynb`).
  - Save the new model as `titanic_model.pkl`.
  - Ensure input features in `app.py` match the model's expected format.
- **Debugging:**
  - Use Streamlit's sidebar and logging for quick UI-based debugging.
  - For model issues, inspect the notebook and model serialization.

## Project-Specific Patterns
- All model inference logic is centralized in `app.py`.
- Model file path is hardcoded; update both code and file if the model changes.
- Input features and preprocessing must match the model's training pipeline.
- No custom build or test scripts detected; use standard Python/Streamlit workflows.

## Integration Points
- External dependencies: Streamlit, scikit-learn (or similar), pandas, numpy.
- Model loading uses `pickle` or `joblib` (check `app.py` for details).
- No API endpoints or external service calls detected.

## Troubleshooting & Dependencies (important)

- This project runs inside a Python virtual environment. Activate the venv before running Streamlit:

```bash
# macOS / zsh example (adjust path if your venv is elsewhere)
source .venv/bin/activate
streamlit run app.py
```

- If you see errors like "ModuleNotFoundError: No module named 'google.generativeai'", install the missing package into the active venv. Common package names to try:

```bash
# preferred (official Google Generative AI SDK)
pip install google-generative-ai
# or if above fails, try alternate package name
pip install google-generativeai
```

- To make dependency management reproducible, create or update `requirements.txt` with the pinned packages you need:

```bash
pip freeze > requirements.txt
```

- If Streamlit appears to run a different project file (see logs referencing `/Users/gangesh/mindmapper_ai/app.py`), confirm you executed the right `streamlit run` command and that your terminal's working directory is `/Users/gangesh/titanic_streamlit_app`.

Example checklist to debug runtime import errors:

1. Activate venv: `source .venv/bin/activate`
2. From project root run: `python -c "import google.generativeai; print('ok')"` to check import
3. If import fails, `pip install <package>` and re-check
4. Restart Streamlit: stop server and `streamlit run app.py`


## Example: Model Prediction Pattern
```python
# ...existing code...
model = pickle.load(open('titanic_model.pkl', 'rb'))
# Collect user input
# Preprocess input
# prediction = model.predict([features])
# ...existing code...
```

## Key Files
- `app.py`: Main logic and UI
- `titanic_model.pkl`: Model artifact
- `Untitled0.ipynb`: Model development (if present)

---
**Feedback:**
If any section is unclear or missing important details, please specify which workflows, conventions, or integration points need further documentation.
