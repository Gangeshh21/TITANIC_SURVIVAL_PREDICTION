import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# Load the trained model (cached)
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "titanic_model.pkl"
    if not model_path.exists():
        st.error(f"Model not found at {model_path}. Please place `titanic_model.pkl` in the project root.")
        st.stop()
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()


# App title and description
st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival chances")
st.markdown("---")


# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    # Sex selection
    sex = st.selectbox("Sex", ["Male", "Female"])  
    sex_encoded = 0 if sex == "Male" else 1

    # Age input
    age = st.slider("Age", min_value=0, max_value=80, value=30)

    # Passenger Class
    pclass = st.selectbox("Passenger Class", [1, 2, 3],
                          help="1 = First Class, 2 = Second Class, 3 = Third Class")

with col2:
    st.subheader("👨‍👩‍👧‍👦 Family & Ticket")
    # Siblings/Spouses
    sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
    # Parents/Children
    parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
    # Fare
    fare = st.number_input("Fare (in £)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
    # Embarked
    embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
    embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]


st.markdown("---")


# Predict button
if st.button("🔮 Predict Survival"):
    # Prepare input data (must match training order)
    features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

    try:
        if hasattr(model, "predict_proba"):
            import streamlit as st
            import pickle
            import numpy as np
            import pandas as pd
            from pathlib import Path


            # Load the trained model (cached)
            @st.cache_resource
            def load_model():
                model_path = Path(__file__).parent / "titanic_model.pkl"
                if not model_path.exists():
                    st.error(f"Model not found at {model_path}. Please place `titanic_model.pkl` in the project root.")
                    st.stop()
                with open(model_path, "rb") as file:
                    model = pickle.load(file)
                return model


            model = load_model()


            # App title and description
            st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
            st.title("🚢 Titanic Survival Prediction")
            st.write("Enter passenger details to predict survival chances")
            st.markdown("---")


            # Create two columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👤 Personal Information")
                # Sex selection
                sex = st.selectbox("Sex", ["Male", "Female"])  
                sex_encoded = 0 if sex == "Male" else 1

                # Age input
                age = st.slider("Age", min_value=0, max_value=80, value=30)

                # Passenger Class
                pclass = st.selectbox("Passenger Class", [1, 2, 3],
                                      help="1 = First Class, 2 = Second Class, 3 = Third Class")

            with col2:
                st.subheader("👨‍👩‍👧‍👦 Family & Ticket")
                # Siblings/Spouses
                sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
                # Parents/Children
                parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
                # Fare
                fare = st.number_input("Fare (in £)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
                # Embarked
                embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
                embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]


            st.markdown("---")


            # Predict button
            if st.button("🔮 Predict Survival"):
                # Prepare input data (must match training order)
                features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)[0][1]
                        label = int(proba >= 0.5)
                    else:
                        label = int(model.predict(features)[0])
                        proba = None

                    # Display result
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    if label == 1:
                        if proba is not None:
                            st.success(f"✅ SURVIVED — Survival probability: {proba*100:.2f}%")
                        else:
                            st.success("✅ SURVIVED")
                        st.balloons()
                    else:
                        if proba is not None:
                            st.error(f"❌ DID NOT SURVIVE — Survival probability: {proba*100:.2f}%")
                        else:
                            st.error("❌ DID NOT SURVIVE")

                    # Show input summary
                    st.markdown("---")
                    st.subheader("📋 Input Summary")
                    input_df = pd.DataFrame({
                        "Feature": ["Passenger Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
                        "Value": [pclass, sex, age, sibsp, parch, f"£{fare}", embarked]
                    })
                    st.table(input_df)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


            # Sidebar information
            st.sidebar.title("ℹ️ About")
            st.sidebar.info(
                """
                This app predicts whether a Titanic passenger would have survived
                based on historical data using a Random Forest machine learning model.

                Features used: Passenger Class, Sex, Age, Siblings/Spouses, Parents/Children, Fare, Embarked

                Model Accuracy: ~82-85% (example)
                """
            )

            st.sidebar.markdown("---")
            st.sidebar.markdown("Made with ❤️ using Streamlit")
            st.title("🚢 Titanic Survival Prediction")
            st.write("Enter passenger details to predict survival chances")
            st.markdown("---")


            # Create two columns for layout
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👤 Personal Information")
                # Sex selection
                sex = st.selectbox("Sex", ["Male", "Female"])  
                sex_encoded = 0 if sex == "Male" else 1

                # Age input
                age = st.slider("Age", min_value=0, max_value=80, value=30)

                # Passenger Class
                pclass = st.selectbox("Passenger Class", [1, 2, 3],
                                      help="1 = First Class, 2 = Second Class, 3 = Third Class")

            with col2:
                st.subheader("👨‍👩‍👧‍👦 Family & Ticket")
                # Siblings/Spouses
                sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
                # Parents/Children
                parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
                # Fare
                fare = st.number_input("Fare (in £)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
                # Embarked
                embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
                embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]


            st.markdown("---")


            # Predict button
            if st.button("🔮 Predict Survival"):
                # Prepare input data (must match training order)
                features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(features)[0][1]
                        label = int(proba >= 0.5)
                    else:
                        label = int(model.predict(features)[0])
                        proba = None

                    # Display result
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    if label == 1:
                        if proba is not None:
                            st.success(f"✅ SURVIVED — Survival probability: {proba*100:.2f}%")
                        else:
                            st.success("✅ SURVIVED")
                        st.balloons()
                    else:
                        if proba is not None:
                            st.error(f"❌ DID NOT SURVIVE — Survival probability: {proba*100:.2f}%")
                        else:
                            st.error("❌ DID NOT SURVIVE")

                    # Show input summary
                    st.markdown("---")
                    st.subheader("📋 Input Summary")
                    input_df = pd.DataFrame({
                        "Feature": ["Passenger Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
                        import streamlit as st
                        import pickle
                        import numpy as np
                        import pandas as pd
                        from pathlib import Path


                        # Load the trained model (cached)
                        @st.cache_resource
                        def load_model():
                            model_path = Path(__file__).parent / "titanic_model.pkl"
                            if not model_path.exists():
                                st.error(f"Model not found at {model_path}. Please place `titanic_model.pkl` in the project root.")
                                st.stop()
                            with open(model_path, "rb") as file:
                                model = pickle.load(file)
                            return model


                        model = load_model()


                        # App title and description
                        st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")
                        st.title("🚢 Titanic Survival Prediction")
                        st.write("Enter passenger details to predict survival chances")
                        st.markdown("---")


                        # Create two columns for layout
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("👤 Personal Information")
                            # Sex selection
                            sex = st.selectbox("Sex", ["Male", "Female"])  
                            sex_encoded = 0 if sex == "Male" else 1

                            # Age input
                            age = st.slider("Age", min_value=0, max_value=80, value=30)

                            # Passenger Class
                            pclass = st.selectbox("Passenger Class", [1, 2, 3],
                                                  help="1 = First Class, 2 = Second Class, 3 = Third Class")

                        with col2:
                            st.subheader("👨‍👩‍👧‍👦 Family & Ticket")
                            # Siblings/Spouses
                            sibsp = st.number_input("Number of Siblings/Spouses", min_value=0, max_value=8, value=0)
                            # Parents/Children
                            parch = st.number_input("Number of Parents/Children", min_value=0, max_value=6, value=0)
                            # Fare
                            fare = st.number_input("Fare (in £)", min_value=0.0, max_value=500.0, value=50.0, step=0.1)
                            # Embarked
                            embarked = st.selectbox("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"])
                            embarked_encoded = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked]


                        st.markdown("---")


                        # Predict button
                        if st.button("🔮 Predict Survival"):
                            # Prepare input data (must match training order)
                            features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

                            try:
                                if hasattr(model, "predict_proba"):
                                    proba = model.predict_proba(features)[0][1]
                                    label = int(proba >= 0.5)
                                else:
                                    label = int(model.predict(features)[0])
                                    proba = None

                                # Display result
                                st.markdown("---")
                                st.subheader("📊 Prediction Result")
                                if label == 1:
                                    if proba is not None:
                                        st.success(f"✅ SURVIVED — Survival probability: {proba*100:.2f}%")
                                    else:
                                        st.success("✅ SURVIVED")
                                    st.balloons()
                                else:
                                    if proba is not None:
                                        st.error(f"❌ DID NOT SURVIVE — Survival probability: {proba*100:.2f}%")
                                    else:
                                        st.error("❌ DID NOT SURVIVE")

                                # Show input summary
                                st.markdown("---")
                                st.subheader("📋 Input Summary")
                                input_df = pd.DataFrame({
                                    "Feature": ["Passenger Class", "Sex", "Age", "Siblings/Spouses", "Parents/Children", "Fare", "Embarked"],
                                    "Value": [pclass, sex, age, sibsp, parch, f"£{fare}", embarked]
                                })
                                st.table(input_df)

                            except Exception as e:
                                st.error(f"Prediction failed: {e}")


                        # Sidebar information
                        st.sidebar.title("ℹ️ About")
                        st.sidebar.info(
                            """
                            This app predicts whether a Titanic passenger would have survived
                            based on historical data using a Random Forest machine learning model.

                            Features used: Passenger Class, Sex, Age, Siblings/Spouses, Parents/Children, Fare, Embarked

                            Model Accuracy: ~82-85% (example)
                            """
                        )

                        st.sidebar.markdown("---")
                        st.sidebar.markdown("Made with ❤️ using Streamlit")
