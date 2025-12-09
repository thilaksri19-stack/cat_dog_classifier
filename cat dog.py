import streamlit as st
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Page Design
st.set_page_config(page_title="Dog vs Cat Classifier", layout="wide")
st.title("🐶🐱 Dog vs Cat Classifier UI")
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.header("📤 Upload Dataset")
upload = st.file_uploader("Upload dog_cat_dataset.csv", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.success("Dataset uploaded successfully!")
    st.write(df.head())

    # Preprocessing
    df['Weight'] = df['Weight'].astype(int)

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col] = le.fit_transform(df[col])

    x = df.drop('Label', axis=1)
    y = df['Label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    st.header("🤖 Select Model to Train")

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K Nearest Neighbors": KNeighborsClassifier()
    }

    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    if st.button("Train Model"):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**Recall:** {rec:.2f}")

        st.success("Model trained successfully!")

else:
    st.info("Please upload a dataset to proceed.")