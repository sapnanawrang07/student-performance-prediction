
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from sklearn.model_selection import train_test_split


# -----------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)


# -----------------------------------
# 2. LOAD DATASET AND MODEL
# -----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/student-mat.csv", sep=";")


@st.cache_resource
def load_model():
    return joblib.load("models/random_forest.pkl")


df = load_data()
model = load_model()

st.title("🎓 Student Performance Analytics")
st.markdown(
    "### Machine Learning Based Student Performance Prediction System"
)

st.write(
    "Analyze student performance and estimate final grades "
    "using a trained Random Forest model."
)

st.divider()


# -----------------------------------
# 3. DASHBOARD METRICS
# -----------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", len(df))

col2.metric(
    "Average Final Grade",
    round(df["G3"].mean(), 2)
)

col3.metric(
    "Highest Final Grade",
    int(df["G3"].max())
)

col4.metric(
    "Lowest Final Grade",
    int(df["G3"].min())
)


st.divider()


# -----------------------------------
# 4. DASHBOARD TABS
# -----------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Analytics",
    "🔮 Grade Prediction",
    "📈 Model Evaluation"
])


# -----------------------------------
# TAB 1: ANALYTICS
# -----------------------------------

with tab1:

    st.header("Student Performance Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Final Grade Distribution")

        fig1 = px.histogram(
            df,
            x="G3",
            nbins=20,
            title="Distribution of Final Grades",
            labels={"G3": "Final Grade"}
        )

        st.plotly_chart(fig1, use_container_width=True)

    with col2:

        st.subheader("Study Time vs Final Grade")

        study_data = (
            df.groupby("studytime", as_index=False)["G3"]
            .mean()
        )

        fig2 = px.bar(
            study_data,
            x="studytime",
            y="G3",
            title="Average Grade by Study Time",
            labels={
                "studytime": "Study Time Category",
                "G3": "Average Final Grade"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Absences vs Final Grade")

    fig3 = px.scatter(
        df,
        x="absences",
        y="G3",
        title="Absences vs Final Grade",
        labels={
            "absences": "Number of Absences",
            "G3": "Final Grade"
        }
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)


# -----------------------------------
# TAB 2: GRADE PREDICTION
# -----------------------------------

with tab2:

    st.header("Predict Student Final Grade")

    st.write(
        "Enter student details below. "
        "The trained Random Forest model will estimate the final grade."
    )

    # Exclude target column
    feature_columns = df.drop(columns=["G3"]).columns.tolist()

    with st.form("student_prediction_form"):

        st.subheader("Enter Student Information")

        student_values = {}

        # Display inputs in two columns
        input_col1, input_col2 = st.columns(2)

        for index, column in enumerate(feature_columns):

            current_col = (
                input_col1 if index % 2 == 0 else input_col2
            )

            with current_col:

                # Categorical features
                if (
                    df[column].dtype == "object"
                    or df[column].dtype.name == "category"
                ):

                    options = sorted(
                        df[column].dropna().unique().tolist()
                    )

                    student_values[column] = st.selectbox(
                        column.replace("_", " ").title(),
                        options
                    )

                # Numerical features
                elif pd.api.types.is_numeric_dtype(df[column]):


                    minimum = int(df[column].min())
                    maximum = int(df[column].max())
                    default = int(df[column].median())

                    student_values[column] = st.number_input(
                        column.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

        submitted = st.form_submit_button(
            "🔮 Predict Final Grade",
            use_container_width=True
        )

    if submitted:

        # Create input DataFrame in training column order
        student_df = pd.DataFrame(
            [student_values],
            columns=feature_columns
        )

        # Predict grade
        prediction = model.predict(student_df)[0]

        prediction = max(0, min(20, prediction))

        st.divider()

        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Predicted Final Grade",
            f"{prediction:.2f} / 20"
        )

        if prediction >= 10:
            result_col2.success("Predicted result: Pass range")
        else:
            result_col2.warning("Predicted result: Below pass range")

        st.info(
            "This is a machine learning estimate, "
            "not a guaranteed academic result."
        )


# -----------------------------------
# TAB 3: MODEL EVALUATION
# -----------------------------------

with tab3:

    st.header("Machine Learning Model Evaluation")

    # Load model comparison results
    try:

        results = pd.read_csv("model_comparison.csv")

        st.subheader("Model Performance Comparison")

        st.dataframe(
            results.round(3),
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:

            fig4 = px.bar(
                results,
                x="Model",
                y="MAE",
                title="Mean Absolute Error (Lower is Better)"
            )

            st.plotly_chart(fig4, use_container_width=True)

        with col2:

            fig5 = px.bar(
                results,
                x="Model",
                y="R2 Score",
                title="R2 Score (Higher is Better)"
            )

            st.plotly_chart(fig5, use_container_width=True)

        # Recreate the same test split used during training
        X = df.drop(columns=["G3"])
        y = df["G3"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        predictions = model.predict(X_test)

        comparison = pd.DataFrame({
            "Actual Grade": y_test.values,
            "Predicted Grade": predictions
        })

        fig6 = px.scatter(
            comparison,
            x="Actual Grade",
            y="Predicted Grade",
            title="Actual vs Predicted Grades"
        )

        fig6.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=20,
            y1=20,
            line=dict(dash="dash")
        )

        st.plotly_chart(fig6, use_container_width=True)

    except FileNotFoundError:

        st.error(
            "model_comparison.csv not found. "
            "Please run train_model.py first."
        )


# -----------------------------------
# 5. FOOTER
# -----------------------------------

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Python, Pandas, Scikit-learn and Streamlit"
)
