import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import uuid
import os

# Load Data Function
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip().str.lower()
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Train ML Model
def train_model(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    return model

# Compute Feature Influence
def compute_influence(model, X):
    return X * model.coef_[0]

# Plot Influence Ranking
def plot_influence_ranking(influences):
    label_map = {
        "smoke": "Smoking", "alco": "Alcohol Consumption", "active": "Physical Activeness",
        "cholesterol": "Cholesterol Levels", "gluc": "Glucose Levels",
        "age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight",
        "ap_hi": "Systolic Blood Pressure", "ap_lo": "Diastolic Blood Pressure"
    }
    mean_influence = influences.abs().mean().sort_values(ascending=False)
    mean_influence = mean_influence.drop(labels=['id'], errors='ignore')
    renamed_index = [label_map.get(col, col) for col in mean_influence.index]
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=mean_influence.values, y=renamed_index, palette='coolwarm', ax=ax)
    ax.set_title("Heart Disease Risk Factor Influence")
    ax.set_xlabel("Influence of Parameter(%)")
    ax.set_ylabel("Parameter")
    return fig

# Plot Affected Locations
def plot_location_ranking(patients_data):
    location_counts = patients_data['location'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=location_counts.values, y=location_counts.index, palette='coolwarm', ax=ax)
    ax.set_title("Heart Disease Prevalence by Location")
    ax.set_ylabel("Cities")
    ax.set_xlabel("No. of Persons at Risk")
    return fig

# Plot Health Distribution
def plot_health_distribution(filtered_data):
    count_data = filtered_data['prediction'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(x=count_data.index, y=count_data.values, palette='coolwarm', ax=ax)
    ax.set_title("People at Risk")
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_xlabel("General Risk Levels of Having Heart Disease")
    ax.set_ylabel("Number of People in Current Demographic")
    return fig

# Plot Sickness Probability
def plot_sickness_percentage(filtered_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.histplot(filtered_data['probability_sick'] * 100, bins=10, kde=True, color='red', ax=ax)
    ax.set_title("Percentage Distribution of Heart Disease Risk")
    ax.set_xlabel("Percentage Risk of Having Heart Disease")
    ax.set_ylabel("Number of People in Current Demographic")
    return fig


# Plot Pie Chart for Binary Categorical Columns
def plot_pie_chart(data, column):
    count_data = data[column].value_counts()
    label_map = {
        "smoke": "Smokers", "alco": "Alcohol Consumers", "active": "Physically Active Persons",
        "cholesterol": "Cholesterol Levels", "gluc": "Glucose Levels"
    }
    labels = {
        "cholesterol": ["Low: <200", "Medium: 200-239", "High: >=240"],
        "gluc": ["Low: <70", "Medium: 70-99", "High: >=100"],
        "gender": ["Male", "Female"],
        "smoke": ["No", "Yes"],
        "alco": ["No", "Yes"],
        "active": ["Inactive", "Active"]
    }
    label = label_map.get(column, column.capitalize())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(count_data, labels=labels.get(column, count_data.index), autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Distribution of {label}")
    return fig

# Plot Frequency Distribution
def plot_frequency_distribution(data, column, bins=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    label_map = {
        "age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight",
        "ap_hi": "Systolic Blood Pressure", "ap_lo": "Diastolic Blood Pressure",
        "cholesterol": "Cholesterol Levels", "gluc": "Glucose Levels"
    }
    label = label_map.get(column, column)
    if column == "age":
        data[column] = data[column] // 365  # Convert age from days to years
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {label}")
    sns.histplot(data[column], bins=bins, kde=True, ax=ax)
    return fig

def government_side():
    st.title("HeartDrive - Government Panel")

    heart_data = load_data("cardio_train.csv")
    patients_file = st.file_uploader("Upload Patients Dataset (CSV)", type=["csv"])

    if heart_data is not None and patients_file is not None:
        patients_data = load_data(patients_file)
        if patients_data is None:
            return

        if 'cardio' not in heart_data.columns:
            st.error("The 'cardio' column is missing in the heart dataset!")
            return

        # Selecting only numeric features for ML model
        X = heart_data.drop(columns=['cardio'], errors='ignore').select_dtypes(include=[np.number])
        Y = heart_data['cardio']

        # Ensure required columns exist in patients_data
        required_columns = set(X.columns).union({'age', 'gender', 'location'})
        missing_columns = required_columns - set(patients_data.columns)

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return

        model = train_model(X, Y)
        patients_data['probability_sick'] = model.predict_proba(patients_data[X.columns])[:, 1]
        patients_data['prediction'] = (patients_data['probability_sick'] > 0.5).astype(int)
        influences = compute_influence(model, patients_data[X.columns])

        st.write(f"**Model Bias (Intercept):** {model.intercept_[0]:.4f}")

        # 📊 Overall Data Analysis - Side by Side
        st.subheader("Overall Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_influence_ranking(influences))
        with col2:
            st.pyplot(plot_location_ranking(patients_data))

        # 📉 *Filtered Data Analysis - Age, Gender, Location*
        st.subheader("📉 Filtered Data Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            age_range = st.selectbox("Select Age Range", ["All", "30-40", "40-50", "50-60", "60-70"], key="filtered_age")
        with col2:
            gender_filter = st.selectbox("Select Gender", ["All", "Male", "Female"], key="filtered_gender")
        with col3:
            location_filter = st.selectbox("Select Location", ["All", "Pune", "Mumbai", "Delhi", "Bangalore", "Chennai"], key="filtered_location")

        # Apply filters only to this section
        filtered_data = patients_data.copy()
        if age_range != "All":
            min_age, max_age = map(int, age_range.split('-'))
            filtered_data = filtered_data[(filtered_data['age'] >= min_age * 365) & (filtered_data['age'] < max_age * 365)]
        if gender_filter != "All":
            gender_code = 1 if gender_filter == "Male" else 2
            filtered_data = filtered_data[filtered_data['gender'] == gender_code]
        if location_filter != "All":
            filtered_data = filtered_data[filtered_data['location'] == location_filter]

        # Display filtered analysis
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_health_distribution(filtered_data))
        with col2:
            st.pyplot(plot_sickness_percentage(filtered_data))

        # 📊 Risk Summary
        sick_percentage = (filtered_data['prediction'].sum() / len(filtered_data)) * 100 if len(filtered_data) > 0 else 0
        if sick_percentage >= 75:
            st.error(f"🚨 Urgent: {sick_percentage:.1f}% of this population is at high risk! Immediate intervention is necessary!")
        elif sick_percentage >= 50:
            st.warning(f"⚠ Warning: {sick_percentage:.1f}% of this population is at risk. Preventive measures are advised.")
        elif sick_percentage >= 25:
            st.info(f"🔍 Caution: {sick_percentage:.1f}% of individuals are at risk. Awareness and monitoring are recommended.")
        else:
            st.success(f"✅ Good News: Only {sick_percentage:.1f}% of this population is at risk.")

         # Advanced View - Parameter Analysis by Location
        st.subheader("Advanced View - Parameter Analysis by Location")
        selected_city = st.selectbox("Select a City for Analysis", ["Mumbai", "Delhi", "Chennai", "Pune", "Bangalore"], key="advanced_location")
        city_data = patients_data[patients_data['location'] == selected_city]

        if not city_data.empty:
            numeric_columns = [col for col in city_data.columns if col not in {'id', 'probability_sick', 'prediction', 'location'}]
            for i in range(0, len(numeric_columns), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(numeric_columns):
                        column = numeric_columns[i + j]
                        label_map = {"age": "Age", "gender": "Gender", "height": "Height", "weight": "Weight", "ap_hi": "Systolic Blood Pressure", "ap_lo": "Diastolic Blood Pressure", "cholesterol": "Cholesterol Levels", "gluc": "Glucose Levels", "smoke": "Smokers", "alco": "Alcohol Consumers", "active": "Physically Active Persons"}
                        label = label_map.get(column, column)
                        with cols[j]:
                            st.subheader(f"Distribution of {label}")
                            if column in ["gender", "smoke", "alco", "active", "cholesterol", "gluc"]:
                                st.pyplot(plot_pie_chart(city_data, column))
                            else:
                                st.pyplot(plot_frequency_distribution(city_data, column))
        else:
            st.warning("No data available for the selected city.")
 
def save_user_data(user_data):
    filename = "user_data.csv"

    # Check if file exists and is not empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        df = pd.read_csv(filename)

        # Ensure 'id' column exists and is numeric
        if 'id' in df.columns and not df.empty:
            df['id'] = pd.to_numeric(df['id'], errors='coerce')  # Convert to numeric, coerce errors to NaN
            last_id = df['id'].dropna().max()  # Drop NaN and get the max ID

            if pd.isna(last_id):  # If last_id is NaN, start with 1
                last_id = 0
        else:
            last_id = 0
    else:
        last_id = 0

    user_data["id"] = int(last_id) + 1  # Ensure ID is an integer and increment

    # Append new user data
    df = pd.DataFrame([user_data])
    df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

    # Display last 3 rows
    df = pd.read_csv(filename).tail(3)
    st.subheader("📋 Last 3 Records:")
    st.dataframe(df)

def user_side():
    st.title("HeartDrive - User Panel")
    st.subheader("Enter Your Health Information")

    user_data = {
        "id": str(uuid.uuid4()),
        "age": st.number_input("Age", min_value=1, max_value=120, step=1) * 365,
        "gender": 1 if st.selectbox("Gender", ["Male", "Female"]) == "Male" else 2,
        "height": st.number_input("Height (cm)", min_value=50, max_value=250, step=1),
        "weight": round(st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, step=0.1), 1),
        "ap_hi": st.number_input("Systolic Blood Pressure", min_value=50, max_value=250, step=1),
        "ap_lo": st.number_input("Diastolic Blood Pressure", min_value=30, max_value=150, step=1),
        "cholesterol": st.number_input("Cholesterol Level", min_value=1, max_value=10, step=1),
        "gluc": st.number_input("Glucose Level", min_value=1, max_value=10, step=1),
        "smoke": 1 if st.selectbox("Smoker", ["Yes", "No"]) == "Yes" else 0,
        "alco": 1 if st.selectbox("Alcohol Intake", ["Yes", "No"]) == "Yes" else 0,
        "active": 1 if st.selectbox("Physically Active", ["Yes", "No"]) == "Yes" else 0,
        "location": st.selectbox("Location", ["Chennai", "Mumbai", "Pune", "Bangalore", "Delhi"]).capitalize()
    }

    if st.button("Submit"):
        save_user_data(user_data)
        st.success("Your information has been recorded.")

# CSV file paths
GOVT_CSV = "govt_log.csv"
USER_CSV = "user_log.csv"

# Ensure CSV files exist
def initialize_csv(file_path, columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

initialize_csv(GOVT_CSV, ["Username", "Gov_UID", "Password"])
initialize_csv(USER_CSV, ["User_ID", "Aadhar", "Password"])

# Function to check if user exists in CSV
def check_credentials(file_path, id_column, id_value, password):
    df = pd.read_csv(file_path)
    user = df[df[id_column] == id_value]
    if not user.empty and user.iloc[0]["Password"] == password:
        return True
    return False

def register_user(file_path, data, id_column):
    df = pd.read_csv(file_path)
    
    # Check if user ID already exists
    if data[id_column] in df[id_column].values:
        return False
    
    # Convert new data into a DataFrame and concatenate
    new_entry = pd.DataFrame([data])
    df = pd.concat([df, new_entry], ignore_index=True)

    # Save updated DataFrame
    df.to_csv(file_path, index=False)
    return True


# Define CSV paths
GOVT_CSV = "govt_log.csv"
USER_CSV = "user_log.csv"

# Ensure CSV files exist
def ensure_csv_exists(file_path, columns):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(file_path, index=False)

ensure_csv_exists(GOVT_CSV, ["Username", "Gov_UID", "Password"])
ensure_csv_exists(USER_CSV, ["User_ID", "Aadhar", "Password"])

# Secure Login Verification
def verify_login(csv_path, id_column, user_id, password):
    """Verify user credentials from the CSV file."""
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")  # Read CSV safely
        user_data = df.loc[df[id_column].str.strip() == str(user_id).strip()]  # Match ID
        if not user_data.empty and user_data["Password"].values[0] == password.strip():
            return True
    except Exception as e:
        print(f"Error verifying login: {e}")
    return False

def register_user(csv_path, new_data, id_column):
    """Register a new user, ensuring uniqueness of the ID."""
    try:
        # Prevent blank fields
        if any(value.strip() == "" for value in new_data.values()):
            return False, "All fields are required."

        df = pd.read_csv(csv_path, dtype=str).fillna("")  # Load CSV safely

        # Check if the ID already exists
        if new_data[id_column].strip() in df[id_column].astype(str).str.strip().values:
            return False, "ID already exists! Please login."

        # Create a DataFrame from the new data
        new_entry = pd.DataFrame([new_data])

        # Concatenate and save
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_path, index=False)

        return True, "Registered successfully! Please login."

    except Exception as e:
        print(f"Error during registration: {e}")
        return False, "Registration failed."

# Government Login
def government_login():
    st.title("🏛 Government Login")
    username = st.text_input("Username")
    gov_uid = st.text_input("Government UID")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "main"
            st.rerun()
    
    with col2:
        if st.button("Login"):
            if verify_login(GOVT_CSV, "Gov_UID", gov_uid, password):
                st.session_state.gov_logged_in = True
                st.session_state.page = "government"
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with col3:
        if st.button("Register"):
            new_gov = {"Username": username.strip(), "Gov_UID": gov_uid.strip(), "Password": password.strip()}
            success, message = register_user(GOVT_CSV, new_gov, "Gov_UID")
            if success:
                st.success(message)
            else:
                st.error(message)

# User Login
def user_login():
    st.title("👥 User Login")
    username = st.text_input("User ID")
    aadhar = st.text_input("Aadhar Number")
    password = st.text_input("Password", type="password")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "main"
            st.rerun()

    with col2:
        if st.button("Login"):
            if verify_login(USER_CSV, "Aadhar", aadhar, password):
                st.session_state.user_logged_in = True
                st.session_state.page = "user"
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with col3:
        if st.button("Register"):
            new_user = {"User_ID": username.strip(), "Aadhar": aadhar.strip(), "Password": password.strip()}
            success, message = register_user(USER_CSV, new_user, "Aadhar")
            if success:
                st.success(message)
            else:
                st.error(message)

def main():
    st.set_page_config(layout="wide")

    if "page" not in st.session_state:
        st.session_state.page = "main"
    if "gov_logged_in" not in st.session_state:
        st.session_state.gov_logged_in = False
    if "user_logged_in" not in st.session_state:
        st.session_state.user_logged_in = False

    if st.session_state.page == "main":
        st.title("Welcome to HeartDrive")
        st.subheader("Choose a Portal:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏛 Government Portal"):
                st.session_state.page = "gov_login"
                st.rerun()
        with col2:
            if st.button("👥 User Portal"):
                st.session_state.page = "user_login"
                st.rerun()

    elif st.session_state.page == "gov_login":
        government_login()

    elif st.session_state.page == "user_login":
        user_login()

    elif st.session_state.page == "government":
        if not st.session_state.gov_logged_in:
            st.session_state.page = "gov_login"
            st.rerun()
        if st.button("⬅ Logout"):
            st.session_state.gov_logged_in = False
            st.session_state.page = "main"
            st.rerun()
        government_side()

    elif st.session_state.page == "user":
        if not st.session_state.user_logged_in:
            st.session_state.page = "user_login"
            st.rerun()
        if st.button("⬅ Logout"):
            st.session_state.user_logged_in = False
            st.session_state.page = "main"
            st.rerun()
        user_side()

if __name__ == "__main__":
    main()