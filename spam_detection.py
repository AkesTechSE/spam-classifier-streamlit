# coding: utf-8
# streamlit run spam_detection.py

# In[153]:


import nltk

nltk.download("stopwords")


# In[ ]:


import pandas as pd  # for data manipulation and analysis
import numpy as np  # it is for numerical valuse
from sklearn.model_selection import train_test_split  # it porvides d/t models
from sklearn.feature_extraction.text import TfidfVectorizer  # futer of extraction in nl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st  # for web app development
import pickle  # for saving the model


# In[155]:


# Step 2: Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")


# Step 3: Data Preprocessing
# Display the first few rows of the dataset
df.head()
# Rename the first two columns (others will be dropped later)
df = df.rename(columns={"v1": "label", "v2": "message"})
df.head()
# Removes duplicate rows from the dataset.
df.drop_duplicates(inplace=True)

# In[159]:

# Keep only relevant columns and drop NaN values
df = df[["label", "message"]].dropna()

# Convert labels to binary values
df["label"] = df["label"].map({"ham": 1, "spam": 0})

# In[161]:


df.head(10)


# In[162]:


y = df["label"]
x = df["message"]


# In[163]:


print(x)


# In[164]:


print(y)


# In[165]:


df["label"].value_counts()
df.info()
df.isnull().sum()


# In[166]:


df2 = df.copy()
# saving preprocessed data and importing the preprocessed data
df.to_csv("preprocessed_spam.csv", index=False)


# In[167]:


# Split data into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=45
)


# In[168]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# Fit and transform the training data
x_train_features = feature_extraction.fit_transform(x_train)

# Transform the test data using the same vectorizer
x_test_features = feature_extraction.transform(x_test)

# In[169]:


print(x_train_features)


# In[170]:


# Step 5: Train model
model = LogisticRegression(class_weight="balanced")
model.fit(x_train_features, y_train)


# Step 6: Evaluate model
# Predict on the test data
prediction_on_test_data = model.predict(x_test_features)

# Evaluate accuracy on the test data
test_data_accuracy = accuracy_score(y_test, prediction_on_test_data)
# This line prints the test accuracy, allowing you to see how well the model performs on unseen data.

print("Test Accuracy:", test_data_accuracy)


# Check overall input shape
print("Total data (x):", x.shape)

# Training and testing input sizes
print("Training data (x_train):", x_train.shape)
print("Testing data (x_test):", x_test.shape)

# Show first few test messages
print("\nFirst 5 test messages:\n", x_test.head())

# Check overall label shape
print("\nTotal labels (y):", y.shape)

# Training and testing label sizes
print("Training labels (y_train):", y_train.shape)
print("Testing labels (y_test):", y_test.shape)
# Show first few test labels


df.head(10)

df.describe()


from sklearn.metrics import accuracy_score, classification_report

# Predictions
train_preds = model.predict(x_train_features)
test_preds = model.predict(x_test_features)

# Accuracy
print("Training Accuracy:", accuracy_score(y_train, train_preds))
print("Test Accuracy:", accuracy_score(y_test, test_preds))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, test_preds))


# Streamlit App
st.title("SPD [Spam Detection Toole]")

st.write("Enter your message below to check if it is spam or not.")

# Custom CSS
st.markdown(
    """
    <style>
        .stButton > button {
            background-color: #4CAF50; /* Green */
            color: white;
            border: none;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px; /* Rounded corners */
            transition: background-color 0.3s; /* Animation */
        }
        .stButton > button:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        .stTextArea {
            border: 2px solid #4CAF50; /* Border color */
            border-radius: 8px; /* Rounded corners */
            padding: 10px; /* Padding inside the text area */
        }
        h1, h2, h3 {
            color: #333; /* Header color */
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Input field for user to enter their message
input_your_mail = st.text_area("Drop your message:")

# Button to trigger prediction
if st.button("Press to Predict"):
    if input_your_mail:
        with st.spinner("Waiting for response..."):
            # Transform the input data
            input_data_features = feature_extraction.transform([input_your_mail])

            # Make prediction
            prediction = model.predict(input_data_features)

        # Display the result
        result = "ham" if prediction[0] == 1 else "spam"
        st.success(f"Prediction for input mail: **{result}**")

        # Clear the input field after prediction
        input_your_mail = ""
    else:
        st.error("⚠️ Please enter a message to predict.")
