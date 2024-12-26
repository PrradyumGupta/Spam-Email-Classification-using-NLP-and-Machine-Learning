import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Streamlit app title
st.title("Spam Email Classifier")

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the uploaded Excel file
    data = pd.read_excel(uploaded_file)
    
    # Print the columns to ensure correct ones are selected
    st.write("Columns in the dataset:", data.columns)

    # Selecting relevant columns (adjust 'v1' and 'v2' based on your dataset)
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']

    # Preprocessing: Remove stopwords
    stop_words = set(stopwords.words('english'))

    # Function to remove stopwords from text
    def remove_stopwords(text):
        if not isinstance(text, str):
            text = str(text)  # Convert non-string data to string
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    # Apply stopword removal to the messages
    data['cleaned_message'] = data['message'].apply(remove_stopwords)

    # Check if there are any rows with invalid data after cleaning
    st.write("Cleaned Data Preview:", data.head())

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned_message'], data['label'], test_size=0.2, random_state=42)

    # Convert text data to numerical data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display accuracy and confusion matrix
    st.write(f"Accuracy: {accuracy}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)

    # Provide an interface for users to input a custom message
    st.subheader("Test the Classifier with Your Own Message")
    user_message = st.text_area("Enter a message to classify:")
    
    if user_message:
        # Preprocess the message
        cleaned_message = remove_stopwords(user_message)
        # Vectorize and predict
        user_vectorized = vectorizer.transform([cleaned_message])
        prediction = classifier.predict(user_vectorized)
        st.write(f"Prediction: {'Spam' if prediction[0] == 'spam' else 'Not Spam'}")

