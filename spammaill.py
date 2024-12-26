import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Ensure stopwords is downloaded
nltk.download('stopwords')

# Load your data (assuming it's in an Excel file)
file_path = "spam01.xlsx"  # Adjust the file path if needed

data = pd.read_excel(file_path)

# Print the columns to ensure correct ones are selected
print("Columns in the dataset:", data.columns)

# Selecting relevant columns (adjust 'v1' and 'v2' based on your dataset)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Preprocessing: Remove stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords from text
def remove_stopwords(text):
    # Ensure the text is a string
    if not isinstance(text, str):
        text = str(text)  # Convert non-string data to string
    words = text.split()  # Split the text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the stopword removal to the messages
data['cleaned_message'] = data['message'].apply(remove_stopwords)

# Check if there are any rows with invalid data after cleaning
print("Cleaned data preview:")
print(data.head())

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

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
