import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
df = pd.read_csv("spam_dataset.csv", encoding='latin-1')

# Keep only necessary columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
elif 'label' in df.columns and 'message' in df.columns:
    df = df[['label', 'message']]
else:
    raise ValueError("CSV must contain either columns 'v1', 'v2' or 'label', 'message'")

# Drop rows with missing values
df.dropna(inplace=True)

# Map 'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Check class balance
print("Label counts:\n", df['label'].value_counts())
print("Total rows after cleaning:", len(df))

# Validate if we have enough data
if len(df) < 10 or df['label'].nunique() < 2:
    raise ValueError("Not enough labeled data. Make sure you have both 'spam' and 'ham' messages.")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Build a pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(model.named_steps['tfidf'], open('vectorizer.pkl', 'wb'))

print("Model training complete. Saved as model.pkl and vectorizer.pkl")
