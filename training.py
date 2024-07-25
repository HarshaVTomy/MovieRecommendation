import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the dataset
file_path = 'imdb_top_movies.csv'
movies_df = pd.read_csv(file_path)

# Preprocess the data
X = movies_df[['Genre', 'Director', 'Cast', 'IMDB Rating']]
y = movies_df['Movie Name']

# One-hot encode the categorical features
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(X[['Genre', 'Director', 'Cast']])

# Combine the encoded features with the numerical feature
X_combined = pd.concat([pd.DataFrame(X['IMDB Rating']), pd.DataFrame(X_encoded.toarray())], axis=1)
X_combined.columns = X_combined.columns.astype(str)  # Ensure all column names are strings

# Initialize and fit the model
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X_combined, y)

# Save the model and the encoder using pickle
model_file_path = 'best_model.pkl'
encoder_file_path = 'encoder.pkl'
with open(model_file_path, 'wb') as model_file:
    pickle.dump(model, model_file)
with open(encoder_file_path, 'wb') as encoder_file:
    pickle.dump(ohe, encoder_file)
