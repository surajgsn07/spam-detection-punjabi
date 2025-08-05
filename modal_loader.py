import pickle

def load_model():
    try:
        # Load the pre-trained vectorizer and model for English
        vectorizer = pickle.load(open('vectorizer_pa.pkl', 'rb'))
        model = pickle.load(open('spam_df_pa.pkl', 'rb'))

        return vectorizer, model

    except Exception as e:
        print(f"[ERROR] Failed to load English model/vectorizer: {e}")
        raise
