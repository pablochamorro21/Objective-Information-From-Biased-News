import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
from datetime import datetime
import spacy  
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm  
from concurrent.futures import ThreadPoolExecutor 
import os 


download("punkt")
download("stopwords")

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

max_workers = os.cpu_count()

try:
    stop_words = set(stopwords.words("spanish"))
except LookupError:
    print("Downloading Spanish stopwords...")
    download("stopwords")
    stop_words = set(stopwords.words("spanish"))

try:
    nlp = spacy.load("es_core_news_sm")
except:
    print("You need to install the spaCy Spanish model. Run: python -m spacy download es_core_news_sm")
    raise

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_text(text):
    """
    Lowercase, lemmatize, and remove stopwords from a given text string.

    Parameters
    ----------
    text : str
        The raw text to preprocess.

    Returns
    -------
    str
        A cleaned, lemmatized string with stopwords removed.
    """
    if not isinstance(text, str):
        return ""
    doc = nlp(text.lower())
    lemmatized_words = [token.lemma_ for token in doc if token.is_alpha]
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    return " ".join(filtered_words)

def compute_beto_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using BETO embeddings.

    Parameters
    ----------
    text1 : str
        The first text string.
    text2 : str
        The second text string.

    Returns
    -------
    float
        Cosine similarity score between the mean-pooled embeddings.
    """
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, padding=True)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings1 = model(**inputs1).last_hidden_state.mean(dim=1)
        embeddings2 = model(**inputs2).last_hidden_state.mean(dim=1)
    return torch.nn.functional.cosine_similarity(embeddings1, embeddings2).item()

def preprocess_with_progress(df, desc):
    """
    Apply `preprocess_text` to each row in a DataFrame, showing a progress bar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a "Body_Text" column to preprocess.
    desc : str
        Description label for the tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        The original DataFrame with a new "processed_text" column.
    """
    processed_texts = []
    with tqdm(total=len(df), desc=desc) as pbar:
        for _, row in df.iterrows():
            processed_texts.append(preprocess_text(row["Body_Text"]))
            pbar.update(1)
    df["processed_text"] = processed_texts
    return df

def process_articles_by_date(date, el_pais, el_mundo):
    """
    Find the most similar El País article for each El Mundo article on a given date.

    Parameters
    ----------
    date : datetime.date or str
        The date to filter articles by.
    el_pais : pd.DataFrame
        Preprocessed El País articles with "Date", "Title", "Body_Text", and "processed_text".
    el_mundo : pd.DataFrame
        Preprocessed El Mundo articles with "Date", "Title", "Body_Text", and "processed_text".

    Returns
    -------
    list of dict
        Each dict contains:
            - El_Mundo_date, El_Mundo_title, El_Mundo_bodytext
            - El_Pais_date, El_Pais_title, El_Pais_bodytext
            - similarity score
    """
    try:
        el_mundo["Date"] = pd.to_datetime(el_mundo["Date"]).dt.date
        el_pais["Date"] = pd.to_datetime(el_pais["Date"]).dt.date
        date = pd.to_datetime(date).date()

        relevant_el_mundo = el_mundo[el_mundo["Date"] == date]
        relevant_el_pais = el_pais[el_pais["Date"] == date]

        print(f"Processing date: {date} | Mundo articles: {len(relevant_el_mundo)} | Pais articles: {len(relevant_el_pais)}")

        if relevant_el_mundo.empty or relevant_el_pais.empty:
            return []

        results = []
        for _, mundo_row in relevant_el_mundo.iterrows():
            best_match = None
            highest_similarity = 0
            for _, pais_row in relevant_el_pais.iterrows():
                similarity = compute_beto_similarity(mundo_row["processed_text"], pais_row["processed_text"])
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = pais_row
            if best_match is not None:
                results.append({
                    "El_Mundo_date": mundo_row["Date"],
                    "El_Mundo_title": mundo_row["Title"],
                    "El_Mundo_bodytext": mundo_row["Body_Text"],
                    "El_Pais_bodytext": best_match["Body_Text"],
                    "El_Pais_date": best_match["Date"],
                    "El_Pais_title": best_match["Title"],
                    "similarity": highest_similarity
                })

        return results
    except Exception as e:
        print(f"Error processing date {date}: {e}")
        return []

def find_most_similar_articles(el_pais_file, el_mundo_file):
    """
    Load, preprocess, and find most similar article pairs between El País and El Mundo.

    Parameters
    ----------
    el_pais_file : str
        Path to the El País CSV file.
    el_mundo_file : str
        Path to the El Mundo CSV file.

    Side Effects
    ------------
    - Prints progress and results.
    - Saves the final DataFrame to
      "/Users/pablochamorro/Desktop/Coding/Thesis/most_similar_articles.csv".
    """
    el_pais = load_data(el_pais_file)
    el_mundo = load_data(el_mundo_file)

    print("\nStarting preprocessing!")
    el_pais = preprocess_with_progress(el_pais, "Processing El Pais articles")
    el_mundo = preprocess_with_progress(el_mundo, "Processing El Mundo articles")

    unique_dates = el_mundo["Date"].unique()

    print(f"\nProcessing articles by date in parallel ({max_workers} at a time)...")
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for date in unique_dates:
            future = executor.submit(process_articles_by_date, date, el_pais, el_mundo)
            futures.append(future)

            if len(futures) >= max_workers:
                for completed_future in futures[:]:
                    results.extend(completed_future.result())
                    futures.remove(completed_future)

        for future in futures:
            results.extend(future.result())

    results_df = pd.DataFrame(results)

    print("RESULTS")
    print(results_df)
    results_df.to_csv("/Users/pablochamorro/Desktop/Coding/Thesis/most_similar_articles.csv", index=False)
    print("Results saved to most_similar_articles.csv")

find_most_similar_articles(
    "/Users/pablochamorro/Desktop/Coding/Thesis/El_Pais_news.csv",
    "/Users/pablochamorro/Desktop/Coding/Thesis/El_Mundo_news.csv"
)
