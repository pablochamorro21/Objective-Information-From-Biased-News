# News Bias Reduction Pipeline

This repository provides a complete pipeline for mitigating political bias in Spanish news articles. It consists of web scrapers to collect articles, a matching script to align content from two sources, a classifier trained to detect bias, and reinforcement-learning-based generative models that use the classifier as a reward signal to rewrite articles in a neutral style.

## Scripts Overview

### 1. `scraper_elmundo.py`
Scrapes El Mundo’s historical archive (“hemeroteca”) by date. It:
- Navigates to the archive page for a given date.
- Filters articles in the “Política” section.
- Opens each article in a new tab, scrolls to load full content, and extracts all paragraphs.
- Returns a list of dictionaries with `Title`, `Body_Text`, and assigned `Date`.

### 2. `scraper_elpais.py`
Scrapes El País’s archive by date. It:
- Iterates over each date in a specified range (from 2024 to today).
- Visits the homepage for that date’s archive.
- Locates article links on the front page.
- Opens each in a new tab, accepts cookies if needed, and extracts body paragraphs.
- Compiles a list of `{Title, Body_Text}` for each day, then saves all to `El_Pais_news.csv`.

### 3. `all_similarity.py`
Aligns El Mundo and El País articles by date and content similarity. It:
- Loads the two CSVs produced by the scrapers.
- Preprocesses text via spaCy lemmatization and stopword removal.
- Encodes with BETO (Spanish BERT) and computes cosine similarity.
- For each El Mundo article, finds the best-matching El País article on the same date.
- Outputs a CSV of matched pairs with similarity scores.

### 4. `classifier.py`
Trains a bias-detection classifier on the matched dataset and/or fine-tunes a generative model with RL. It:
- Loads matched article pairs and assigns labels (`1` for El Mundo/right-wing, `0` for El País/left-wing).
- Preprocesses text and computes embeddings with a Spanish RoBERTa model.
- Trains a BiLSTM classifier to predict bias probability.
- This classifier is later used to provide reward signals for a generative LLaMA-2 model in later scripts.

### 5. `gen_model_single.py`
Performs reinforcement-learning fine-tuning of a LLaMA-2 LoRA-adapted chat model. It:
- Uses the trained BiLSTM classifier as a reward model (higher reward for reduced bias).
- Defines prompts to extract article by article.
- Computes policy gradients to encourage neutral, concise rewriting (~10% shorter).
- Logs reward, bias reduction, and loss history.
- Saves the fine-tuned generative model for inference.

### 6. `gen_model_pairs.py`
Performs reinforcement-learning fine-tuning of a LLaMA-2 LoRA-adapted chat model. It:
- Uses the trained BiLSTM classifier as a reward model (higher reward for reduced bias).
- Defines prompts to extract and then fuse summaries of paired articles.
- Computes policy gradients to encourage neutral, concise rewriting (~10% shorter).
- Logs reward, bias reduction, and loss history.
- Saves the fine-tuned generative model for inference.

### 7. `gen_model_testing.py`
Evaluates the bias-reduction performance on a held-out test set. It:
- Loads the fine-tuned LLaMA-2 LoRA model and the bias classifier.
- Iterates over test article pairs by date.
- Generates fused articles via extraction and fusion prompts.
- Computes original and rewritten bias scores.
- Records results and computes overall average bias reduction.

## Workflow Summary

1. **Data Collection**:  
   - `scraper_elmundo.py` gathers El Mundo politics articles.  
   - `scraper_elpais.py` gathers El País front-page articles.  

2. **Article Matching**:  
   - `all_similarity.py` pairs articles from both sources by date and semantic similarity.  

3. **Classifier Training**:  
   - `classifier.py` trains a BiLSTM classifier to detect political bias.  

4. **Generative RL Fine-Tuning**:  
   - `gen_model_single.py` & `gen_model_pairs.py` uses the classifier’s output as a reward to fine-tune a LLaMA-2 model, rewriting articles neutrally.  

5. **Evaluation**:  
   - `gen_model_testing.py` measures bias reduction on a test set to validate the pipeline.

By following these steps, this pipeline scrapes real-world news data, aligns comparable content, learns to detect bias, and teaches a generative model to produce more neutral news articles. 


By Pablo Chamorro Casero