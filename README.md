# Low-Resource Feedback Aspect-Based Sentiment Analysis

This repository contains the experimental notebooks for my MSc Data Science and Machine Learning dissertation at University College London (UCL), completed in December 2023.

The project investigates how training-data volume and text augmentation affect multi-label aspect-based sentiment classification of customer feedback. Each feedback item can be assigned one or more aspect–sentiment labels, combining a business aspect (for example, app/website experience, value for money, or staff attitude) with positive, neutral, or negative sentiment.

## Research scope

The experiments cover three connected questions:

1. How well can customer-feedback aspects and sentiment be classified when only a small fraction of labelled training data is available?
2. Can text augmentation improve performance in low-resource settings?
3. How do classical classifiers using transformer embeddings compare with fine-tuned transformer models?

Training subsets at 1%, 5%, 10%, 20%, and 30% of the original data are compared with the full training set. The augmentation experiments use:

- Easy Data Augmentation (EDA)
- English–Spanish–English back-translation
- English–Korean–English back-translation

## Models and evaluation

Two modelling approaches are explored:

- **Transformer embeddings + classical classifiers:** BERT and RoBERTa sentence embeddings with logistic regression, support vector machines, and random forests
- **End-to-end transformer fine-tuning:** BERT, RoBERTa, and XLNet in base and large variants

Performance is evaluated primarily with F1-based metrics across data sizes, augmentation methods, aspect categories, and sentiment labels.

## Repository structure

| Stage | Notebook(s) | Purpose |
| --- | --- | --- |
| 1. Data preparation | `Task1_1_Data_exploration_and_subsampling.ipynb` | Explore the feedback dataset, create train/validation splits, and generate stratified training subsets |
| 1. Data augmentation | `Task1_2. EDA.ipynb` | Generate EDA-augmented samples |
| 1. Data augmentation | `Task1_3. Backtranslation.ipynb` | Generate Spanish and Korean back-translated samples |
| 2. Representation | `Task2_1. Sentence_Embedding.ipynb` | Create and visualise BERT and RoBERTa sentence embeddings |
| 2. Baselines | `Task2_2. Baseline.ipynb` | Train classical multi-label classifiers on transformer embeddings |
| 3. Tuning | `Task3_1`–`Task3_4` notebooks | Tune BERT, RoBERTa, and XLNet configurations |
| 3. Fine-tuning | `Task3_5`–`Task3_10` notebooks | Fine-tune base and large transformer variants and compare results across data sizes |

## Running the notebooks

The notebooks were developed in Google Colab with GPU acceleration. They use Python and the following main libraries:

- PyTorch
- Hugging Face Transformers and Datasets
- pandas and NumPy
- scikit-learn and scikit-multilearn
- TextAttack
- Matplotlib and Seaborn
- Google Cloud Translation (for Korean back-translation)

Run the notebooks in task-number order. Before running them, update the Google Drive paths to match your environment and provide the required source and intermediate CSV files.

> **Data availability:** the customer-feedback dataset and generated CSV files are not included in this repository. The notebooks currently reference files under `/content/gdrive/My Drive/MSc Project/`.

## Author

Eunju Lee  
MSc Data Science and Machine Learning, University College London

