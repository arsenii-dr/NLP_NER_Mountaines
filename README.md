# **NER for Mountain Entities**

## **Introduction**  
This project is a Named Entity Recognition (NER) system designed to identify entities related to mountains, such as mountain names, locations, and other related terms, from natural language text. It uses a fine-tuned BERT-based model to perform token classification for entity extraction.

## **Features**  
- Token classification for recognizing mountain-related entities.
- Fine-tuned pre-trained BERT model for enhanced performance.
- Standard metrics evaluation for model performance.

## **Dataset**  
- **Source**: 
```bash
   https://huggingface.co/datasets/telord/mountains-ner-dataset
```
- **Structure**:  
  - **Columns**:  
    - `sentence`: Original sentence.  
    - `tokens`: Tokenized words from the sentence.  
    - `labels`: Entity labels for each token in numeric format.  
  - **Preprocessing**:  
    - Converted labels to token-level annotations compatible with BERT.  
    - Handled alignment between tokenized words and original labels.  

## **Model and Architecture**  
- Pre-trained **DistilBERT** (`distilbert-base-uncased`) model fine-tuned for mountain token classification.  
- Supports token alignment for subword tokenization during training and inference.  
- Configured for 3 entity types: `['O', 'B-MNTN', 'I-MNTN']`.

## **Project Structure**
- __data/__
    - __tokenized_dataset/__ - Preprocessed and tokenized dataset used for training and evaluation. Contains train, validation and test subsets.
- __logs/__ - Logs created during the training.
- __models__/
    - __fine_tuned_ner_mountains_model/__ - Directory containing the fine-tuned model and tokenizer.
- __notebooks/__
    - __data_preprocessing.ipynb__ - Notebook for data preprocessing and exploration.
- __results/__ - Dir for a results (checkpoints) obtained during the training. P.S. The dir is empty in a cause of very big size of checkpoints.
- __src/__
    - __inference.py__ - Script for final evaluating using the fine-tuned BERT model and test dataset.
    - __train.py__ - Script for training the NER model.
- __venv/__
- __README.md__
- __requirements.txt__

## **Installation**  
Clone the repository:  
   ```bash
   git clone https://github.com/arsenii-dr/ner-mountain-entities.git
   cd ner-mountain-entities
```
