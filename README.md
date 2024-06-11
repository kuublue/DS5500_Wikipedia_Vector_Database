# DS5500_Wikipedia_Vector_Database

## Project 1 - Wikipedia Vector Database
### wiki_vectors.ipynb contents
1. Setup functions necessary for the projects, including importing text from Wikipedia, Generates embedding using S-BERT, Creating FAISS index, searching FAISS given text
2. Testing the implemented functions
3. Fine-tuning S-BERT for Question-Answer Pair using GooAQ dataset. Test Performance (correlation, F1) using validation set.
4. Implementing Gemma-2b LLM using vector database search.
5. Run WebUI for LLM Chatbot

### Other Files
//train-00000-of-00002.parquet and train-00001-of-00002.parquet - GooAQ data for training Question-Answer Pair
//index.html - HTML file for webui
//outout folder - result of S-BERT training
