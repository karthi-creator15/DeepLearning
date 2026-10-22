Deep Learning + NLP


## Requirements

Python 3.10–3.12 is recommended. Some PyTorch/Transformers packages may not yet support every newest Python release.

Install dependencies for each task using its requirements file. Run each task from its own folder so generated model files and graphs are kept with that task.

### Task 1

```bash
cd task1_neural_network
pip install -r requirements.txt
python main.py
```

Trains a 2-4-1 XOR network, tests activation/loss functions, confirms gradients with PyTorch autograd, and saves loss plots.

### Task 2

```bash
cd task2_pytorch_pipeline
pip install -r requirements.txt
python main.py
```

Uses scikit-learn's breast cancer dataset, StandardScaler, Dataset/DataLoader, an MLP, early stopping, state-dict saving/reloading, and logistic regression comparison.

### Task 3

```bash
cd task3_nlp_baseline
pip install -r requirements.txt
python main.py
```

Builds a synthetic labelled 360-text sentiment dataset, trains TF-IDF + Logistic Regression, prints metrics and class-specific words, and saves the pipeline. GloVe vectors are downloaded by Gensim on first use and require internet access.

### Task 4

```bash
cd task4_transformers_api
pip install -r requirements.txt
python main.py
```

Displays tokenizer examples, runs sentiment and NER pipelines, fine-tunes DistilBERT for two epochs, evaluates against the TF-IDF baseline, and saves the transformer model.

Then start the API from the `task4_transformers_api` folder:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Test health:

```bash
curl http://127.0.0.1:8000/health
```

Test prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I was charged twice and support was unhelpful\"}"
```

The API returns a label and confidence. Empty or whitespace-only text returns HTTP 422. The first transformer run downloads model files and may take time; internet access and adequate RAM are required.

## Notes

- The generated sentiment dataset is intended for demonstrating the pipeline, not for claiming real-world model quality.
- Scores vary by environment and training run. Do not hardcode the example assessment metrics.
- GloVe and Hugging Face model downloads require internet access.
- Model files and local environments should not be committed unless required by your trainer.

## GitHub

From the project root:

```bash
git init
git add .
git commit -m "Add Week 3 deep learning and NLP assessment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/week3-github-project.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository URL.
