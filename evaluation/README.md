## Running Ollama

To use the `llama3:latest` model with Ollama, follow these steps:

1. Pull the `llama3:latest` model:

   ```bash
   ollama pull llama3:latest
   ```

2. Start the Ollama server:

   ```bash
   ollama serve
   ```

## Running the Evaluation

To run the evaluation, execute the following command:

```bash
python evaluation/run_evaluation.py
```

### How Results Are Saved
- The evaluation results are saved incrementally in the CSV file located at:
  `evaluation/evaluation_results/evaluation_results.csv`
- After each instance is evaluated, its results are appended to the CSV file.
- The summary of the evaluation is updated in the JSON file located at:
  `evaluation/evaluation_results/summary.json`

This ensures that if the evaluation is interrupted, already computed results are not lost, and the process can resume without re-evaluating completed instances.
