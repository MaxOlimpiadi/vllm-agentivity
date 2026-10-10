# Agentivity Annotation with vLLM

A research project using a local language model to identify **participants in literary events** and classify their **agentivity**.

Inference runs through **vLLM**, with JSON-schema-constrained responses.

**Method**

The pipeline processes each target phrase in two stages:

1. **Participant identification:** select involved characters from the document’s character list.
2. **Agentivity classification:** assign each predicted participant one label:
   - `agentive`
   - `low_agentive`
   - `passive`

Prompts include surrounding context and annotation guidelines. Character mentions are highlighted using the supplied reference annotations.

**Project Structure**

```text
main.py                 # Data preparation, inference, and logging
binary_accuracy.py      # Evaluation of predicted character sets by category
data/
  Krambambuli.json       # Annotated input document
logs/
  char_lists_logs/      # Participant predictions and prompts
  labels_logs/          # Agentivity predictions
output.json             # Automatically generated intermediate file
```

**Input Data**

Input JSON documents must contain:

- `title` and `text`
- `sentences`: character-offset spans
- `mentions`: character mentions grouped by annotator
- `participations`: event annotations grouped by annotator

The default annotator is `gold`. Each participation record contains `spans` and character lists for `agentive`, `low_agentive`, and `passive`.

Place additional documents with the same structure in `data/`. Plain text alone is not sufficient: the pipeline uses annotated character mentions and reference participation labels.

**Requirements**

Use a Linux environment supported by vLLM.

Main dependencies:

```bash
pip install vllm openai "pydantic>=2" scikit-learn matplotlib
```

**Configuration**

The current model settings in `main.py` are:

```python
llm = LLM(
    model="google/gemma-4-31B-it",
    tensor_parallel_size=2,
    max_model_len=16384,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 0, "audio": 0},
)
```

This configuration requires **two compatible GPUs** with sufficient memory. 

Other parameters:

- `author = "gold"` — reference annotator.
- `context_sent_size = 10` — context-window setting.
- `N = 3000` — maximum eligible participation records per document.
- `temperature = 0` — generation temperature.
- `max_tokens = 1024` — maximum response length.

Participation records near document boundaries are excluded to allow surrounding context.

**Running**

Run from the project root:

```bash
python main.py
```

The script prepares prompts, identifies participants, and predicts their agentivity labels. Requests run locally through vLLM.

**Outputs**

- `logs/char_lists_logs/machine_log.jsonl` — participant predictions and reference sets.
- `logs/char_lists_logs/full_text_log.txt` — participant prompts and context.
- `logs/labels_logs/machine_log.jsonl` — reference and predicted character sets for each category.
- `logs/auxiliary_prompt_data.json` — prepared prompt inputs.
- `output.json` — transformed annotations for the most recently processed document.

Both stages also write `pretty_log.json` for manual inspection.

**Each run clears existing files in the two log directories.** The intermediate `output.json` is generated automatically and does not need to exist beforehand.

**Evaluation**

After inference, run:

```bash
python binary_accuracy.py
```

The script reports **exact-match accuracy** for each document and agentivity category.
