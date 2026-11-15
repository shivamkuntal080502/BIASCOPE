# BIASCOPE

**A Multilingual and Code-Mixed Explainable News Bias Detection System using Parameter-Efficient Fine-Tuning and Large Language Models**

BIASCOPE is an experimental research/demo system for analysing linguistic framing and potential bias signals in English and Hindi news text. Its classifier produces the bias score; the LLM layer is used for explanation and rewriting, not as the source of the classifier verdict.

> **Important:** BIASCOPE is not a fact-checker and does not determine whether a claim is true or false. A bias score is a model-estimated linguistic signal, not a judgment about the truthfulness of an article, publisher, or person. Outputs should be reviewed by a human.

## Repository contents

```text
BIASCOPE/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_environment_and_baseline.ipynb
│   ├── 02_dataset_preparation_and_eda.ipynb
│   ├── 03_baselines_and_lora.ipynb
│   ├── 04_multilingual_dataset_preparation.ipynb
│   ├── 05_hindi_hinglish_annotation_pilot.ipynb
│   ├── 06_mbib_hindi_translation_quality.ipynb
│   ├── 07_multilingual_models_xlmr_muril.ipynb
│   ├── 08_calibration_explainability_robustness.ipynb
│   ├── 09_llm_explanation_and_rewriting.ipynb
│   ├── 10_interactive_gradio_demo.ipynb
│   └── 11_uncertainty_and_crosslingual_consistency.ipynb
└── src/
    └── biascope_core.py
```

Notebook outputs were cleared in this GitHub copy. The code and markdown cells are retained. This keeps the repository lighter and avoids accidentally publishing runtime output or credentials.

## Main components

- **Data preparation:** MBIB-based English data preparation, exploratory analysis and train/validation/test splits.
- **Baselines:** TF-IDF/logistic-regression and transformer baselines, including RoBERTa fine-tuning and LoRA experiments.
- **Multilingual preparation:** HINMIX language-resource analysis, Hindi translation workflow and translation-quality heuristics.
- **Multilingual modelling:** XLM-R and MuRIL experiments, with MBIB/BABE-based training and evaluation.
- **Calibration and explainability:** temperature scaling, calibration metrics, Integrated Gradients and robustness checks.
- **LLM layer:** classifier-grounded explanations and counterfactual/neutral rewriting; local Qwen model is the configured demo backend, with optional Gemini code retained in the research notebook.
- **Demo:** Gradio interface with analysis, source comparison, cross-lingual checks, related-coverage retrieval and report export.
- **Uncertainty:** ensemble disagreement, uncertainty-aware handling and cross-lingual consistency experiments.

## Recommended execution order

Run the notebooks in numeric order in **Google Colab**. Some stages depend on files and model adapters created by earlier notebooks.

1. `01_environment_and_baseline.ipynb`
2. `02_dataset_preparation_and_eda.ipynb`
3. `03_baselines_and_lora.ipynb`
4. `04_multilingual_dataset_preparation.ipynb`
5. `05_hindi_hinglish_annotation_pilot.ipynb`
6. `06_mbib_hindi_translation_quality.ipynb`
7. `07_multilingual_models_xlmr_muril.ipynb`
8. `08_calibration_explainability_robustness.ipynb`
9. `09_llm_explanation_and_rewriting.ipynb`
10. `11_uncertainty_and_crosslingual_consistency.ipynb`
11. `10_interactive_gradio_demo.ipynb`

NB11 should be run before NB10 because the demo reads the uncertainty configuration produced by NB11. NB09 creates `src/biascope_core.py`; that source file is also included in this repository.

## Quick start (Google Colab)

1. Download or clone this repository.
2. Open the notebooks folder in GitHub and open a notebook with **Open in Colab** (or upload the `.ipynb` file to Colab).
3. In Colab, select **Runtime → Change runtime type → T4 GPU** for model-training and local Qwen inference stages. CPU may work for data preparation and some analysis, but large-model stages require substantial RAM/VRAM.
4. Mount Google Drive when prompted. The notebooks expect the project workspace at:

   `/content/drive/MyDrive/BIASCOPE`

5. Create the directory structure and place the required datasets/checkpoints there, or run the earlier notebooks to generate them. Later notebooks expect paths such as `data/processed/`, `data/multilingual/`, `models/final/` and `outputs/`.
6. Run the notebooks in the order above. Do not skip prerequisite stages unless you already have their expected files.

### Demo prerequisites

The Gradio demo notebook expects:

- `src/biascope_core.py`
- `models/final/xlmr_lora_mbib_babe_s42/`
- `models/final/xlmr_lora_mbib_babe_s43/`
- `models/final/xlmr_lora_mbib_babe_s44/`
- `models/final/uncertainty_config.json`
- a working GPU runtime for the configured local Qwen3-8B backend (or adjust the model/backend configuration in the notebook if using a different supported setup)

The trained model adapters, datasets, generated caches and outputs are **not** bundled in this ZIP. They are large and may have separate dataset/model licence conditions. Generate them by running the notebooks or manage them separately in Drive/model hosting.

## Dependencies

The `requirements.txt` lists the main Python packages used across the notebooks. Google Colab provides some packages (including PyTorch) and the notebooks also contain stage-specific installation cells. For reproducibility, use a fresh Colab runtime and follow the install cells in each notebook. Package versions evolve; this repository does not claim a fully pinned environment.

## Data and model notes

- MBIB and BABE are third-party datasets; consult their original dataset cards/papers and follow their terms of use.
- HINMIX is used as a language/code-mixing resource, not as bias ground truth.
- Hindi translations of English labels inherit the source labels; they are not independently human-annotated labels.
- LLM-generated labels, where used for exploratory evaluation, must not be described as human gold annotations.
- Romanized Hinglish is a known weak area in the experiments. Treat such predictions cautiously.
- Related-news retrieval is a relevance aid, not independent verification or fact-checking.

## Security before publishing

- Never commit API keys, `.env` files, Hugging Face tokens, Google credentials, private datasets or personal Drive files.
- Use Colab Secrets or environment variables for API credentials. Do not hard-code keys in notebooks.
- Review notebook cells and GitHub's secret-scanning warnings before making the repository public.
- The `.gitignore` excludes datasets, checkpoints, caches and generated outputs by default.

## Citation and acknowledgement

If you use BIASCOPE or adapt its code, cite/acknowledge the project and the original datasets, pretrained models and libraries used. Add the final project citation/author metadata here before publication if required by your department.

## Limitations

This is a research prototype. Performance depends on training data, language/script, domain and text length. Sentence-level training and article-level aggregation may introduce errors. Calibration and ensemble agreement do not guarantee correctness. Explanations and highlighted tokens are model interpretations, not causal proof of bias. Human review remains necessary.
