---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:6766
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Lactobionic Acid
  sentences:
  - Purified Water
  - Lactobionic Acid
  - Acetyl Octapeptide
- source_sentence: Prifrac 2960
  sentences:
  - n-Octadecanoic Acid
  - Syn-Ake
  - Beeswax
- source_sentence: Flaxseed Oil
  sentences:
  - Linseed Oil
  - Granactive Retinoid
  - Adenosine
- source_sentence: Yeast Ferment
  sentences:
  - D-Panthenol
  - Fermented Yeast Extract
  - Evonik Chamomile
- source_sentence: Phytosphingosine
  sentences:
  - Ascorbic Acid
  - Pomegranate Extract
  - Tetrahydrosphingosine
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@1
- cosine_ndcg@5
- cosine_mrr@1
- cosine_mrr@5
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: inci eval
      type: inci_eval
    metrics:
    - type: cosine_accuracy@1
      value: 0.8
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.975
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.975
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.975
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.8
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.325
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.195
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.0975
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.8
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.975
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.975
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.975
      name: Cosine Recall@10
    - type: cosine_ndcg@1
      value: 0.8
      name: Cosine Ndcg@1
    - type: cosine_ndcg@5
      value: 0.9071394630357187
      name: Cosine Ndcg@5
    - type: cosine_mrr@1
      value: 0.8
      name: Cosine Mrr@1
    - type: cosine_mrr@5
      value: 0.8833333333333332
      name: Cosine Mrr@5
    - type: cosine_map@100
      value: 0.8845238095238095
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 64 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 64, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Phytosphingosine',
    'Tetrahydrosphingosine',
    'Ascorbic Acid',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6606, 0.3366],
#         [0.6606, 1.0000, 0.3018],
#         [0.3366, 0.3018, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `inci_eval`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.8        |
| cosine_accuracy@3   | 0.975      |
| cosine_accuracy@5   | 0.975      |
| cosine_accuracy@10  | 0.975      |
| cosine_precision@1  | 0.8        |
| cosine_precision@3  | 0.325      |
| cosine_precision@5  | 0.195      |
| cosine_precision@10 | 0.0975     |
| cosine_recall@1     | 0.8        |
| cosine_recall@3     | 0.975      |
| cosine_recall@5     | 0.975      |
| cosine_recall@10    | 0.975      |
| cosine_ndcg@1       | 0.8        |
| **cosine_ndcg@5**   | **0.9071** |
| cosine_mrr@1        | 0.8        |
| cosine_mrr@5        | 0.8833     |
| cosine_map@100      | 0.8845     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 6,766 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                       | sentence_1                                                                       |
  |:--------|:---------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                           |
  | details | <ul><li>min: 3 tokens</li><li>mean: 7.65 tokens</li><li>max: 28 tokens</li></ul> | <ul><li>min: 3 tokens</li><li>mean: 6.96 tokens</li><li>max: 32 tokens</li></ul> |
* Samples:
  | sentence_0                   | sentence_1                             |
  |:-----------------------------|:---------------------------------------|
  | <code>Petroleum Jelly</code> | <code>Unipetrol</code>                 |
  | <code>Arlasolve DMI</code>   | <code>Isosorbide Dimethyl Ether</code> |
  | <code>Curcuminoid</code>     | <code>Turmeric Extract</code>          |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 64
- `num_train_epochs`: 5
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 64
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 64
- `num_train_epochs`: 5
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 64
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | inci_eval_cosine_ndcg@5 |
|:------:|:----:|:-----------------------:|
| 0.4717 | 50   | 0.6193                  |
| 0.9434 | 100  | 0.7105                  |
| 1.0    | 106  | 0.7112                  |
| 1.4151 | 150  | 0.7572                  |
| 1.8868 | 200  | 0.7991                  |
| 2.0    | 212  | 0.8008                  |
| 2.3585 | 250  | 0.8254                  |
| 2.8302 | 300  | 0.8603                  |
| 3.0    | 318  | 0.8631                  |
| 3.3019 | 350  | 0.8696                  |
| 3.7736 | 400  | 0.8854                  |
| 4.0    | 424  | 0.9071                  |


### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.2.3
- Transformers: 5.2.0
- PyTorch: 2.10.0+cpu
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->