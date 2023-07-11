"""
Fine-tuning de los modelos de HuggingFace para token classification.
"""
# Archivo adaptado de https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py 
# (5 de julio de 2023)
# Preparado para correr con Python 3.10

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.30.2")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

# Set variables

# General
account_hf = "jpherrerap"
task_name = "ner"
last_checkpoint = None # Directorio del último checkpoint si quiero retomar un entrenamiento
use_auth_token = True # Para usar datasets privados de HuggingFace y subir el modelo a HuggingFace Hub
push_to_hub = True # Para subir el modelo a HuggingFace Hub
do_train = True
do_eval = True
do_predict = False
output_predict_file = "predictions.txt"
evaluate_metric = "overall_f1"
resume_from_checkpoint = None
language = "es"
overwrite_output_dir = True
seed = 13
preprocessing_num_workers = 4

# Datasets variables
dataset_name = f"{account_hf}/competencia2"
text_column_name = "text"
label_column_name = "nertags"
dataset_config_name = None
pad_to_max_length = True
max_train_samples = None # Por ej 1000 si no quiero entrenar con todo el dataset de training (sirve para pruebas)
max_eval_samples = None # si no quiero evaluar con todo el dataset de evaluación
max_predict_samples = None # si no quiero predecir con todo el dataset de testing

# Task and models
label_all_tokens = False
return_entity_level_metrics = True
learning_rate = 2e-5
lr_scheduler_type = "linear"
auto_find_batch_size = True
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
max_seq_length = 512
optim = "adamw_torch" #"adamw_torch"
weight_decay = 0.01
num_train_epochs = 2
save_total_limit = 2
is_regression = False
model_name_or_path = "lcampillos/roberta-es-clinical-trials-ner"
use_fast_tokenizer = True
ignore_mismatched_sizes = True
fp16 = True
output_dir = f"models/{task_name}-{model_name_or_path.split('/')[-1]}"
evaluation_strategy = "epoch" # or "steps"
save_strategy = "epoch" # or "steps"
load_best_model_at_end = True
save_safetensors = True




# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)



# Detecting last checkpoint.
last_checkpoint = None
if os.path.isdir(output_dir) and do_train and not overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
        raise ValueError(
            f"Output directory ({output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif last_checkpoint is not None and resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# Set seed before initializing model.
set_seed(seed)

# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub).
#
# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
# 'text' is found. You can easily tweak this behavior (see below).
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
if dataset_name is not None:
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        dataset_name,
        dataset_config_name,
        use_auth_token=True if use_auth_token else None,
    )
# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

if do_train:
    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features
else:
    column_names = raw_datasets["validation"].column_names
    features = raw_datasets["validation"].features

if text_column_name is not None:
    text_column_name = text_column_name
elif "tokens" in column_names:
    text_column_name = "tokens"
else:
    text_column_name = column_names[0]

if label_column_name is not None:
    label_column_name = label_column_name
elif f"{task_name}_tags" in column_names:
    label_column_name = f"{task_name}_tags"
else:
    label_column_name = column_names[1]

# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
# Otherwise, we have to get the list of labels manually.
labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
if labels_are_int:
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

num_labels = len(label_list)

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    finetuning_task=task_name,
    use_auth_token=True if use_auth_token else None,
)

if config.model_type in {"bloom", "gpt2", "roberta"}:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        use_auth_token=True if use_auth_token else None,
        add_prefix_space=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        use_auth_token=True if use_auth_token else None,
    )

model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    use_auth_token=True if use_auth_token else None,
    ignore_mismatched_sizes=ignore_mismatched_sizes,
)

# Tokenizer check: this script requires a fast tokenizer.
if not isinstance(tokenizer, PreTrainedTokenizerFast):
    raise ValueError(
        "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
        " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
        " this requirement"
    )

# Model has labels -> use them.
if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
    if sorted(model.config.label2id.keys()) == sorted(label_list):
        # Reorganize `label_list` to match the ordering of the model.
        if labels_are_int:
            label_to_id = {i: int(model.config.label2id[l]) for i, l in enumerate(label_list)}
            label_list = [model.config.id2label[i] for i in range(num_labels)]
        else:
            label_list = [model.config.id2label[i] for i in range(num_labels)]
            label_to_id = {l: i for i, l in enumerate(label_list)}
    else:
        logger.warning(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {sorted(model.config.label2id.keys())}, dataset labels:"
            f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
        )

# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = dict(enumerate(label_list))

# Map that sends B-Xxx label to its I-Xxx counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

# Preprocessing the dataset
# Padding strategy
padding = "max_length" if pad_to_max_length else False

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            #elif word_idx != previous_word_idx:
            #    label_ids.append(label_to_id[label[word_idx]])
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if do_train:
    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if max_train_samples is not None:
        max_train_samples = min(len(train_dataset), max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    train_dataset = train_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc="Running tokenizer on train dataset",
    )

if do_eval:
    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    if max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    eval_dataset = eval_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc="Running tokenizer on validation dataset",
    )

if do_predict:
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]
    if max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))
    predict_dataset = predict_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=preprocessing_num_workers,
        desc="Running tokenizer on prediction dataset",
    )

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if fp16 else None)

# Metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

training_args = TrainingArguments(output_dir=output_dir, 
                                  overwrite_output_dir=overwrite_output_dir,
                                    evaluation_strategy=evaluation_strategy, 
                                    save_strategy=save_strategy,
                                    load_best_model_at_end=load_best_model_at_end,
                                    learning_rate=learning_rate, 
                                    lr_scheduler_type=lr_scheduler_type,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    per_device_eval_batch_size=per_device_eval_batch_size,
                                    auto_find_batch_size=True,
                                    num_train_epochs=num_train_epochs, 
                                    weight_decay=weight_decay, 
                                    fp16=fp16,
                                    metric_for_best_model= f"eval_{evaluate_metric}",
                                    seed=seed,
                                    data_seed=seed,
                                    optim=optim,
                                    save_total_limit=save_total_limit,
                                    save_safetensors=save_safetensors,
                                    hub_strategy="end" if push_to_hub else None,
                                    push_to_hub = push_to_hub,
                                    logging_strategy = "epoch",
                                    )      

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if do_train else None,
    eval_dataset=eval_dataset if do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
if do_train:
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.save_model()  # Saves the tokenizer too for easy upload

    max_train_samples = (
        max_train_samples if max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluation
if do_eval:
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate()

    max_eval_samples = max_eval_samples if max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# Predict
if do_predict:
    logger.info("*** Predict ***")

    predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)

    # Save predictions
    output_predictions_file = os.path.join(output_dir, "predictions.txt")
    if trainer.is_world_process_zero():
        with open(output_predictions_file, "w") as writer:
            for prediction in true_predictions:
                writer.write(" ".join(prediction) + "\n")

kwargs = {"finetuned_from": model_name_or_path, "tasks": "token-classification"}
kwargs["language"] = language
if dataset_name is not None:
    kwargs["dataset_tags"] = dataset_name
    if dataset_config_name is not None:
        kwargs["dataset_args"] = dataset_config_name
        kwargs["dataset"] = f"{dataset_name} {dataset_config_name}"
    else:
        kwargs["dataset"] = dataset_name

if push_to_hub:
    trainer.push_to_hub(**kwargs)
else:
    trainer.create_model_card(**kwargs)


