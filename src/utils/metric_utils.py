"""Utility functions for computing metrics."""

import numpy as np

import torch
from torch.nn import CrossEntropyLoss
import numpy as np


def compute_metrics(eval_preds,
                    eval_labels,
                    pad_token_id,
                    last_n_tokens=1,
                    **kwargs):
  """Computes squence-level and token-level accuracy."""
  total_count, total_token_count = 0, 0
  correct_count, correct_token_count = 0, 0
  for eval_pred, eval_label in zip(eval_preds, eval_labels):
    actual_test_labels = eval_label[:, -last_n_tokens:]
    if len(eval_pred.shape) == 3:
      # eval_preds is in the form of logits.
      pred_test_labels = torch.argmax(eval_pred[:, -last_n_tokens:], dim=-1)
    else:
      # eval_preds is in the form of token ids.
      pred_test_labels = eval_pred[:, -last_n_tokens:]
    padding_tokens = torch.logical_or(actual_test_labels == pad_token_id,
                                      actual_test_labels < 0)
    match_tokens = actual_test_labels == pred_test_labels
    correct_labels = torch.logical_or(match_tokens, padding_tokens)
    total_count += len(correct_labels)
    correct_count += torch.all(correct_labels, axis=-1).float().sum().tolist()
    total_token_count += (~padding_tokens).float().sum().tolist()
    correct_token_count += (~padding_tokens &
                            match_tokens).float().sum().tolist()
  accuracy = round(correct_count / total_count, 2)
  token_accuracy = round(correct_token_count / total_token_count, 2)
  return {"accuracy": accuracy, "token_accuracy": token_accuracy}


def compute_cross_entropy_loss(logits, labels, pad_token_id, next_n_tokens=1):
  """Computes cross-entropy loss over the last n tokens."""
  vocab_size = logits.shape[-1]
  labels = labels.clone()
  shift_logits = logits[..., -next_n_tokens - 1:-1, :].contiguous()
  shift_labels = labels[..., -next_n_tokens:].contiguous()
  shift_logits = shift_logits.view(-1, vocab_size)
  shift_labels = shift_labels.view(-1)
  shift_labels = shift_labels.to(shift_logits.device)
  shift_labels[shift_labels == pad_token_id] = -100
  loss = CrossEntropyLoss()(shift_logits, shift_labels)
  return loss


def compute_disentangle_score(log_data,
                              attribute_to_iso_tasks,
                              attribute_to_cause_tasks):
  """Compute disentanglement score from iso/cause scores."""
  match_base = np.mean([
      np.mean([log_data[t]['metrics']['base_labels']['accuracy']
               for t in ts if t in log_data])
      for a, ts in attribute_to_iso_tasks.items()])
  match_source = np.mean([
      np.mean([log_data[t]['metrics']['labels']['accuracy']
               for t in ts if t in log_data])
      for a, ts in attribute_to_cause_tasks.items()])
  return {'disentangle': 0.5 * (match_base + match_source),
          'isolate': match_base,
          'cause': match_source}

def compute_disentangle_scores_possible_empties(log_data,
                              attribute_to_iso_tasks,
                              attribute_to_cause_tasks):
    """Compute disentanglement score from iso/cause scores.
    If we don't run with the full dataset, then some isolation tasks will be missing.
    This way we can still view disentanglement scores"""
    
    # Debug print for iso tasks
    print("Attribute to iso tasks:", attribute_to_iso_tasks)
    
    iso_scores = []
    for a, ts in attribute_to_iso_tasks.items():
        task_scores = [log_data[t]['metrics']['base_labels']['accuracy']
                       for t in ts if t in log_data]
        print(f"Iso scores for attribute {a}:", task_scores)
        if task_scores:
            iso_scores.append(np.mean(task_scores))
    
    print("All iso scores:", iso_scores)
    
    match_base = np.mean(iso_scores) if iso_scores else np.nan
    print("match_base:", match_base)
    
    # Similar debug prints for cause tasks
    print("Attribute to cause tasks:", attribute_to_cause_tasks)
    
    cause_scores = []
    for a, ts in attribute_to_cause_tasks.items():
        task_scores = [log_data[t]['metrics']['labels']['accuracy']
                       for t in ts if t in log_data]
        print(f"Cause scores for attribute {a}:", task_scores)
        if task_scores:
            cause_scores.append(np.mean(task_scores))
    
    print("All cause scores:", cause_scores)
    
    match_source = np.mean(cause_scores) if cause_scores else np.nan
    print("match_source:", match_source)
    
    return {'disentangle': 0.5 * (match_base + match_source),
            'isolate': match_base,
            'cause': match_source}