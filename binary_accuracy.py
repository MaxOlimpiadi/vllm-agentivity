# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 01:02:19 2026

@author: zidan
"""
from collections import defaultdict
import json
from sklearn.metrics import accuracy_score


def main():
    logs = []
    with open('logs/labels_logs/machine_log.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            logs.append(json.loads(line))

    grouped_logs = defaultdict(list)
    for instance in logs:
        grouped_logs[instance["doc_title"]].append(instance)

    accuracy_by_doc_and_category = defaultdict(dict)

    for doc_title, doc_instances in grouped_logs.items():
        for category in ("agentive", "low_agentive", "passive"):
            gold_data = [
                instance["binarized_labels_data"][category]["gold"]
                for instance in doc_instances
            ]
            pred_data = [
                instance["binarized_labels_data"][category]["predicted"]
                for instance in doc_instances
            ]

            acc = accuracy_score(gold_data, pred_data)
            accuracy_by_doc_and_category[doc_title][category] = acc

    for doc_title, metrics in accuracy_by_doc_and_category.items():
        print(f"\nDocument: {doc_title}")
        for category, acc in metrics.items():
            print(f"  {category}: {acc:.4f}")


main()