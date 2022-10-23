"""DiffCSE Utils"""

from random import choice
from pathlib import Path

import torch
from tqdm import tqdm
import yaml


def load_multinli_data(file_path: str, is_sup: bool = True, encoding: str = "utf8") -> tuple:
    with Path(file_path).open("r", encoding=encoding) as fp:
        raw_data = fp.readlines()[1:]
    data = []
    query, pos, neg = [], [], []
    for idx, d in tqdm(enumerate(raw_data), total=len(raw_data), desc="multinli dataset"):
        row = d.split("\t")
        if is_sup:
            if row[-1].strip() == "contradiction":
                query.append(row[0])
                pos.append(row[0])
                neg.append(row[1])
            elif row[-1].strip() == "entailment" and idx >= 10:
                query.append(row[0])
                pos.append(row[1])
                choiced_sent = choice(query)
                while choiced_sent == row[0]:
                    choiced_sent = choice(query)
                neg.append(choiced_sent)
                
                query.append(row[0])
                pos.append(row[0])
                neg.append(choiced_sent)
            else:
                continue
        else:
            data.append(row[0])
    if is_sup:
        for q, p, n in zip(query, pos, neg):
            data.append([q, p, n])
    return tuple(data)


def load_snli_data(file_path: str, is_sup: bool = True, encoding: str = "utf8") -> tuple:
    with Path(file_path).open("r", encoding=encoding) as fp:
        raw_data = fp.readlines()[1:]
    data = []
    query, pos, neg = [], [], []
    for idx, d in tqdm(enumerate(raw_data), total=len(raw_data), desc="snli dataset"):
        row = d.split("\t")
        if row[-1].strip() == "contradiction":
            if is_sup:
                if row[-1].strip() == "contradiction":
                    query.append(row[0])
                    pos.append(row[0])
                    neg.append(row[1])
                elif row[-1].strip() == "entailment" and idx >= 4:
                    query.append(row[0])
                    pos.append(row[1])
                    choiced_sent = choice(query)
                    while choiced_sent == row[0]:
                        choiced_sent = choice(query)
                    neg.append(choiced_sent)
                    
                    query.append(row[0])
                    pos.append(row[0])
                    neg.append(choiced_sent)
                else:
                    continue
            else:
                data.append(row[0])
    if is_sup:
        for q, p, n in zip(query, pos, neg):
            data.append([q, p, n])
    return tuple(data)


def get_sts_dataset(file_path: str, encoding: str = "utf8") -> tuple:
    with Path(file_path).open("r", encoding=encoding) as fp:
        raw_data = fp.readlines()[1:]
    data = []
    for d in raw_data:
        row = d.split("\t")
        data.append((float(row[4]), row[5], row[6].strip()))
    return tuple(data)


def get_config(file_path: str, tag: str, encoding: str = "utf8") -> dict:
    with Path(file_path).open("r", encoding=encoding) as fp:
        config = yaml.load(stream=fp, Loader=yaml.FullLoader)
    return config.get("SimCSE").get(tag)


def contrastive_loss(
    q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, temperature: float = 0.05
    ) -> torch.Tensor:
    q_pos_sim = torch.cosine_similarity(q, pos) / temperature
    q_neg_sim = torch.cosine_similarity(q, neg) / temperature
    denominator = torch.exp(q_pos_sim)
    numerator = torch.sum(torch.exp(q_pos_sim) + torch.exp(q_neg_sim))
    return torch.mean(-torch.log(denominator / numerator))
