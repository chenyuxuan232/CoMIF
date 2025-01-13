from dataset.data import batch_to_device
from configs.configs import EvalConfig, DataConfig, ModelConfig
from utils.utils import DataLogger
from .utils import prepare_data, get_model

import torch

import os
import pandas as pd
from tqdm import tqdm
import math


EVAL_NUM = 1000


def do_evaluation(config: EvalConfig):

    # 计算predict文件路径
    predict_file_name = os.path.join(
        config.predict_dir, config.model_name, f"{config.dataset}-{config.epoch_idx}e.json"
    )
    # 计算输出路径
    output_dir = os.path.join(config.eval_output_dir, config.model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file_name = os.path.join(output_dir, f"{config.dataset}-{config.epoch_idx}e.txt")

    print("generate file dir: ", predict_file_name)
    print("eval     file dir: ", output_file_name)

    # 加载predict数据
    df = pd.read_json(predict_file_name, orient="records")
    context = df["context"].to_list()[:EVAL_NUM]  # list[list[str]]
    personas = df["personas"].to_list()[:EVAL_NUM]  # list[list[str]]
    response = df["response"].to_list()[:EVAL_NUM]  # list[str]
    target_response = df["target_response"].to_list()[:EVAL_NUM]  # list[str]
    all_row_datas = {
        "context": context,
        "personas": personas,
        "response": target_response,
    }

    # 333
    data_config = DataConfig(
        dataset=config.dataset, split="test", batch_size=config.batch_size, shuffle=False, all_raw_datas=all_row_datas
    )
    data_info = prepare_data(data_config)
    dataloader = data_info["dataloader"]
    # dataloader = None

    # 数据集中可能存在 target_response 为空的情况
    for i in range(len(target_response)):
        if target_response[i] == "":
            target_response[i] = response[i]

    # 222
    # 加载模型
    loadFilename = os.path.join(config.ckpt_dir, config.model_name, f"{config.dataset}-{config.epoch_idx}e.ckpt")
    ckpt = torch.load(loadFilename, map_location="cpu")
    model_config = ModelConfig(factor=config.factor)
    model = get_model(model_config, ckpt=ckpt, device=config.device)
    # model = None

    results = evaluation(model, dataloader, target_response, response, config.device, ablation=config.ablation)

    log_results(output_file_name, results)


def evaluation(model, dataloader, references, hypotheses, device, ablation):
    # breakpoint()
    results = {}
    # 111
    results["distinct_n"] = calc_distinct_ngram(hypotheses, 3)
    results["bleu_score"] = calc_bleu(references, hypotheses)
    results["rouge_score"] = calc_rouge(references, hypotheses)
    results["ppl"] = calc_perplexity(model, dataloader, device, ablation)

    results["em_f1_acc"] = calc_em_f1_score(references, hypotheses, device)

    return results


def log_results(output_file_name, results):
    with open(output_file_name, "w") as f:
        for k, v in results.items():
            print("=" * 30, file=f)
            print(f"{k}:", end=" ", file=f)
            print(v, file=f)


###########################################
# Distinct-n
###########################################
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


def preprocess_text(text):
    # # 转化为小写
    # text = text.lower()

    # # 删除数字与符号
    # text = re.sub(r"\W+|\d+", " ", text)

    # 分词
    words = word_tokenize(text)

    # # 去除停止词
    # stop_words = set(stopwords.words("english"))
    # words = [word for word in words if word not in stop_words]

    # # 词干提取
    # ps = PorterStemmer()
    # words = [ps.stem(word) for word in words]

    # # 词型还原
    # lemmatizer = WordNetLemmatizer()
    # words = [lemmatizer.lemmatize(word) for word in words]

    return words


def get_dict(tokens, ngram, gdict=None):
    """
    统计 n-gram 频率并用dict存储
    """
    token_dict = {}
    if gdict is not None:
        token_dict = gdict
    tlen = len(tokens)
    for i in range(0, tlen - ngram + 1):
        ngram_token = "".join(tokens[i : (i + ngram)])
        if token_dict.get(ngram_token) is not None:
            token_dict[ngram_token] += 1
        else:
            token_dict[ngram_token] = 1
    return token_dict


def calc_distinct_ngram(texts: list[str], max_ngram):
    dist_n = []
    for ngram in tqdm(range(1, max_ngram + 1), desc=f"Calc dist-n...", leave=False):
        ngram_total = 0.0
        ngram_distinct_count = 0.0
        pred_dict = {}
        for text in texts:
            predict_tokens = preprocess_text(text)
            pred_dict = get_dict(predict_tokens, ngram, pred_dict)
        for _, freq in pred_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
        dist_n.append(ngram_distinct_count / ngram_total)
    return dist_n


###########################################
# BLEU
###########################################
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def calc_bleu(references, hypotheses):
    """
    reference, prediction: list[str]
    """
    references = [[ref.split()] for ref in references]
    hypotheses = [pre.split() for pre in hypotheses]

    scores = []
    weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
    for w in tqdm(weights, desc="Calc bleu...", leave=False):
        scores.append(corpus_bleu(references, hypotheses, weights=w))
    return scores


###########################################
# Rouge
###########################################
from rouge import Rouge


def calc_rouge(references, hypotheses):
    rouge = Rouge()

    scores = rouge.get_scores(hypotheses, references, avg=True)
    return [
        scores["rouge-1"]["f"],
        scores["rouge-2"]["f"],
        scores["rouge-l"]["f"],
    ]


###########################################
# Perplexity
###########################################
def calc_perplexity(model, dataloader, device, ablation):

    # torch.cuda.empty_cache()
    model.train()  # 过程与训练一致，但无需计算梯度
    loss_logger = DataLogger()
    tbar = tqdm(dataloader, desc="Calc ppl...", leave=False)
    for batch in tbar:
        # ========== data to device ==========
        batch = batch_to_device(batch, device)

        context_ids = batch["context"]["input_ids"]
        context_attn_mask = batch["context"]["attention_mask"]

        persona_ids = batch["personas"]["input_ids"]
        persona_attn_mask = batch["personas"]["attention_mask"]

        dec_input_ids = batch["inputs"]["input_ids"]
        dec_attention_mask = batch["inputs"]["attention_mask"]
        dec_token_type_ids = batch["inputs"]["token_type_ids"]

        labels = batch["labels"]

        # 训练模式，但不计算梯度
        with torch.no_grad():
            # logits: (bsz, tgt_seq_len, vocab_size)
            outputs = model(
                context_ids=context_ids,
                context_attn_mask=context_attn_mask,
                persona_ids=persona_ids,
                persona_attn_mask=persona_attn_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_attention_mask,
                dec_token_type_ids=dec_token_type_ids,
                labels=labels,
                ablation_study=ablation,
            )
        reconstruction_loss = outputs["loss"]

        loss_logger.add(reconstruction_loss.cpu().item())

    return math.exp(loss_logger.avg_all())


###########################################
# Emotion_F1
###########################################
from transformers import pipeline
from configs.configs import emotion_label_id_map, emotion_threshold_dict, n_emotion_class
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score


def calc_em_f1_score(references, hypotheses, device):
    classifier = pipeline(
        task="text-classification",
        model="/home/cyx/workspace/ChatBot/model_hubs/roberta-base-go_emotions",
        top_k=None,
        device=device,
    )

    def get_em_ids_with_threshold(em_distribution) -> list[int]:
        em_ids = [0] * n_emotion_class
        for d in em_distribution:
            label = d["label"]
            score = d["score"]
            if score >= emotion_threshold_dict[label]:
                em_ids[emotion_label_id_map[label]] = 1
        return em_ids

    ref_ids, hyp_ids = [], []
    for ref, hyp in tqdm(zip(references, hypotheses), desc="calc em f1...", total=len(references)):
        pair = [ref, hyp]
        dis = classifier(pair)
        ref_ids.append(get_em_ids_with_threshold(dis[0]))
        hyp_ids.append(get_em_ids_with_threshold(dis[1]))

    return f1_score(y_true=ref_ids, y_pred=hyp_ids, average="samples"), accuracy_score(y_true=ref_ids, y_pred=hyp_ids)
