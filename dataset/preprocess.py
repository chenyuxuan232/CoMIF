import pandas as pd
import os

from transformers import pipeline
from keybert import KeyBERT

from tqdm import tqdm
import random
import numpy as np

# user 1 personas,user 2 personas,Best Generated Conversation
PROCESS_JSON = False
PROCESS_KEYWORD = False
PROCESS_EMOTION = False

from configs.configs import emotion_label_id_map, emotion_threshold_dict, n_emotion_class


def csv2json(file_paths: list[str], save_paths: list[str]):
    print("========== csv->json ==========")

    def str2list(string: str) -> list[str]:
        res = string.split("\n")
        res = [s.strip() for s in res]
        res = [s for s in res if s != ""]
        return res

    def test(conv: list[str]):
        for utterance in conv:
            try:
                utterance.split(":")[1]
            except:
                return False
        return True

    for file_path, save_path in zip(file_paths, save_paths):
        df = pd.read_csv(file_path)

        conversations = df["Best Generated Conversation"].to_list()
        personas_1 = df["user 1 personas"].to_list()
        personas_2 = df["user 2 personas"].to_list()
        # 将 str 转为 list[str]
        conversations = [str2list(conv) for conv in conversations]
        personas_1 = [str2list(persona) for persona in personas_1]
        personas_2 = [str2list(persona) for persona in personas_2]

        tmp_conversations = []
        tmp_personas_1 = []
        tmp_personas_2 = []
        # 删除 conversation 中包含非对话内容的对话
        for conv, p_1, p_2 in zip(conversations, personas_1, personas_2):
            if test(conv):
                tmp_conversations.append(conv)
                tmp_personas_1.append(p_1)
                tmp_personas_2.append(p_2)

        df = pd.DataFrame()
        df["Best Generated Conversation"] = tmp_conversations
        df["user 1 personas"] = tmp_personas_1
        df["user 2 personas"] = tmp_personas_2

        df.to_json(save_path, orient="records", force_ascii=False)
    return


def generate_keywords(
    file_paths: list[str],
    model: str = "/home/cyx/workspace/ChatBot/model_hubs/paraphrase-multilingual-mpnet-base-v2",
    threshold: float = 0.2,
    show_progress: bool = False,
):
    print("========== generate keywords ==========")

    def weight_threshold(keywords: list[list[tuple[str, float]]], threshold: float = 0.2) -> list[dict]:
        # 删除权重小于阈值的关键词
        new_keywords: list[list[tuple[str, float]]] = []
        for keyword in keywords:
            new_keywords.append([(k, w) for k, w in keyword if w > threshold])
        # 将数据转换为 list[dict] 的格式，其中dict的K为关键词，V为权重
        res_keywords: list[dict] = []
        for keyword in new_keywords:
            tmp_dict = dict()
            for k, w in keyword:
                tmp_dict[k] = w
            res_keywords.append(tmp_dict)
        return res_keywords

    if show_progress:
        print("加载KeyBert模型...")
    kw_model = KeyBERT(model=model)

    for i, path in enumerate(file_paths):
        df = pd.read_json(path, orient="records")
        conversations: list[list[str]] = df["Best Generated Conversation"].to_list()

        topic_keywords = []
        for conv in tqdm(conversations, desc=f'File_{i}: "{path}"', leave=False) if show_progress else conversations:
            # 删除句子前面的 user 1 和 user 2
            conv = [utterance.split(":", 1)[1] for utterance in conv]
            # 获取整个对话中每个 utterance 的 keywords
            keywords = kw_model.extract_keywords(conv, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=10)
            # 删除权重低于阈值的 keyword
            keywords = weight_threshold(keywords, threshold=threshold)

            topic_keywords.append(keywords)

        df["topic keywords"] = topic_keywords
        df.to_json(path, orient="records", force_ascii=False)
    return


def generate_emotions(
    file_paths: list[str],
    model: str = "/home/cyx/workspace/ChatBot/model_hubs/roberta-base-go_emotions",
    show_progress: bool = False,
):
    print("========== generate emotion ==========")
    # 将 emotion_threshold_dict 转为 list/np.array
    thresholds = [0] * n_emotion_class
    for label, threshold in emotion_threshold_dict.items():
        thresholds[emotion_label_id_map[label]] = threshold
    thresholds = np.array(thresholds)

    def probability_threshold(distributions: list[list[dict]]) -> list[list[int]]:
        # 将数据转换为 list[np.array] 格式
        new_distribution: list[np.ndarray] = []
        for distribution in distributions:
            tmp_distribution_list = [0] * n_emotion_class
            for emo_dict in distribution:
                label = emo_dict["label"]
                score = emo_dict["score"]
                tmp_distribution_list[emotion_label_id_map[label]] = score
            new_distribution.append(np.array(tmp_distribution_list))

        # 获取概率大于阈值的情感类别的id
        res_emotion_ids: list[list[int]] = []
        for distribution in new_distribution:
            tmp_res = (distribution > thresholds).astype(int)
            tmp_res_emotion_ids = [id for id, v in enumerate(tmp_res) if v == 1]
            res_emotion_ids.append(tmp_res_emotion_ids)

        return res_emotion_ids

    if show_progress:
        print("加载emotion classifier模型...")
    classifier = pipeline(task="text-classification", model=model, top_k=None, device=3)

    for i, path in enumerate(file_paths):
        df = pd.read_json(path, orient="records")
        conversations: list[list[str]] = df["Best Generated Conversation"].to_list()

        emotion_distributions = []
        for conv in tqdm(conversations, desc=f'File_{i}: "{path}"', leave=False) if show_progress else conversations:
            # 删除句子前面的 user 1 和 user 2
            conv = [utterance.split(":", 1)[1] for utterance in conv]
            # 获取整个对话中每个 utterance 的 emotion distribution
            distributions: list[list[dict]] = classifier(conv)
            # 保存概率高于阈值的 emotion
            distributions: list[list[int]] = probability_threshold(distributions)

            emotion_distributions.append(distributions)

        df["emotion distribution"] = emotion_distributions
        df.to_json(path, orient="records", force_ascii=False)
    return


def preprocess_SPC_data(show_progress: bool = False):
    dir_name = os.path.dirname(__file__)
    ori_file_paths = [
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_valid.csv",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_test.csv",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_train.csv",
        "../data/Synthetic-Persona-Chat/New-Persona-New-Conversations.csv",
    ]
    save_file_paths = [
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_valid.json",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_test.json",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_train.json",
        "../data/Synthetic-Persona-Chat/New-Persona-New-Conversations.json",
    ]
    ori_file_paths = [os.path.join(dir_name, path) for path in ori_file_paths]
    save_file_paths = [os.path.join(dir_name, path) for path in save_file_paths]

    if PROCESS_JSON:
        csv2json(ori_file_paths, save_file_paths)

    if PROCESS_KEYWORD:
        generate_keywords(save_file_paths, threshold=0.2, show_progress=show_progress)

    if PROCESS_EMOTION:
        generate_emotions(save_file_paths, show_progress=show_progress)
    return

def generate_PC_keywords(
    file_paths: list[str],
    model: str = "/home/cyx/workspace/ChatBot/model_hubs/paraphrase-multilingual-mpnet-base-v2",
    threshold: float = 0.2,
    show_progress: bool = False,
):
    print("========== generate keywords ==========")

    def weight_threshold(keywords: list[list[tuple[str, float]]], threshold: float = 0.2) -> list[dict]:
        # 删除权重小于阈值的关键词
        new_keywords: list[list[tuple[str, float]]] = []
        for keyword in keywords:
            new_keywords.append([(k, w) for k, w in keyword if w > threshold])
        # 将数据转换为 list[dict] 的格式，其中dict的K为关键词，V为权重
        res_keywords: list[dict] = []
        for keyword in new_keywords:
            tmp_dict = dict()
            for k, w in keyword:
                tmp_dict[k] = w
            res_keywords.append(tmp_dict)
        return res_keywords

    if show_progress:
        print("加载KeyBert模型...")
    kw_model = KeyBERT(model=model)

    for i, path in enumerate(file_paths):
        df = pd.read_json(path, orient="records")
        conversations: list[list[str]] = df["conversation"].to_list()

        topic_keywords = []
        for conv in tqdm(conversations, desc=f'File_{i}: "{path}"', leave=False) if show_progress else conversations:
            # 获取整个对话中每个 utterance 的 keywords
            keywords = kw_model.extract_keywords(conv, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=10)
            # 删除权重低于阈值的 keyword
            keywords = weight_threshold(keywords, threshold=threshold)

            topic_keywords.append(keywords)

        df["topic keywords"] = topic_keywords
        df.to_json(path, orient="records", force_ascii=False)
    return

def generate_PC_emotions(
    file_paths: list[str],
    model: str = "/home/cyx/workspace/ChatBot/model_hubs/roberta-base-go_emotions",
    show_progress: bool = False,
):
    print("========== generate emotion ==========")
    # 将 emotion_threshold_dict 转为 list/np.array
    thresholds = [0] * n_emotion_class
    for label, threshold in emotion_threshold_dict.items():
        thresholds[emotion_label_id_map[label]] = threshold
    thresholds = np.array(thresholds)

    def probability_threshold(distributions: list[list[dict]]) -> list[list[int]]:
        # 将数据转换为 list[np.array] 格式
        new_distribution: list[np.ndarray] = []
        for distribution in distributions:
            tmp_distribution_list = [0] * n_emotion_class
            for emo_dict in distribution:
                label = emo_dict["label"]
                score = emo_dict["score"]
                tmp_distribution_list[emotion_label_id_map[label]] = score
            new_distribution.append(np.array(tmp_distribution_list))

        # 获取概率大于阈值的情感类别的id
        res_emotion_ids: list[list[int]] = []
        for distribution in new_distribution:
            tmp_res = (distribution > thresholds).astype(int)
            tmp_res_emotion_ids = [id for id, v in enumerate(tmp_res) if v == 1]
            res_emotion_ids.append(tmp_res_emotion_ids)

        return res_emotion_ids

    if show_progress:
        print("加载emotion classifier模型...")
    classifier = pipeline(task="text-classification", model=model, top_k=None, device=5)

    for i, path in enumerate(file_paths):
        df = pd.read_json(path, orient="records")
        conversations: list[list[str]] = df["conversation"].to_list()

        emotion_distributions = []
        for conv in tqdm(conversations, desc=f'File_{i}: "{path}"', leave=False) if show_progress else conversations:
            # 获取整个对话中每个 utterance 的 emotion distribution
            distributions: list[list[dict]] = classifier(conv)
            # 保存概率高于阈值的 emotion
            distributions: list[list[int]] = probability_threshold(distributions)

            emotion_distributions.append(distributions)

        df["emotion distribution"] = emotion_distributions
        df.to_json(path, orient="records", force_ascii=False)
    return

def preprocess_PC_data(show_progress: bool = False):
    dir_name = os.path.dirname(__file__)
    ori_file_paths = [
        "../data/Persona-Chat/personachat_truecased_full_valid.json",
        "../data/Persona-Chat/personachat_truecased_full_train.json",
    ]
    save_file_paths = [
        "../data/Persona-Chat_splited/personachat_truecased_full_valid.json",
        "../data/Persona-Chat_splited/personachat_truecased_full_train.json",
    ]

    ori_file_paths = [os.path.join(dir_name, path) for path in ori_file_paths]
    save_file_paths = [os.path.join(dir_name, path) for path in save_file_paths]

    # # 1.提取persona和conversation
    for file_path, save_path in zip(ori_file_paths, save_file_paths):
        df = pd.read_json(file_path, orient="records")
        persona = df["personality"].tolist()
        conversation = df["utterances"].tolist()
        conversation = [conv[-1]["history"] + conv[-1]["candidates"][-1:] for conv in conversation]
        df = pd.DataFrame()
        df["persona"] = persona
        df["conversation"] = conversation
        df.to_json(save_path, orient="records", force_ascii=True)
    
    # 2.生成keywords
    generate_PC_keywords(file_paths=save_file_paths, show_progress=True)
    # 3.生成emotions
    generate_PC_emotions(file_paths=save_file_paths, show_progress=True)

def show_dataset_info():
    from nltk.tokenize import word_tokenize

    dir_name = os.path.dirname(__file__)
    save_file_paths = [
        "./Synthetic-Persona-Chat/Synthetic-Persona-Chat_train.json",
        "./Synthetic-Persona-Chat/Synthetic-Persona-Chat_valid.json",
        "./Synthetic-Persona-Chat/Synthetic-Persona-Chat_test.json",
        "./Synthetic-Persona-Chat/New-Persona-New-Conversations.json",
    ]
    save_file_paths = [os.path.join(dir_name, path) for path in save_file_paths]

    # TODO: utterance num (max min avg)
    utterance_num = []
    # TODO: utterance length (max min avg)
    utterance_length = []
    # TODO: personas num (max min avg)
    personas_num = []
    # TODO: personas length (max min avg)
    personas_length = []
    # TODO: topic num (max min avg)
    topic_num = []
    # TODO: emotion distriburion

    for path in tqdm(save_file_paths, leave=False):
        df = pd.read_json(path, orient="records")
        # list[list[str]]
        conversations = df["Best Generated Conversation"].to_list()
        # list[list[str]]
        personas_1 = df["user 1 personas"].to_list()
        # list[list[str]]
        personas_2 = df["user 2 personas"].to_list()
        # list[list[str]]
        personas = personas_1 + personas_2
        # list[list[dict[str,float]]]
        topic_keywords: list[list[dict[str, float]]] = df["topic keywords"].to_list()
        # list[list[list[float]]]
        emotion_distributions = df["emotion distribution"].to_list()

        utterance_num.extend([len(conv) for conv in tqdm(conversations, desc="utterance_num", leave=False)])
        utterance_length.extend(
            [
                len(word_tokenize(utterance))
                for conv in tqdm(conversations, desc="utterance_length", leave=False)
                for utterance in conv
            ]
        )
        personas_num.extend([len(p) for p in tqdm(personas, desc="personas_num", leave=False)])
        personas_length.extend(
            [len(word_tokenize(p)) for ps in tqdm(personas, desc="personas_length", leave=False) for p in ps]
        )
        topic_num.extend(
            [
                len(keyword)
                for topic_keyword in tqdm(topic_keywords, desc="topic_num", leave=False)
                for keyword in topic_keyword
            ]
        )

    print("===== utterance_num =====")
    print(
        f"max: {max(utterance_num):2d}  min: {min(utterance_num):2d} avg: {sum(utterance_num) / len(utterance_num) : 2.4f}"
    )
    print("===== utterance_length =====")
    print(
        f"max: {max(utterance_length):2d}  min: {min(utterance_length):2d} avg: {sum(utterance_length) / len(utterance_length) : 2.4f}"
    )

    print("===== personas_num =====")
    print(
        f"max: {max(personas_num):2d}  min: {min(personas_num):2d} avg: {sum(personas_num) / len(personas_num) : 2.4f}"
    )
    print("===== personas_length =====")
    print(
        f"max: {max(personas_length):2d}  min: {min(personas_length):2d} avg: {sum(personas_length) / len(personas_length) : 2.4f}"
    )

    print("===== topic_num =====")
    print(f"max: {max(topic_num):2d}  min: {min(topic_num):2d} avg: {sum(topic_num) / len(topic_num) : 2.4f}")

    # ===== utterance_num =====
    # max: 74  min:  6 avg:  22.2870
    # ===== utterance_length =====
    # max: 168  min:  3 avg:  15.8030
    # ===== personas_num =====
    # max:  5  min:  3 avg:  4.7508
    # ===== personas_length =====
    # max: 42  min:  3 avg:  10.2509
    # ===== topic_num =====
    # max: 10  min:  0 avg:  3.3584


def split_dataset(split_weight: list[float] = [0.8, 0.1, 0.1]):
    if len(split_weight) != 3:
        raise ValueError(f"error in num of weight (expected 3, got {len(split_weight)})")
    # 处理文件路径
    dir_name = os.path.dirname(__file__)
    ori_file_paths = [
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_train.json",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_valid.json",
        "../data/Synthetic-Persona-Chat/Synthetic-Persona-Chat_test.json",
        "../data/Synthetic-Persona-Chat/New-Persona-New-Conversations.json",
    ]
    ori_file_paths = [os.path.join(dir_name, path) for path in ori_file_paths]
    save_file_dir = os.path.join(dir_name, "../data/Synthetic-Persona-Chat-splited/Synthetic-Persona-Chat-splited")
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)

    # 获取全部数据
    best_generated_conversation = []
    user_1_personas = []
    user_2_personas = []
    topic_keywords = []
    emotion_distribution = []
    for path in ori_file_paths:
        df = pd.read_json(path, orient="records")
        best_generated_conversation.extend(df["Best Generated Conversation"].to_list())
        user_1_personas.extend(df["user 1 personas"].to_list())
        user_2_personas.extend(df["user 2 personas"].to_list())
        topic_keywords.extend(df["topic keywords"].to_list())
        emotion_distribution.extend(df["emotion distribution"].to_list())

    # 打乱全部数据数据
    num_sample = len(best_generated_conversation)
    indexes = [i for i in range(num_sample)]
    random.shuffle(indexes)

    # 计算各类别数据量
    train_num = round(num_sample * split_weight[0] / sum(split_weight))
    valid_num = round(num_sample * split_weight[1] / sum(split_weight))
    test_num = num_sample - train_num - valid_num

    start = 0
    for num, split in zip([train_num, valid_num, test_num], ["train", "valid", "test"]):
        end = start + num
        sample_indexes = indexes[start:end]
        start += num

        df = pd.DataFrame()
        df["Best Generated Conversation"] = [best_generated_conversation[idx] for idx in sample_indexes]
        df["user 1 personas"] = [user_1_personas[idx] for idx in sample_indexes]
        df["user 2 personas"] = [user_2_personas[idx] for idx in sample_indexes]
        df["topic keywords"] = [topic_keywords[idx] for idx in sample_indexes]
        df["emotion distribution"] = [emotion_distribution[idx] for idx in sample_indexes]

        save_path = os.path.join(save_file_dir, f"{split}.json")
        df.to_json(save_path, orient="records", force_ascii=False)
    return


if __name__ == "__main__":
    preprocess_PC_data()

    # preprocess_SPC_data(show_progress=True)
    # show_dataset_info()
    # split_dataset([8.0, 1.0, 1.0])
