import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import os
import pandas as pd
from tqdm import tqdm
from itertools import chain


class SPCDataset(Dataset):
    def __init__(
        self,
        split,
        enc_tokenizer: PreTrainedTokenizer,
        dec_tokenizer: PreTrainedTokenizer,
        *,
        dataset_dir: str,
        utterance_truncate: int = -1,
        context_utterance_length: int = 64,
        input_utterance_length: int = 512,
        persona_length: int = 32,
    ):
        super(SPCDataset, self).__init__()

        dataset_path = os.path.join(dataset_dir, f"Synthetic-Persona-Chat_{split}.json")
        # 读取数据文件
        df = pd.read_json(dataset_path, orient="records")
        conversation: list[list[str]] = df["Best Generated Conversation"].to_list()
        personas_1: list[list[str]] = df["user 1 personas"].to_list()
        personas_2: list[list[str]] = df["user 2 personas"].to_list()
        topic_keywords: list[list[dict]] = df["topic keywords"].to_list()
        emotion_ids: list[list[list[int]]] = df["emotion distribution"].to_list()

        self.split = split
        # 上下文句子数量限制
        self.utterance_truncate = (
            utterance_truncate if utterance_truncate != -1 else self._get_max_utterance_num(conversation)
        )
        self.personas_truncate = self._get_max_personas_num(personas_1, personas_2)
        self.keywords_truncate = self._get_max_keywords_num(topic_keywords)
        # 每个句子的长度限制
        self.context_utterance_length = context_utterance_length
        self.personas_length = persona_length
        self.input_utterance_length = input_utterance_length
        # 情感类别数量
        self.emotion_n_class = 28

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

        self.all_raw_datas: list[dict] = []
        # 处理原始数据，将其划分为样本
        # 每个样本包含：context
        # 用于encoder的 personas
        # 用于decoder的 response
        # 用于计算loss的 his_keyword_ids、his_emotion_ids、tgt_keyword_ids、tgt_emotion_ids
        for i in tqdm(range(len(conversation)), desc=f"Loading {split} dataset...", leave=False):
            datas = self._generate_raw_datas(
                conversation[i],
                personas_1[i],
                personas_2[i],
                topic_keywords[i],
                emotion_ids[i],
            )
            self.all_raw_datas.extend(datas)

    def _generate_raw_datas(
        self,
        conversation: list[str],
        personas_1: list[str],
        personas_2: list[str],
        topic_keywords: list[dict[str, float]],
        emotion_ids: list[list[int]],
    ):
        """
        输入：一段对话以及个性、主题、情感
        输出：dict，
             每个样本包含：context
             用于encoder的 personas
             用于decoder的 response
             用于计算loss的 his_keyword_ids、his_emotion_ids、tgt_keyword_ids、tgt_emotion_ids
        """
        # 删除前面的 "User1:"
        conversation = [(utterance.split(":", 1)[-1]).strip() for utterance in conversation]

        # keywords -> kw_ids & remove all special token
        keyword_ids: list[list[int]] = self._keywords2kw_ids(topic_keywords)

        # 将一段对话划分为多个数据样本(raw data)
        datas: list[dict] = self._conversation2datas(
            conversation,
            personas_1,
            personas_2,
            keyword_ids,
            emotion_ids,
        )
        return datas

    def _conversation2datas(
        self,
        conversation: list[str],
        personas_1: list[str],
        personas_2: list[str],
        keyword_ids: list[list[int]],
        emotion_ids: list[list[int]],
    ):
        """
        将一段对话划分为多条数据
        """
        utterance_num = len(conversation)
        datas = []
        for end in range(1, utterance_num):
            start = 0 if end < self.utterance_truncate else end - self.utterance_truncate
            datas.append(
                {
                    "context": conversation[start:end],  # list[str]
                    "personas": personas_1 if end % 2 == 0 else personas_2,  # list[str]
                    "response": conversation[end],  # str
                    "his_keyword_ids": keyword_ids[start:end],  # list[list[int]]
                    "his_emotion_ids": emotion_ids[start:end],  # list[list[int]]
                    "tgt_keyword_ids": keyword_ids[end],  # list[int]
                    "tgt_emotion_ids": emotion_ids[end],  # list[int]
                }
            )
        return datas

    def _keywords2kw_ids(self, topic_keywords: list[dict[str, float]]):
        # 将keyword转为id，并出去全部的特殊token
        keyword_ids: list[list[int]] = [
            self.enc_tokenizer.convert_tokens_to_ids(
                [k for k, _ in kws.items() if k not in self.enc_tokenizer.all_special_ids]
            )
            for kws in topic_keywords
        ]
        return keyword_ids

    def _get_max_utterance_num(self, conversation: list[list[str]]):
        utterance_num = [len(conv) for conv in conversation]
        return max(utterance_num)

    def _get_max_personas_num(self, personas_1: list[list[str]], personas_2: list[list[str]]):
        personas_num = [len(p) for p in personas_1]
        personas_num.extend([len(p) for p in personas_2])
        return max(personas_num)

    def _get_max_keywords_num(self, keywords: list[list[dict[str, float]]]):
        max_keywords_num = 0
        for kws in keywords:
            for kw in kws:
                if len(kw) > max_keywords_num:
                    max_keywords_num = len(kw)
        return max_keywords_num

    def __getitem__(self, index):
        raw_data = self.all_raw_datas[index]
        return self.encode_raw_data(raw_data, index)

    def encode_raw_data(self, raw_data, index):
        # tokenizer、pad、ids->distribution
        context: list[str] = raw_data["context"]
        personas: list[str] = raw_data["personas"]
        response: str = raw_data["response"]
        his_keyword_ids: list[list[int]] = raw_data["his_keyword_ids"] if "his_keyword_ids" in raw_data else None
        his_emotion_ids: list[list[int]] = raw_data["his_emotion_ids"] if "his_emotion_ids" in raw_data else None
        tgt_keyword_ids: list[int] = raw_data["tgt_keyword_ids"] if "tgt_keyword_ids" in raw_data else None
        tgt_emotion_ids: list[int] = raw_data["tgt_emotion_ids"] if "tgt_emotion_ids" in raw_data else None

        res_encoded_data = {}
        res_encoded_data["index"] = index

        res_encoded_data["context"] = self._tokenize_pad(context, opt="context")
        res_encoded_data["personas"] = self._tokenize_pad(personas, opt="personas")
        # 此处将opt修改为 "dec_inputs" 可得到用于GPT2的输入和labels
        inputs, labels = self._tokenize_pad((context, response), opt="dec_inputs")
        res_encoded_data["inputs"] = inputs
        res_encoded_data["labels"] = labels

        if his_keyword_ids is not None:
            res_encoded_data["his_kw_dis"] = self._tokenize_pad(his_keyword_ids, opt="his_keyword_ids")
        if his_emotion_ids is not None:
            res_encoded_data["his_em_dis"] = self._tokenize_pad(his_emotion_ids, opt="his_emotion_ids")
        if tgt_keyword_ids is not None:
            res_encoded_data["tgt_kw_dis"] = self._tokenize_pad(tgt_keyword_ids, opt="tgt_keyword_ids")
        if tgt_emotion_ids is not None:
            res_encoded_data["tgt_em_dis"] = self._tokenize_pad(tgt_emotion_ids, opt="tgt_emotion_ids")

        return res_encoded_data

    def _tokenize_pad(self, input, *, opt: str):
        if opt == "context" or opt == "personas":
            # input: list[str]
            text_num = self.utterance_truncate if opt == "context" else self.personas_truncate
            max_length = self.context_utterance_length if opt == "context" else self.personas_length
            tokenized = self.enc_tokenizer(
                input, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
            pad_data = {
                "input_ids": self._pad_tensor(
                    tokenized["input_ids"], [text_num, max_length], self.enc_tokenizer.pad_token_id
                ),
                "attention_mask": self._pad_tensor(tokenized["attention_mask"], [text_num, max_length], 0),
            }
            return pad_data

        elif opt == "his_keyword_ids" or opt == "his_emotion_ids":
            # input: list[list[int]]
            n_class = self.enc_tokenizer.vocab_size if opt == "his_keyword_ids" else 28
            pad_distributions = torch.zeros([self.utterance_truncate, n_class], dtype=torch.long)
            for i, indices in enumerate(input):
                pad_distributions[i, indices] = 1
            return pad_distributions

        elif opt == "tgt_keyword_ids" or opt == "tgt_emotion_ids":
            # input: list[int]
            n_class = self.enc_tokenizer.vocab_size if opt == "tgt_keyword_ids" else 28
            distribution = torch.tensor([(1 if i in input else 0) for i in range(n_class)], dtype=torch.long)
            return distribution

        elif opt == "enc_inputs" or opt == "dec_inputs":
            tokenizer = self.enc_tokenizer if opt == "enc_inputs" else self.dec_tokenizer

            bos_id, eos_id, sep_id, pad_id = (
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )

            input_ids = (
                [[bos_id]]
                + [(tokenizer.encode(utter, add_special_tokens=False) + [sep_id]) for utter in input[0]]
                + [tokenizer.encode(input[1], add_special_tokens=False)]
                + [[eos_id]]
            )
            token_type_ids = [
                (
                    [0] * (1 + len(tokenizer.encode(utter, add_special_tokens=False)))
                    if i % 2 == 0
                    else [1] * (1 + len(tokenizer.encode(utter, add_special_tokens=False)))
                )
                for i, utter in enumerate(input[0] + [input[1]])
            ]
            token_type_ids += [[token_type_ids[-1][0]]]

            input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            token_type_ids = torch.tensor(list(chain(*token_type_ids)), dtype=torch.long)

            if input_ids.numel() > self.input_utterance_length:
                raise Warning(f"当前数据长度({input_ids.numel()})超过设定长度({self.input_utterance_length})")

            # 计算对应的labels
            if opt == "enc_inputs":
                if len(self.enc_tokenizer.encode(input[1], add_special_tokens=False)) != len(
                    self.dec_tokenizer.encode(input[1], add_special_tokens=False)
                ):
                    raise Warning(f"'{input[1]}'在tokenize后长度不相同")
            response_ids = self.dec_tokenizer.encode(input[1], add_special_tokens=False)
            labels = (
                [[-100] * (input_ids.numel() - len(response_ids) - 2)]
                + [[self.dec_tokenizer.sep_token_id]]
                + [response_ids]
                + [[self.dec_tokenizer.eos_token_id]]
            )
            labels = torch.tensor(list(chain(*labels)), dtype=torch.long)

            inputs = {
                "input_ids": self._pad_tensor(input_ids, [self.input_utterance_length], pad_value=pad_id),
                "attention_mask": self._pad_tensor(attention_mask, [self.input_utterance_length], pad_value=0),
                "token_type_ids": self._pad_tensor(token_type_ids, [self.input_utterance_length], pad_value=0),
            }
            return inputs, self._pad_tensor(labels, [self.input_utterance_length], pad_value=-100)

        else:
            raise ValueError(f"'{opt}' is not support")

    def _pad_tensor(self, tensor: torch.Tensor, target_shape: list[int], pad_value):
        assert len(tensor.shape) == len(target_shape), "'target_shape'的维度与'tensor'不匹配。"
        assert len(tensor.shape) <= 3 or len(tensor.shape) == 0, f"不支持当前维度({len(tensor.shape)})的操作"

        res_tensor = torch.full(target_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        ori_shape = tensor.shape

        if len(tensor.shape) == 1:
            res_tensor[: ori_shape[0]] = tensor
        elif len(tensor.shape) == 2:
            res_tensor[: ori_shape[0], : ori_shape[1]] = tensor
        else:
            res_tensor[: ori_shape[0], : ori_shape[1], : ori_shape[2]] = tensor
        return res_tensor

    def __len__(self):
        return len(self.all_raw_datas)

    def get_max_length(self):
        max_utterance_length = 0
        for index in tqdm(range(self.__len__())):
            context = self.all_raw_datas[index]["context"]
            response = self.all_raw_datas[index]["response"]

            tokenizer = self.enc_tokenizer

            bos_id, eos_id, sep_id, pad_id = (
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )

            input_ids = (
                [[bos_id]]
                + [(tokenizer.encode(utter, add_special_tokens=False) + [sep_id]) for utter in context]
                + [tokenizer.encode(response, add_special_tokens=False)]
                + [[eos_id]]
            )

            input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.long)

            if input_ids.numel() > max_utterance_length:
                max_utterance_length = input_ids.numel()
        return max_utterance_length

    def get_pos_weight(self, param_name, device):
        """
        param_name: one of (em_his, kw_his, em_g, kw_g)
        """
        valid_param_names = ["em_his", "kw_his", "em_g", "kw_g"]
        assert param_name in valid_param_names, f"'param_name' must be one of {valid_param_names}."

        if param_name == "em_his":
            raw_data = [d for data in self.all_raw_datas for d in data["his_emotion_ids"]]
            n_class = self.emotion_n_class
        elif param_name == "kw_his":
            raw_data = [d for data in self.all_raw_datas for d in data["his_keyword_ids"]]
            n_class = self.enc_tokenizer.vocab_size
        elif param_name == "em_g":
            raw_data = [data["tgt_emotion_ids"] for data in self.all_raw_datas]
            n_class = self.emotion_n_class
        elif param_name == "kw_g":
            raw_data = [data["tgt_keyword_ids"] for data in self.all_raw_datas]
            n_class = self.enc_tokenizer.vocab_size

        pos_num = {}
        for data in raw_data:
            for id in data:
                if id in pos_num:
                    pos_num[id] += 1
                else:
                    pos_num[id] = 1

        pos_weigth = [0] * n_class
        for i in range(n_class):
            if i in pos_num:
                pos_weigth[i] = len(raw_data) / pos_num[i] - 1

        return torch.tensor(pos_weigth, dtype=torch.float, device=device)

class PCDataset(Dataset):
    def __init__(
        self,
        split,
        enc_tokenizer: PreTrainedTokenizer,
        dec_tokenizer: PreTrainedTokenizer,
        *,
        dataset_dir: str,
        utterance_truncate: int = -1,
        context_utterance_length: int = 64,
        input_utterance_length: int = 512,
        persona_length: int = 32,
    ):
        super(PCDataset, self).__init__()

        # raise NotImplementedError("需要先为原始数据集添加情感和主题后再使用")
        _split = "train" if split == "train" else  "valid"
        dataset_path = os.path.join(dataset_dir, f"personachat_truecased_full_{_split}.json")
        # 读取数据文件
        df = pd.read_json(dataset_path, orient="records")
        conversation: list[list[str]] = df["conversation"].to_list()
        personas: list[list[str]] = df["persona"].to_list()
        topic_keywords: list[list[dict]] = df["topic keywords"].to_list()
        emotion_ids: list[list[list[int]]] = df["emotion distribution"].to_list()

        self.split = split
        # 上下文句子数量限制
        self.utterance_truncate = (
            utterance_truncate if utterance_truncate != -1 else self._get_max_utterance_num(conversation)
        )
        self.personas_truncate = self._get_max_personas_num(personas)
        self.keywords_truncate = self._get_max_keywords_num(topic_keywords)
        # 每个句子的长度限制
        self.context_utterance_length = context_utterance_length
        self.personas_length = persona_length
        self.input_utterance_length = input_utterance_length
        # 情感类别数量
        self.emotion_n_class = 28

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer

        self.all_raw_datas: list[dict] = []
        # 处理原始数据，将其划分为样本
        # 每个样本包含：context
        # 用于encoder的 personas
        # 用于decoder的 response
        # 用于计算loss的 his_keyword_ids、his_emotion_ids、tgt_keyword_ids、tgt_emotion_ids
        for i in tqdm(range(len(conversation)), desc=f"Loading {split} dataset...", leave=False):
            datas = self._generate_raw_datas(
                conversation[i],
                personas[i],
                topic_keywords[i],
                emotion_ids[i],
            )
            self.all_raw_datas.extend(datas)

    def _generate_raw_datas(
        self,
        conversation: list[str],
        personas: list[str],
        topic_keywords: list[dict[str, float]],
        emotion_ids: list[list[int]],
    ):
        """
        输入：一段对话以及个性、主题、情感
        输出：dict，
             每个样本包含：context
             用于encoder的 personas
             用于decoder的 response
             用于计算loss的 his_keyword_ids、his_emotion_ids、tgt_keyword_ids、tgt_emotion_ids
        """
        # keywords -> kw_ids & remove all special token
        keyword_ids: list[list[int]] = self._keywords2kw_ids(topic_keywords)

        # 将一段对话划分为多个数据样本(raw data)
        datas: list[dict] = self._conversation2datas(
            conversation,
            personas,
            keyword_ids,
            emotion_ids,
        )
        return datas

    def _conversation2datas(
        self,
        conversation: list[str],
        personas: list[str],
        keyword_ids: list[list[int]],
        emotion_ids: list[list[int]],
    ):
        """
        将一段对话划分为多条数据
        """
        utterance_num = len(conversation)
        datas = []
        for end in range(1, utterance_num, 2):
            start = 0 if end < self.utterance_truncate else end - self.utterance_truncate
            datas.append(
                {
                    "context": conversation[start:end],  # list[str]
                    "personas": personas,  # list[str]
                    "response": conversation[end],  # str
                    "his_keyword_ids": keyword_ids[start:end],  # list[list[int]]
                    "his_emotion_ids": emotion_ids[start:end],  # list[list[int]]
                    "tgt_keyword_ids": keyword_ids[end],  # list[int]
                    "tgt_emotion_ids": emotion_ids[end],  # list[int]
                }
            )
        return datas

    def _keywords2kw_ids(self, topic_keywords: list[dict[str, float]]):
        # 将keyword转为id，并出去全部的特殊token
        keyword_ids: list[list[int]] = [
            self.enc_tokenizer.convert_tokens_to_ids(
                [k for k, _ in kws.items() if k not in self.enc_tokenizer.all_special_ids]
            )
            for kws in topic_keywords
        ]
        return keyword_ids

    def _get_max_utterance_num(self, conversation: list[list[str]]):
        utterance_num = [len(conv) for conv in conversation]
        return max(utterance_num)

    def _get_max_personas_num(self, personas: list[list[str]]):
        personas_num = [len(p) for p in personas]
        return max(personas_num)

    def _get_max_keywords_num(self, keywords: list[list[dict[str, float]]]):
        max_keywords_num = 0
        for kws in keywords:
            for kw in kws:
                if len(kw) > max_keywords_num:
                    max_keywords_num = len(kw)
        return max_keywords_num

    def __getitem__(self, index):
        raw_data = self.all_raw_datas[index]
        return self.encode_raw_data(raw_data, index)

    def encode_raw_data(self, raw_data, index):
        # tokenizer、pad、ids->distribution
        context: list[str] = raw_data["context"]
        personas: list[str] = raw_data["personas"]
        response: str = raw_data["response"]
        his_keyword_ids: list[list[int]] = raw_data["his_keyword_ids"] if "his_keyword_ids" in raw_data else None
        his_emotion_ids: list[list[int]] = raw_data["his_emotion_ids"] if "his_emotion_ids" in raw_data else None
        tgt_keyword_ids: list[int] = raw_data["tgt_keyword_ids"] if "tgt_keyword_ids" in raw_data else None
        tgt_emotion_ids: list[int] = raw_data["tgt_emotion_ids"] if "tgt_emotion_ids" in raw_data else None

        res_encoded_data = {}
        res_encoded_data["index"] = index
        res_encoded_data["context"] = self._tokenize_pad(context, opt="context")
        res_encoded_data["personas"] = self._tokenize_pad(personas, opt="personas")
        # 此处将opt修改为 "dec_inputs" 可得到用于GPT2的输入和labels
        inputs, labels = self._tokenize_pad((context, response), opt="dec_inputs")
        res_encoded_data["inputs"] = inputs
        res_encoded_data["labels"] = labels

        if his_keyword_ids is not None:
            res_encoded_data["his_kw_dis"] = self._tokenize_pad(his_keyword_ids, opt="his_keyword_ids")
        if his_emotion_ids is not None:
            res_encoded_data["his_em_dis"] = self._tokenize_pad(his_emotion_ids, opt="his_emotion_ids")
        if tgt_keyword_ids is not None:
            res_encoded_data["tgt_kw_dis"] = self._tokenize_pad(tgt_keyword_ids, opt="tgt_keyword_ids")
        if tgt_emotion_ids is not None:
            res_encoded_data["tgt_em_dis"] = self._tokenize_pad(tgt_emotion_ids, opt="tgt_emotion_ids")

        return res_encoded_data

    def _tokenize_pad(self, input, *, opt: str):
        if opt == "context" or opt == "personas":
            # input: list[str]
            text_num = self.utterance_truncate if opt == "context" else self.personas_truncate
            max_length = self.context_utterance_length if opt == "context" else self.personas_length
            tokenized = self.enc_tokenizer(
                input, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
            )
            pad_data = {
                "input_ids": self._pad_tensor(
                    tokenized["input_ids"], [text_num, max_length], self.enc_tokenizer.pad_token_id
                ),
                "attention_mask": self._pad_tensor(tokenized["attention_mask"], [text_num, max_length], 0),
            }
            return pad_data

        elif opt == "his_keyword_ids" or opt == "his_emotion_ids":
            # input: list[list[int]]
            n_class = self.enc_tokenizer.vocab_size if opt == "his_keyword_ids" else self.emotion_n_class
            pad_distributions = torch.zeros([self.utterance_truncate, n_class], dtype=torch.long)
            for i, indices in enumerate(input):
                pad_distributions[i, indices] = 1
            return pad_distributions

        elif opt == "tgt_keyword_ids" or opt == "tgt_emotion_ids":
            # input: list[int]
            n_class = self.enc_tokenizer.vocab_size if opt == "tgt_keyword_ids" else self.emotion_n_class
            distribution = torch.tensor([(1 if i in input else 0) for i in range(n_class)], dtype=torch.long)
            return distribution

        elif opt == "enc_inputs" or opt == "dec_inputs":
            tokenizer = self.enc_tokenizer if opt == "enc_inputs" else self.dec_tokenizer

            bos_id, eos_id, sep_id, pad_id = (
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )

            input_ids = (
                [[bos_id]]
                + [(tokenizer.encode(utter, add_special_tokens=False) + [sep_id]) for utter in input[0]]
                + [tokenizer.encode(input[1], add_special_tokens=False)]
                + [[eos_id]]
            )
            token_type_ids = [
                (
                    [0] * (1 + len(tokenizer.encode(utter, add_special_tokens=False)))
                    if i % 2 == 0
                    else [1] * (1 + len(tokenizer.encode(utter, add_special_tokens=False)))
                )
                for i, utter in enumerate(input[0] + [input[1]])
            ]
            token_type_ids += [[token_type_ids[-1][0]]]

            input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            token_type_ids = torch.tensor(list(chain(*token_type_ids)), dtype=torch.long)

            if input_ids.numel() > self.input_utterance_length:
                raise Warning(f"当前数据长度({input_ids.numel()})超过设定长度({self.input_utterance_length})")

            # 计算对应的labels
            if opt == "enc_inputs":
                if len(self.enc_tokenizer.encode(input[1], add_special_tokens=False)) != len(
                    self.dec_tokenizer.encode(input[1], add_special_tokens=False)
                ):
                    raise Warning(f"'{input[1]}'在tokenize后长度不相同")
            response_ids = self.dec_tokenizer.encode(input[1], add_special_tokens=False)
            labels = (
                [[-100] * (input_ids.numel() - len(response_ids) - 2)]
                + [[self.dec_tokenizer.sep_token_id]]
                + [response_ids]
                + [[self.dec_tokenizer.eos_token_id]]
            )
            labels = torch.tensor(list(chain(*labels)), dtype=torch.long)

            inputs = {
                "input_ids": self._pad_tensor(input_ids, [self.input_utterance_length], pad_value=pad_id),
                "attention_mask": self._pad_tensor(attention_mask, [self.input_utterance_length], pad_value=0),
                "token_type_ids": self._pad_tensor(token_type_ids, [self.input_utterance_length], pad_value=0),
            }
            return inputs, self._pad_tensor(labels, [self.input_utterance_length], pad_value=-100)

        else:
            raise ValueError(f"'{opt}' is not support")

    def _pad_tensor(self, tensor: torch.Tensor, target_shape: list[int], pad_value):
        assert len(tensor.shape) == len(target_shape), "'target_shape'的维度与'tensor'不匹配。"
        assert len(tensor.shape) <= 3 or len(tensor.shape) == 0, f"不支持当前维度({len(tensor.shape)})的操作"

        res_tensor = torch.full(target_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        ori_shape = tensor.shape

        if len(tensor.shape) == 1:
            res_tensor[: ori_shape[0]] = tensor
        elif len(tensor.shape) == 2:
            res_tensor[: ori_shape[0], : ori_shape[1]] = tensor
        else:
            res_tensor[: ori_shape[0], : ori_shape[1], : ori_shape[2]] = tensor
        return res_tensor

    def __len__(self):
        return len(self.all_raw_datas)

    def get_max_length(self):
        max_utterance_length = 0
        for index in tqdm(range(self.__len__())):
            context = self.all_raw_datas[index]["context"]
            response = self.all_raw_datas[index]["response"]

            tokenizer = self.enc_tokenizer

            bos_id, eos_id, sep_id, pad_id = (
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            )

            input_ids = (
                [[bos_id]]
                + [(tokenizer.encode(utter, add_special_tokens=False) + [sep_id]) for utter in context]
                + [tokenizer.encode(response, add_special_tokens=False)]
                + [[eos_id]]
            )

            input_ids = torch.tensor(list(chain(*input_ids)), dtype=torch.long)

            if input_ids.numel() > max_utterance_length:
                max_utterance_length = input_ids.numel()
        return max_utterance_length

    def get_pos_weight(self, param_name, device):
        """
        param_name: one of (em_his, kw_his, em_g, kw_g)
        """
        valid_param_names = ["em_his", "kw_his", "em_g", "kw_g"]
        assert param_name in valid_param_names, f"'param_name' must be one of {valid_param_names}."

        if param_name == "em_his":
            raw_data = [d for data in self.all_raw_datas for d in data["his_emotion_ids"]]
            n_class = self.emotion_n_class
        elif param_name == "kw_his":
            raw_data = [d for data in self.all_raw_datas for d in data["his_keyword_ids"]]
            n_class = self.enc_tokenizer.vocab_size
        elif param_name == "em_g":
            raw_data = [data["tgt_emotion_ids"] for data in self.all_raw_datas]
            n_class = self.emotion_n_class
        elif param_name == "kw_g":
            raw_data = [data["tgt_keyword_ids"] for data in self.all_raw_datas]
            n_class = self.enc_tokenizer.vocab_size

        pos_num = {}
        for data in raw_data:
            for id in data:
                if id in pos_num:
                    pos_num[id] += 1
                else:
                    pos_num[id] = 1

        pos_weigth = [0] * n_class
        for i in range(n_class):
            if i in pos_num:
                pos_weigth[i] = len(raw_data) / pos_num[i] - 1

        return torch.tensor(pos_weigth, dtype=torch.float, device=device)



def batch_to_device(batch, device):
    def _to_device(data):
        if "token_type_ids" in data.keys():
            return {
                "input_ids": data["input_ids"].to(device),
                "attention_mask": data["attention_mask"].to(device),
                "token_type_ids": data["token_type_ids"].to(device),
            }
        else:
            return {
                "input_ids": data["input_ids"].to(device),
                "attention_mask": data["attention_mask"].to(device),
            }

    res_batch = {
        "index": batch["index"].to(device),
        "context": _to_device(batch["context"]),
        "personas": _to_device(batch["personas"]),
        "inputs": _to_device(batch["inputs"]),
        "labels": batch["labels"].to(device),
    }

    for key in ["his_kw_dis", "his_em_dis", "tgt_kw_dis", "tgt_em_dis"]:
        if key in batch:
            res_batch[key] = batch[key].to(device)

    return res_batch

def get_raw_data(dataset:SPCDataset|PCDataset, index:int, key:str):
    return dataset.all_raw_datas[index][key]