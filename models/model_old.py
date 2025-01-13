import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.utils import set_ignore_index, pack_batch, pad_batch

import math

from transformers import RobertaModel, GPT2Model, GPT2Config, RobertaConfig

from logs.log_logits import log_logit


class LuongAttn(nn.Module):
    def __init__(self, method, hidden_size):
        super(LuongAttn, self).__init__()

        if method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")

        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = torch.nn.Linear(hidden_size, hidden_size)
        elif self.method == "concat":
            self.attn = torch.nn.Linear(hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        # hidden: (bsz, hidden_size)
        # encoder_output: (bsz, max_num, hidden_size)
        # return: (bsz, max_num)
        return torch.sum((hidden * encoder_output.transpose(0, 1)).transpose(0, 1), dim=2)

    def general_score(self, hidden, encoder_output):
        # hidden: (bsz, hidden_size)
        # encoder_output: (bsz, max_num, hidden_size)
        # return: (bsz, max_num)
        energy = self.attn(encoder_output)
        return torch.sum((hidden * energy.transpose(0, 1)).transpose(0, 1), dim=2)

    def concat_score(self, hidden, encoder_output):
        # hidden: (bsz, hidden_size)
        # encoder_output: (bsz, max_num, hidden_size)
        # return: (bsz, max_num)
        encoder_output = encoder_output.transpose(0, 1)
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum((self.v * energy).transpose(0, 1), dim=2)

    def forward(self, hidden, encoder_outputs, real_num):
        # hidden: (bsz, hidden_size)
        # encoder_outputs: (bsz, max_num, hidden_size)
        # real_num: (bsz) 用于获取padding部分

        device = hidden.device

        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # 这一步需要将输入的encoder_outputs中padding部分的权重将为-inf
        # (bsz, max_num)
        bsz, max_num = encoder_outputs.shape[0], encoder_outputs.shape[1]
        mask = torch.arange(max_num).expand(bsz, max_num).to(device) >= real_num.unsqueeze(-1)
        attn_energies[mask] = float("-inf")

        # (bsz, max_num)
        return F.softmax(attn_energies, dim=1)


class RobertaRNNEncoder(nn.Module):
    def __init__(self, roberta, rnn_n_layers=2, luong_attn_method="concat", dropout=0.1):
        super(RobertaRNNEncoder, self).__init__()
        self.hidden_size = roberta.config.hidden_size

        self.roberta = roberta

        self.context_rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.contextLuongAttn = LuongAttn(method=luong_attn_method, hidden_size=self.hidden_size)

        self.emotion_rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.topic_rnn = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=rnn_n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

    def forward(self, *, context_ids, context_attn_mask, persona_ids, persona_attn_mask):
        res_dict = dict()

        context_real_num = (context_attn_mask.sum(dim=-1) != 0).sum(dim=-1)
        # ========== 1. 处理 context-level 的语义信息, 得到context_3 ==========
        # 将 context 输入到 roberta 中，获取对应的表征
        # context_1: (bsz, max_num, hidden_size) 每个batch中所有句子的句子级的语义表征
        # last_context: (bsz, hidden_size)       每个batch中最后一条句子的句子级语义表征
        context_1, last_context = self.roberta_forward(context_ids, context_attn_mask, process_context=True)
        # 将 utterance-level 的语义表征输入到 semantic-rnn 中，让每个语义表征结合上下文信息
        # context_2: (bsz, max_num, hidden_size)
        # h_n: (bsz, hidden_size)
        context_2, _ = self.rnn_forward(context_1, context_real_num, rnn_type="semantic")
        # 采用 LuongAttention，根据 context_2 和 last_context 计算每个句子语义表征的权重
        # context_weights: (bsz, max_num)
        context_weights = self.contextLuongAttn(
            hidden=last_context, encoder_outputs=context_2, real_num=context_real_num
        )
        # 根据权重计算得到 context-level 的语义表征(即 context)
        # 1.weights (bsz,max_num) -> (bsz,1,max_num)
        # 2.context_unsqueezed = weights * context_2 (bsz, 1, hidden_size)
        # 3.context_3 = context_unsqueezed.squeeze(1) (bsz, hidden_size)
        context_3 = (context_weights.unsqueeze(1)).bmm(context_2).squeeze(1)
        res_dict["context"] = context_3

        # ========== 2. 处理 emotion_his 信息 ==========
        # 将 utterance-level 的语义表征输入到 emotion-rnn 中, 得到emotion_his
        # emotion_his: (bsz, max_num_context, hidden_size)
        emotion_his, _ = self.rnn_forward(context_1, context_real_num, rnn_type="emotion")
        res_dict["emotion_his"] = emotion_his

        # ========== 3. 处理 topic_his 信息 ==========
        # 将 utterance-level 的语义表征输入到 topic-rnn 中, 得到 topic_his
        # topic_his: (bsz, max_num_context, hidden_size)
        topic_his, _ = self.rnn_forward(context_1, context_real_num, rnn_type="topic")
        res_dict["topic_his"] = topic_his

        # ========== 编码persona信息 ==========
        # persona: (bsz, max_num_persona, hidden_size)
        persona = self.roberta_forward(persona_ids, persona_attn_mask)
        res_dict["persona"] = persona

        return res_dict

    def roberta_forward(self, input_ids, attn_mask, process_context=False):
        """
        Params:
            input_ids, attn_mask: (bsz, max_num, seq_len)
            process_context: bool 表示是否在处理上下文，处理上下文时除了返回output_embs之外，还会返回last_context_emb
        Return:
            output_embs: (bsz, max_num, hidden_size)  每个batch中所有句子的句子级的语义表征
            last_context_emb: (bsz, hidden_size)      每个batch中最后一条句子的句子级语义表征
        """
        # 由于使用 dataloader 时需要保证不同样本中的句子数量相同
        # 因此样本中的句子进行了 pad 填充（如: 3条语句扩展成5(max_num)条语句）
        # 为了节约计算资源，需要将所有输入句子压缩后再输入 roberta

        # 在数据处理时，pad 部分的句子对应的 mask 全为0
        # 因此可以通过 mask 判断当前输入中每个样本实际上包含多少条句子
        bsz, max_num = input_ids.shape[0], input_ids.shape[1]
        real_num = (attn_mask.sum(dim=-1) != 0).sum(dim=-1)  # (bsz,)

        # 将 input_ids 和 attn_mask 压缩
        # (bsz, max_num, seq_len) -> (bsz * max_num - pad_num, seq_len)
        packed_input_ids = pack_batch(input_ids, real_num)
        packed_attn_mask = pack_batch(attn_mask, real_num)

        # 将数据输入 roberta，返回结果如下：
        outputs = self.roberta(input_ids=packed_input_ids, attention_mask=packed_attn_mask, return_dict=True)
        # last_hidden_state = outputs["last_hidden_state"]  # (bsz * max_num - pad_num, seq_len, hidden_size)
        pooler_output = outputs["pooler_output"]  # (bsz * max_num - pad_num, hidden_size) -> utterance_level

        # 将数据重新 pad
        # (bsz * max_num - pad_num, hidden_size) -> (bsz, max_num, hidden_size)
        output_embs = pad_batch(pooler_output, real_num=real_num, max_num=max_num, pad_id=0)

        if process_context:
            # 当处理context时，需要将每个batch中最后一句的emb取出作为之后的注意力项
            last_context_emb = output_embs[torch.arange(bsz), (real_num - 1)]
            return output_embs, last_context_emb
        else:
            return output_embs

    def rnn_forward(self, inputs, real_num, rnn_type: str):
        max_num = inputs.shape[1]

        if rnn_type == "semantic":
            rnn = self.context_rnn
        elif rnn_type == "emotion":
            rnn = self.emotion_rnn
        elif rnn_type == "topic":
            rnn = self.topic_rnn
        else:
            raise ValueError(f"'rnn_type' must be one of ['semantic','emotion','topic'], but got '{rnn_type}'")

        inputs = pack_padded_sequence(inputs, real_num.cpu(), batch_first=True, enforce_sorted=False)
        # h_n: (D * num_layers, bsz, n_hidden)
        outputs, (h_n, c_n) = rnn(inputs)
        # output: (bsz, max_num, D * n_hidden)
        outputs, lens = pad_packed_sequence(outputs, batch_first=True, total_length=max_num)

        # 前向和后向的最后一个输出
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]

        # 前向和后向的最后一个隐藏状态
        forward_hn = h_n[-2, :, :]
        backward_hn = h_n[-1, :, :]
        h_n = forward_hn + backward_hn

        # outputs: (bsz, max_num, n_hidden)
        # h_n: (bsz, n_hidden)
        return outputs, h_n


class MHAWithResidualFFN(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(MHAWithResidualFFN, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward, bias=True),
            nn.ReLU(),
            nn.Linear(dim_feedforward, hidden_size, bias=True),
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # MHA
        outputs, attn_weight = self.mha(query=query, key=key, value=value, key_padding_mask=key_padding_mask)
        outputs = self.layer_norm_1(outputs)
        # ResFFN
        residual = outputs
        outputs = self.fc(outputs)
        outputs = outputs + residual
        # Dropout
        outputs = self.dropout(outputs)
        return outputs


class FactorGenerator(nn.Module):
    def __init__(self, *, hidden_size, num_heads: int = 16, dropout: float = 0.1):
        super(FactorGenerator, self).__init__()

        self.persona_attn_ffn = MHAWithResidualFFN(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)

        self.emotion_user_attn_ffns = nn.ModuleList(
            [
                MHAWithResidualFFN(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout),
                MHAWithResidualFFN(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout),
            ]
        )

        self.topic_attn_ffn = MHAWithResidualFFN(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.gate_t_g = nn.Linear(hidden_size * 3, hidden_size)

        self.emotion_attn_ffn = MHAWithResidualFFN(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout)
        self.gate_e_g = nn.Linear(hidden_size * 3, hidden_size)

    def forward(self, context: torch.Tensor, persona, emotion_his, topic_his, context_mask, persona_mask):
        """
        Params:
            context: (bsz, n_hidden),
            persona: (bsz, max_num_persona, n_hidden),
            emotion_his: (bsz, max_num_context, n_hidden),
            topic_his: (bsz, max_num_context, n_hidden),
            context_mask: (bsz, max_num_context, seq_len) 用于计算 emotion、topic 的 key_padding_mask
            persona_mask: (bsz, max_num_persona, seq_len) 用于计算 persona 的 key_padding_mask
            # utterance_num: (bsz,),
            # persona_num: (bsz,)
        Returns:
            P_g: (bsz, n_hidden),
            T_g: (bsz, n_hidden),
            E_g: (bsz, n_hidden)
        """
        # 对于（bsz, 1, hidden_size)的context，MHA等效于LuongAttn
        device = context.device
        context = context.unsqueeze(1)
        # 1. P_g = LuongAttn(context, persona)
        persona_key_padding_mask = self.generate_key_padding_mask(persona_mask, device)
        P_g = self.persona_attn_ffn(  # P_g (bsz, 1, hidden_size)
            query=context, key=persona, value=persona, key_padding_mask=persona_key_padding_mask
        )

        # 2.1. tmp_e = MHA(topic_his, emotion_his, emotion_his)
        emotion_key_padding_mask = self.generate_key_padding_mask(context_mask, device)
        tmp_e = self.emotion_user_attn_ffns[0](
            query=topic_his, key=emotion_his, value=emotion_his, key_padding_mask=emotion_key_padding_mask
        )
        # 2.2. E_user = LuongAttn(context, tmp_e) (bsz, 1, hidden_size)
        E_user = self.emotion_user_attn_ffns[1](query=context, key=tmp_e, value=tmp_e)

        # 3. T_g = LuongAttn(context+P_g+E_user, topic_his)
        g_t = torch.sigmoid(self.gate_t_g(torch.cat([context, P_g, E_user], dim=-1)))
        query = context + g_t * P_g + (1 - g_t) * E_user
        topic_key_padding_mask = self.generate_key_padding_mask(context_mask, device)
        T_g = self.topic_attn_ffn(query=query, key=topic_his, value=topic_his, key_padding_mask=topic_key_padding_mask)

        # 4. E_g = LuongAttn(context+P_g+T_g, E_user)
        g_e = torch.sigmoid(self.gate_e_g(torch.cat([context, P_g, T_g], dim=-1)))
        query = context + g_e * P_g + (1 - g_e) * T_g
        E_g = self.emotion_attn_ffn(query=query, key=E_user, value=E_user)

        return P_g.squeeze(1), T_g.squeeze(1), E_g.squeeze(1)

    def generate_key_padding_mask(self, attention_mask, device):
        """
        Params:
            attention_mask: (bsz, max_num, seq_len)
        Return:
            key_padding_mask: (bsz, max_num)
        """
        bsz, max_num = attention_mask.shape[0], attention_mask.shape[1]
        real_num = (attention_mask.sum(dim=-1) != 0).sum(dim=-1)

        # (bsz, max_num)
        key_padding_mask = torch.arange(max_num, device=device).expand(bsz, -1) >= real_num.unsqueeze(1)
        return key_padding_mask


class GPT2Decoder(nn.Module):
    def __init__(self, *, gpt2_model: GPT2Model, dropout=0.1, bos_token_id: int, eos_token_id: int, pad_token_id: int):
        super(GPT2Decoder, self).__init__()
        self.gpt2 = gpt2_model

        gpt2_config = GPT2Config()
        self.hidden_size = gpt2_config.n_embd
        self.lm_head = nn.Linear(self.hidden_size, gpt2_config.vocab_size, bias=False)

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.factor_gate_S = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.factor_gate_P = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.factor_gate_T = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.factor_gate_E = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(
        self,
        *,
        context=None,
        P_g=None,
        T_g=None,
        E_g=None,
        dec_input_ids,
        dec_attention_mask,
        dec_token_type_ids,
        labels,
        max_generate_length=None,
    ):
        """
        Params:
            context, P_g, T_g, E_g: (bsz, hidden_size)
            dec_input_ids, dec_attention_mask, dec_token_type_ids,: (bsz, seq_len)
        """

        if self.training:  # train mode, teacher-forcing
            model_inputs = {
                "input_ids": dec_input_ids,
                "attention_mask": dec_attention_mask,
                "token_type_ids": dec_token_type_ids,
            }
            model_outputs = self.gpt2(**model_inputs)
            # model_outputs: last_hidden_state, past_key_values
            last_hidden_state = model_outputs["last_hidden_state"]  # (bsz, seq_len, hidden_size)
            # past_key_values = model_outputs["past_key_values"]

            last_hidden_state = self.add_factors(last_hidden_state, context, P_g, T_g, E_g)
            lm_logits = self.lm_head(last_hidden_state)

        else:  # eval mode, greedy-search
            assert dec_input_ids.shape[0] == 1, "在验证模式下，batch_size > 1 时可能存在问题，不建议使用。"
            # 当 batch_size 设置为 1 时：
            if dec_input_ids.shape[0] == 1:
                model_inputs, labels = self.get_eval_model_inputs_and_labels(
                    dec_input_ids, dec_attention_mask, dec_token_type_ids, labels
                )
            else:  # 当 batch_size 大于 1 时：
                model_inputs, labels = self.get_eval_model_inputs_and_labels_multi_sample(
                    dec_input_ids, dec_attention_mask, dec_token_type_ids, labels
                )

            generate_length = labels.shape[-1] if max_generate_length is None else max_generate_length
            # 用于保存生成的 hidden_state, list <- (1, 1, vocab_size)
            lm_logits = []
            for _ in range(generate_length):
                model_outputs = self.gpt2(**model_inputs)
                last_hidden_state = model_outputs["last_hidden_state"]  # (1, seq_len, hidden_size)
                past_key_values = model_outputs["past_key_values"]

                # 1. 获取生成的最后一个 hidden_state
                hidden_state = last_hidden_state[:, -1:, :]  # (1, 1, hidden_size)
                # 2. 为最后一个 hidden_state 加上 factors
                hidden_state = self.add_factors(hidden_state, context, P_g, T_g, E_g)
                # 3. 通过分类器得到 logit
                logit: torch.Tensor = self.lm_head(hidden_state)  # (1, 1, vocab_size)
                # 4. 保存该 logit
                lm_logits.append(logit)
                # 5. 根据 logit 获取生成的 token 对应的 id
                new_id = logit.argmax(dim=-1)  # (bsz=1, seq_len=1)
                # 6. 更新 model_inputs
                model_inputs = self.update_eval_model_inputs(model_inputs, new_id)
                # 7. 更新 past_key_valuse
                model_inputs["past_key_values"] = past_key_values

                # import pdb
                # pdb.set_trace()
            # 将logits中的元素在第一维上拼接
            lm_logits = torch.cat(lm_logits, dim=1)  # (1, seq_len, vocab_size)

        if self.training or max_generate_length is None:
            # 根据 labels 和 lm_logits 计算 loss
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss().to(labels.device)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            log_logit(lm_logits[0][:-1], labels[0][1:], self.training)
        else:
            loss = None

        return {"loss": loss, "lm_logits": lm_logits}

    def get_eval_model_inputs_and_labels(self, input_ids, attention_mask, token_type_ids, labels):
        """
        Params:
            input_ids, attention_mask, token_type_ids, labels: (1, seq_len)
        """
        # 1. 根据labels找到 target response 的起始位置
        response_start_index = torch.where(labels != -100)[1].min()
        # 2. 根据labels找到填充的起始位置
        pad_start_index = torch.where(labels != -100)[1].max() + 1
        # 3. 修改 input_ids, attention_mask, token_type_ids, 删除其中 response 部分和 pad 部分
        # 此处 +1 是为了将 response 前的那个 [sep] 加到输入里
        input_ids = input_ids[:, : response_start_index + 1]
        attention_mask = attention_mask[:, : response_start_index + 1]
        token_type_ids = token_type_ids[:, : response_start_index + 1]

        # 4. 修改 labels, 使其只包括 response 和其前后的 [sep],[eos]
        labels = labels[:, response_start_index:pad_start_index]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, labels

    def get_eval_model_inputs_and_labels_multi_sample(self, input_ids, attention_mask, token_type_ids, labels):
        """
        Params:
            input_ids, attention_mask, token_type_ids, labels: (bsz, seq_len)
        """
        # 1. 根据 labels 找到每个行中 target response 的 [sep] 之后的位置
        # 此处 >1 是为了排除 [sep] 的位置
        response_positions_without_sep = (labels != -100).cumsum(dim=1) > 1
        # 2. 根据 response_positions 修改 input_ids 中对应位置为 pad_ids, 同时修改attention_mask
        input_ids[response_positions_without_sep] = self.pad_token_id
        attention_mask[response_positions_without_sep] = 0
        # 3. 修改 labels, 使其只包括 response 和其前后的 [sep],[eos]
        # 3.1 获取所有labels中最大长度
        mask = labels != -100
        non_pad_counts = mask.sum(dim=1)
        max_len = non_pad_counts.max().item()
        # 3.2 新建一个 new_labels, 其元素为原始labels + -100
        new_labels = torch.full((labels.size(0), max_len), -100, dtype=labels.dtype, device=labels.device)
        for i, count in enumerate(non_pad_counts):
            new_labels[i, :count] = labels[i, mask[i]]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}, new_labels

    def update_eval_model_inputs(self, model_inputs, id):
        """
        Params:
            model_inputs: {input_ids, attention_mask, token_type_ids (1, seq_len)}
            id: (bsz=1, seq_len=1)
        """
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        token_type_ids = model_inputs["token_type_ids"]

        # input_ids = torch.cat([input_ids, id], dim=-1)
        # attention_mask = torch.cat(
        #     [attention_mask, torch.tensor([[1]], dtype=torch.int, device=attention_mask.device)], dim=-1
        # )
        # token_type_ids = torch.cat([token_type_ids, token_type_ids[:, -1:]], dim=-1)

        input_ids = id
        attention_mask = torch.ones_like(input_ids, device=attention_mask.device)
        token_type_ids = torch.full(input_ids.shape, token_type_ids[0, -1], device=token_type_ids.device)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

    def add_factors(self, last_hidden_state, context: torch.Tensor = None, P_g=None, T_g=None, E_g=None):
        add_tensor = torch.zeros_like(last_hidden_state, device=last_hidden_state.device)
        seq_len = last_hidden_state.shape[1]
        factor_num = 0

        for factor, gate in zip(
            [context, P_g, T_g, E_g], [self.factor_gate_S, self.factor_gate_P, self.factor_gate_T, self.factor_gate_E]
        ):
            if factor is not None:
                factor = factor.unsqueeze(1).repeat(1, seq_len, 1)
                g = torch.sigmoid(gate(torch.cat([last_hidden_state, factor], dim=-1)))
                add_tensor = add_tensor + g * factor + (1 - g) * last_hidden_state
                factor_num += 1

        # import pdb
        # pdb.set_trace()
        # return last_hidden_state + add_tensor
        if factor_num == 0:
            return last_hidden_state
        return add_tensor / factor_num


class ChatBot(nn.Module):
    def __init__(
        self,
        *,
        encoder_model_path: str,
        decoder_model_path: str,
        rnn_n_layers: int,
        factor: str = "SPTE",
        emotion_n_class: int = 28,
        dropout: float = 0.1,
        freeze_enc: bool = False,
        freeze_dec: bool = False,
    ):
        """
        factor 可以为 空字符串、None、SPTE的随机组合
        """
        super(ChatBot, self).__init__()

        roberta_config = RobertaConfig()
        gpt2_config = GPT2Config()

        self.emotion_n_class = emotion_n_class
        self.enc_hidden_size = roberta_config.hidden_size
        self.enc_vocab_size = roberta_config.vocab_size
        self.dec_hidden_size = gpt2_config.n_embd
        self.dec_vocab_size = gpt2_config.vocab_size
        # 生成所用的条件
        if factor is None:
            self.factors = []
        else:
            # S:context语义信息，P:个性信息，T:主题信息，E:情感信息
            self.factors = [
                factor.strip() for factor in " ".join(factor).split(" ") if factor.strip() in ["S", "P", "T", "E"]
            ]

        # import pdb
        # pdb.set_trace()

        # Encoder
        roberta: RobertaModel = RobertaModel.from_pretrained(encoder_model_path)
        self.encoder = RobertaRNNEncoder(roberta, rnn_n_layers=rnn_n_layers, dropout=dropout)

        # PTE 生成
        self.factor_generator = FactorGenerator(hidden_size=self.enc_hidden_size, dropout=dropout)

        # 将encoder的输出维度转换为deocder的输入维度
        self.context_enc2dec = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)
        self.P_g_enc2dec = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)
        self.T_g_enc2dec = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)
        self.E_g_enc2dec = nn.Linear(self.enc_hidden_size, self.dec_hidden_size)

        # Decoder
        gpt2_model = GPT2Model.from_pretrained(decoder_model_path, pad_token_id=gpt2_config.eos_token_id)
        self.decoder = GPT2Decoder(
            gpt2_model=gpt2_model,
            dropout=dropout,
            bos_token_id=gpt2_config.bos_token_id,
            eos_token_id=gpt2_config.eos_token_id,
            pad_token_id=gpt2_config.eos_token_id,
        )

        # classifier of emo_his, topic_his, emo_g, topic_g
        self.emotion_his_classifier = nn.Linear(self.enc_hidden_size, self.emotion_n_class)
        self.topic_his_classifier = nn.Linear(self.enc_hidden_size, self.enc_vocab_size)
        self.emotion_g_classifier = nn.Linear(self.enc_hidden_size, self.emotion_n_class)
        self.topic_g_classifier = nn.Linear(self.enc_hidden_size, self.enc_vocab_size)

        if freeze_enc:
            self.freeze_parameters(self.encoder.roberta)
        if freeze_dec:
            self.freeze_parameters(self.decoder.gpt2)

    def forward(
        self,
        *,
        context_ids,
        context_attn_mask,
        persona_ids,
        persona_attn_mask,
        dec_input_ids,
        dec_attention_mask,
        dec_token_type_ids,
        labels,
        max_generate_length: int = None,
    ):
        if len(self.factors) != 0:
            # context: (bsz, hidden_size)           每个batch的上下文级语义表征
            encoder_outputs: dict = self.encoder(
                context_ids=context_ids,
                context_attn_mask=context_attn_mask,
                persona_ids=persona_ids,
                persona_attn_mask=persona_attn_mask,
            )

            # (bsz, hidden_size)
            context = encoder_outputs["context"]
            # (bsz, max_num_context, hidden_size)
            emotion_his = encoder_outputs["emotion_his"]
            # (bsz, max_num_context, hidden_size)
            topic_his = encoder_outputs["topic_his"]
            # (bsz, max_num_persona, hidden_size)
            persona = encoder_outputs["persona"]

            # 根据 context, emotion_his, topic_his, persona 获得用于生成的 P_g, T_g, E_g
            P_g, T_g, E_g = self.factor_generator(
                context, persona, emotion_his, topic_his, context_attn_mask, persona_attn_mask
            )

        # 维度转换
        context = self.context_enc2dec(context) if "S" in self.factors else None
        P_g = self.P_g_enc2dec(P_g) if "P" in self.factors else None
        T_g = self.T_g_enc2dec(T_g) if "T" in self.factors else None
        E_g = self.E_g_enc2dec(E_g) if "E" in self.factors else None

        # import pdb
        # pdb.set_trace()

        # out: (bsz, tgt_seq_len, hidden_size)
        decoder_outputs = self.decoder(
            context=context,
            P_g=P_g,
            T_g=T_g,
            E_g=E_g,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            dec_token_type_ids=dec_token_type_ids,
            labels=labels,
            max_generate_length=max_generate_length,
        )
        # reconstruction_loss = decoder_outputs["loss"]
        # lm_logits = decoder_outputs["lm_logits"]

        # 返回 topic_kw, emotion的分类logits
        if "E" in self.factors:
            decoder_outputs["emotion_his_logits"] = self.emotion_his_classifier(emotion_his)
            decoder_outputs["emotion_g_logits"] = self.emotion_g_classifier(E_g)
        if "T" in self.factors:
            decoder_outputs["topic_his_logits"] = self.topic_his_classifier(topic_his)
            decoder_outputs["topic_g_logits"] = self.topic_g_classifier(T_g)

        return decoder_outputs

    def get_update_parameters(self):
        updata_params = []
        for param in self.parameters():
            if param.requires_grad == True:
                updata_params.append(param)
        return updata_params

    def freeze_parameters(self, model):
        for name, param in model.named_parameters():
            param.requires_grad = False
