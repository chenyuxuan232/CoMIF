import torch

from dataset.data import batch_to_device, get_raw_data
from configs.configs import PredictConfig, DataConfig, ModelConfig
from .utils import prepare_data, get_model

import os
import pandas as pd
from tqdm import tqdm


def do_predict(config: PredictConfig):
    loadFilename = os.path.join(config.ckpt_dir, config.model_name, f"{config.dataset}-{config.epoch_idx}e.ckpt")
    ckpt = torch.load(loadFilename, map_location="cpu")

    data_config = DataConfig(dataset=config.dataset, split="test", batch_size=config.batch_size, shuffle=False)
    data_info = prepare_data(data_config)
    dataset = data_info["dataset"]
    dataloader = data_info["dataloader"]
    enc_tokenizer = data_info["enc_tokenizer"]
    dec_tokenizer = data_info["dec_tokenizer"]

    model_config = ModelConfig(factor=config.factor)
    model = get_model(model_config, ckpt=ckpt, device=config.device)

    preidct(
        model,
        dataset,
        dataloader,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer,
        output_dir=config.predict_output_dir,
        model_name=config.model_name,
        dataset_name=config.dataset,
        epoch_idx=config.epoch_idx,
        max_generate_length=config.max_generate_length,
        device=config.device,
        ablation=config.ablation,
    )


def preidct(
    model,
    dataset,
    dataloader,
    *,
    enc_tokenizer,
    dec_tokenizer,
    output_dir,
    model_name,
    dataset_name,
    epoch_idx,
    max_generate_length,
    device,
    ablation,
):
    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{dataset_name}-{epoch_idx}e.json")

    print("Genate output dir: ", output_file)

    contexts = []
    target_responses = []
    responses = []
    personas = []
    his_em_ids = []
    his_kw_ids = []

    model.eval()

    tbar = tqdm(dataloader, "Generating...")
    i = 0
    for batch in tbar:
        i += 1
        # 每10条数据取一条进行预测
        if i % 10 != 0:
            continue

        batch = batch_to_device(batch=batch, device=device)

        output = predict_loop(
            model, batch, enc_tokenizer, dec_tokenizer, max_generate_length=max_generate_length, ablation=ablation
        )

        contexts.extend(output["contexts"])
        target_responses.extend(output["target_responses"])
        responses.extend(output["responses"])
        personas.extend(output["personas"])
        his_em_ids.extend([get_raw_data(dataset, batch["index"][0], "his_emotion_ids")])
        his_kw_ids.extend([get_raw_data(dataset, batch["index"][0], "his_keyword_ids")])

        if i == 10000:
            break
        # if i == 1000:
        #     break

    

    # TODO:save to output_file
    df = pd.DataFrame()
    df["context"] = contexts
    df["response"] = responses
    df["target_response"] = target_responses
    df["personas"] = personas
    df["his_em_ids"] = his_em_ids
    df["his_kw_ids"] = his_kw_ids
    df.to_json(output_file, orient="records", force_ascii=False)


def predict_loop(model, batch, enc_tokenizer, dec_tokenizer, *, max_generate_length, ablation):
    context_ids = batch["context"]["input_ids"]
    context_attn_mask = batch["context"]["attention_mask"]

    persona_ids = batch["personas"]["input_ids"]
    persona_attn_mask = batch["personas"]["attention_mask"]

    dec_input_ids = batch["inputs"]["input_ids"]
    dec_attention_mask = batch["inputs"]["attention_mask"]
    dec_token_type_ids = batch["inputs"]["token_type_ids"]

    labels = batch["labels"]

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
        max_generate_length=max_generate_length,
        ablation_study=ablation,
    )
    reconstruction_loss = outputs["loss"]
    lm_logits: torch.Tensor = outputs["lm_logits"]

    output_ids = lm_logits.argmax(dim=-1)

    def ids2text(ids, tokenizer, attn_mask=None, is_response=False) -> list[str]:
        """
        Params:
            ids: (nums, seq_len) or (seq_len)
            mask_attn: (nums, seq_len)
        """

        if len(ids.shape) == 1:
            if is_response:
                # 当解码response时，需要截取 [eos] 前的id
                eos_positions = torch.where(ids == tokenizer.eos_token_id)[0]
                ids = ids[: eos_positions.min()] if eos_positions.numel() != 0 else ids
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        elif len(ids.shape) == 2:
            if attn_mask is None:
                raise ValueError("'attn_mask' 不应为 None")
            real_num = (attn_mask.sum(dim=-1) != 0).sum()  # (bsz,)
            ids = ids[:real_num, ...]
            return [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in ids]
        else:
            raise ValueError("'ids' 应该为一维或二维张量")

    # decode
    contexts = []
    target_responses = []
    responses = []
    personas = []

    batch_size = context_ids.shape[0]
    tgt_ids = labels.clone().detach()
    tgt_ids[labels == -100] = dec_tokenizer.pad_token_id
    for i in range(batch_size):
        # print("==", context_ids[i].shape)
        # print("++", context_attn_mask[i].shape)
        contexts.append(ids2text(context_ids[i], enc_tokenizer, context_attn_mask[i]))
        target_responses.append(ids2text(tgt_ids[i], dec_tokenizer))
        responses.append(ids2text(output_ids[i], dec_tokenizer, is_response=True))
        personas.append(ids2text(persona_ids[i], enc_tokenizer, persona_attn_mask[i]))

        # with open("test.txt", "w") as f:
        #     print(context_ids[0], file=f)
        #     print(context_attn_mask[0], file=f)
        # raise ValueError()

    return {
        "contexts": contexts,
        "target_responses": target_responses,
        "responses": responses,
        "personas": personas,
    }
