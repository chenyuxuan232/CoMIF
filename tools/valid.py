import torch
import torch.nn as nn

from dataset.data import SPCDataset, batch_to_device
from utils.utils import DataLogger, pack_batch
from configs.configs import ValidConfig, DataConfig, ModelConfig
from .utils import prepare_data, get_model, calc_weighted_loss

from tqdm import tqdm
import os
import time


def valid_model(config: ValidConfig):
    loadFilename = os.path.join(config.ckpt_dir, config.model_name, f"{config.dataset}-{config.epoch_idx}e.ckpt")
    ckpt = torch.load(loadFilename, map_location="cpu")

    data_config = DataConfig(dataset=config.dataset, split="valid", batch_size=config.batch_size, shuffle=False)
    data_info = prepare_data(data_config)
    dataset: SPCDataset = data_info["dataset"]
    dataloader = data_info["dataloader"]

    model_config = ModelConfig(factor=config.factor)
    model = get_model(model_config, ckpt=ckpt, device=config.device)

    # 以下均为多标签多分类损失
    loss_fns = {
        "em_his": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("em_his", config.device)),
        "kw_his": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("kw_his", config.device)),
        "em_g": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("em_g", config.device)),
        "kw_g": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("kw_g", config.device)),
    }

    valid(
        model,
        dataloader,
        loss_fns,
        output_dir=config.valid_output_dir,
        model_name=config.model_name,
        dataset_name=config.dataset,
        epoch_idx=config.epoch_idx,
        device=config.device,
    )


def valid(model, dataloader, loss_fns, *, output_dir, model_name, dataset_name, epoch_idx, device):
    model.eval()
    loss_logger = DataLogger()
    tbar = tqdm(dataloader)
    i = 0
    for batch in tbar:
        batch = batch_to_device(batch=batch, device=device)

        losses = valid_loop(model, batch, loss_fns)

        loss_logger.add(losses)

        tbar.set_description(
            "Valid {}-{:2d}e -loss:{:4.4f} -rec:{:4.4f} -em_his:{:4.4f} -kw_his:{:4.4f} -em_g:{:4.4f} -kw_g:{:4.4f}".format(
                model_name,
                epoch_idx,
                loss_logger.avg_all(key="loss"),
                loss_logger.avg_all(key="reconstruction_loss"),
                loss_logger.avg_all(key="em_his_loss"),
                loss_logger.avg_all(key="kw_his_loss"),
                loss_logger.avg_all(key="em_g_loss"),
                loss_logger.avg_all(key="kw_g_loss"),
            ),
            refresh=True,
        )
        tbar.refresh()

        i += 1
        # if i == 10:
        #     break

    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{dataset_name}-{epoch_idx}e.txt"), "w") as f:
        time.timezone = 8 * 3600  # 东八区
        print(
            "{:2d}e -loss:{:4.4f} -rec:{:4.4f} -em_his:{:4.4f} -kw_his:{:4.4f} -em_g:{:4.4f} -kw_g:{:4.4f} -time:{}".format(
                epoch_idx,
                loss_logger.avg_all(key="loss"),
                loss_logger.avg_all(key="reconstruction_loss"),
                loss_logger.avg_all(key="em_his_loss"),
                loss_logger.avg_all(key="kw_his_loss"),
                loss_logger.avg_all(key="em_g_loss"),
                loss_logger.avg_all(key="kw_g_loss"),
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            ),
            file=f,
        )


def valid_loop(model, batch, loss_fns):
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
    )
    reconstruction_loss = outputs["loss"]
    lm_logits = outputs["lm_logits"]

    # 计算损失函数
    # 1. 根据 context_attn_mask 计算每个batch中句子数量
    real_num = (context_attn_mask.sum(dim=-1) != 0).sum(dim=-1)  # (bsz,)
    # 2. 计算 emotion_his
    if "emotion_his_logits" in outputs:
        em_his = pack_batch(outputs["emotion_his_logits"], real_num)  # (bsz * max_num - pad, n_class)
        em_his_labels = pack_batch(batch["his_em_dis"], real_num)  # (bsz * max_num - pad, n_class)
        em_his_loss = loss_fns["em_his"](em_his, em_his_labels.float())
    else:
        em_his_loss = torch.tensor(0)
    # 3. 计算 keyword_his
    if "topic_his_logits" in outputs:
        kw_his = pack_batch(outputs["topic_his_logits"], real_num)  # (bsz * max_num - pad, n_class)
        kw_his_labels = pack_batch(batch["his_kw_dis"], real_num)  # (bsz * max_num - pad, n_class)
        kw_his_loss = loss_fns["kw_his"](kw_his, kw_his_labels.float())
    else:
        kw_his_loss = torch.tensor(0)
    # 4. 计算 emotion_generate
    if "emotion_g_logits" in outputs:
        em_g_loss = loss_fns["em_g"](outputs["emotion_g_logits"], batch["tgt_em_dis"].float())
    else:
        em_g_loss = torch.tensor(0)
    # 5. 计算 keyword_generate
    if "topic_g_logits" in outputs:
        kw_g_loss = loss_fns["kw_g"](outputs["topic_g_logits"], batch["tgt_kw_dis"].float())
    else:
        kw_g_loss = torch.tensor(0)
    # 6. 计算加权损失
    loss = calc_weighted_loss(reconstruction_loss, em_his_loss, kw_his_loss, em_g_loss, kw_g_loss)

    # import pdb
    # pdb.set_trace()

    return {
        "loss": loss.cpu().item(),
        "reconstruction_loss": reconstruction_loss.cpu().item(),
        "em_his_loss": em_his_loss.cpu().item(),
        "kw_his_loss": kw_his_loss.cpu().item(),
        "em_g_loss": em_g_loss.cpu().item(),
        "kw_g_loss": kw_g_loss.cpu().item(),
    }
