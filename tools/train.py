import torch
import torch.nn as nn

import os
import math
from tqdm import tqdm

from dataset.data import SPCDataset, batch_to_device
from utils.utils import DataLogger, pack_batch
from configs.configs import TrainConfig, DataConfig, ModelConfig
from .utils import prepare_data, get_model, calc_weighted_loss


def train_model(config: TrainConfig):
    ckpt = None
    if config.loadFilename is not None:
        ckpt = torch.load(config.loadFilename, map_location="cpu")

    # 加载数据集
    data_config = DataConfig(
        dataset=config.dataset,
        split="train",
        batch_size=config.batch_size,
        shuffle=True,
    )
    data_info = prepare_data(data_config)
    dataset: SPCDataset = data_info["dataset"]
    dataloader = data_info["dataloader"]

    # 加载模型
    model_config = ModelConfig(factor=config.factor)
    model = get_model(model_config, ckpt=ckpt, device=config.device)

    # 初始化/加载优化器
    optimizer = torch.optim.Adam(model.get_update_parameters(), lr=config.learning_rate)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    # 以下均为多标签多分类损失
    loss_fns = {
        "em_his": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("em_his", config.device)),
        "kw_his": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("kw_his", config.device)),
        "em_g": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("em_g", config.device)),
        "kw_g": nn.BCEWithLogitsLoss(pos_weight=dataset.get_pos_weight("kw_g", config.device)),
    }

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        loss_fns=loss_fns,
        epoches=config.epoches,
        clip=config.clip,
        save_dir=config.ckpt_save_dir,
        model_name=config.model_name,
        dataset_name=config.dataset,
        checkpoint=ckpt,
        device=config.device,
    )


def train(
    *,
    model,
    dataloader,
    optimizer,
    loss_fns,
    epoches,
    clip,
    save_dir,
    model_name,
    dataset_name,
    checkpoint=None,
    device,
):

    start_epoch = 1 if (checkpoint is None) else (checkpoint["epoch"] + 1)

    loss_logger = DataLogger(num=10)

    if checkpoint is not None:
        loss_logger.__dict__ = checkpoint["loss_logger"]

    model.train()
    for epoch in range(start_epoch, epoches + 1):
        epoch_loss, i = 0, 0
        tbar = tqdm(dataloader)
        for batch in tbar:
            batch = batch_to_device(batch=batch, device=device)

            losses = train_loop(model, optimizer, batch, loss_fns, clip=clip)

            epoch_loss += losses["loss"]
            i += 1
            loss_logger.add(losses)

            tbar.set_description(
                "Epoch:{:2d} -epoch_loss:{:4.4f} -rec:{:4.4f} -em_his:{:4.4f} -kw_his:{:4.4f} -em_g:{:4.4f} -kw_g:{:4.4f}".format(
                    epoch,
                    epoch_loss / i,
                    loss_logger.avg(key="reconstruction_loss"),
                    loss_logger.avg(key="em_his_loss"),
                    loss_logger.avg(key="kw_his_loss"),
                    loss_logger.avg(key="em_g_loss"),
                    loss_logger.avg(key="kw_g_loss"),
                ),
                refresh=True,
            )
            tbar.refresh()

            # if i == 10:
            #     break

        directory = os.path.join(save_dir, model_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = os.path.join(directory, f"{dataset_name}-{epoch}e.ckpt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": epoch_loss / i,
                "loss_logger": loss_logger.__dict__,
            },
            save_path,
        )


def train_loop(model, optimizer, batch, loss_fns, *, clip):
    context_ids = batch["context"]["input_ids"]
    context_attn_mask = batch["context"]["attention_mask"]

    persona_ids = batch["personas"]["input_ids"]
    persona_attn_mask = batch["personas"]["attention_mask"]

    dec_input_ids = batch["inputs"]["input_ids"]
    dec_attention_mask = batch["inputs"]["attention_mask"]
    dec_token_type_ids = batch["inputs"]["token_type_ids"]

    labels = batch["labels"]

    # 梯度清空
    optimizer.zero_grad()

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

    # 反向计算
    loss.backward()

    # 对model进行梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    # 更新参数
    optimizer.step()

    if math.isnan(loss):
        with open("/home/cyx/workspace/ChatBot/outputs/test_log.txt", "a") as f:
            print("lm_logits:", file=f)
            print(lm_logits, file=f)
        raise ValueError("nan!!!!!!!!!!")

    return {
        "loss": loss.cpu().item(),
        "reconstruction_loss": reconstruction_loss.cpu().item(),
        "em_his_loss": em_his_loss.cpu().item(),
        "kw_his_loss": kw_his_loss.cpu().item(),
        "em_g_loss": em_g_loss.cpu().item(),
        "kw_g_loss": kw_g_loss.cpu().item(),
    }
