from configs.configs import ModelConfig, DataConfig
from models.model import ChatBot as ChatBot
from dataset.data import SPCDataset, PCDataset

from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def get_model(config: ModelConfig, *, ckpt=None, device="cpu"):
    model = ChatBot(
        encoder_model_path=config.encoder_model_path,
        decoder_model_path=config.decoder_model_path,
        rnn_n_layers=config.rnn_n_layers,
        factor=config.factor,
        dropout=config.dropout,
        freeze_enc=config.freeze_enc,
        freeze_dec=config.freeze_dec,
    )
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
    model.to(device)

    with open("model_structure.txt", "w") as f:
        print(model, file=f)
    return model


def prepare_data(config: DataConfig):
    enc_tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_path)
    dec_tokenizer = AutoTokenizer.from_pretrained(config.decoder_model_path)
    dec_tokenizer.pad_token_id = dec_tokenizer.eos_token_id
    dec_tokenizer.sep_token_id = dec_tokenizer.eos_token_id

    if config.dataset == "SPC":
        DataClass = SPCDataset
    elif config.dataset == "PC":
        DataClass = PCDataset
    dataset = DataClass(
        config.split,
        enc_tokenizer,
        dec_tokenizer,
        dataset_dir=config.dataset_dir,
        utterance_truncate=config.utterance_truncate,
        context_utterance_length=config.context_utterance_length,
        input_utterance_length=config.input_utterance_length,
        persona_length=config.persona_length,
    )
    
    if config.all_raw_datas is not None:
        # 由于数据集中的all_raw_datas的类型为list[dict]，
        # 而传入的all_raw_datas类型为dict{list, list, list}
        # 故在此进行一次转换
        tmp_raw_datas = []
        for i in range(len(config.all_raw_datas["context"])):
            tmp_raw_datas.append(
                {
                    "context": config.all_raw_datas["context"][i],
                    "personas": config.all_raw_datas["personas"][i],
                    "response": config.all_raw_datas["response"][i],
                }
            )
        dataset.all_raw_datas = tmp_raw_datas

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=8)

    data_info = {
        "dataset": dataset,
        "dataloader": dataloader,
        "enc_tokenizer": enc_tokenizer,
        "dec_tokenizer": dec_tokenizer,
    }

    return data_info


from configs.configs import REC, EM_HIS, KW_HIS, EM_G, KW_G


def calc_weighted_loss(rec, em_his, kw_his, em_g, kw_g):
    # import pdb
    # pdb.set_trace()
    return REC * rec + EM_HIS * em_his + KW_HIS * kw_his + EM_G * em_g + KW_G * kw_g
