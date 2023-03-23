from lib.models.language_model.bert import BERT
# from lib.checkpoints.language_model.bert_huggingface import BERT_HUGGINGFACE


def build_bert(cfg):
    # position_embedding = build_position_encoding(cfg)
    train_bert = cfg.MODEL.LANGUAGE.BERT.LR > 0
    bert_type = cfg.MODEL.LANGUAGE.IMPLEMENT
    if bert_type == "pytorch":
        bert_model = BERT(cfg.MODEL.LANGUAGE.TYPE, cfg.MODEL.LANGUAGE.PATH, train_bert,
                          cfg.MODEL.LANGUAGE.BERT.HIDDEN_DIM,
                          cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN, cfg.MODEL.LANGUAGE.BERT.ENC_NUM)
    else:
        raise ValueError("Undefined BERT TYPE '%s'" % bert_type)
    return bert_model
