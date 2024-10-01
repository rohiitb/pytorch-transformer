def get_config():
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_length": 500,
        "d_model": 512,
        "datasource": "Helsinki-NLP/opus_books",
        "src_lang": "en",
        "tgt_lang": "fr",
        "tokenizer_path": "tokenizers",
    }