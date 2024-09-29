import torch
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
  def __init__(self, ds, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, seq_length):
    super().__init__()
    self.ds = ds
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.seq_length = seq_length

    self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
    self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
    self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index):
    src_tgt_pair = self.ds[index]
    src_text = src_tgt_pair["translation"][self.src_lang]
    tgt_text = src_tgt_pair["translation"][self.tgt_lang]

    encoder_token = self.src_tokenizer.encode(src_text).ids
    decoder_token = self.tgt_tokenizer.encode(tgt_text).ids

    num_encoder_pad_tokens = self.seq_length - len(encoder_token) - 2
    num_decoder_pad_tokens = self.seq_length - len(decoder_token) - 1

    if num_encoder_pad_tokens < 0 or num_decoder_pad_tokens < 0:
      raise ValueError("Sentence is too long")

    encoder_input = torch.cat([self.sos_token, torch.tensor(encoder_token, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token]*num_encoder_pad_tokens)])
    decoder_input = torch.cat([self.sos_token, torch.tensor(decoder_token, dtype=torch.int64), torch.tensor([self.pad_token]*num_decoder_pad_tokens, dtype=torch.int64)])
    label = torch.cat([torch.tensor(decoder_token, dtype=torch.int64), self.eos_token, torch.tensor([self.pad_token]*num_decoder_pad_tokens, dtype=torch.int64)])

    assert encoder_input.shape[0] == self.seq_length
    assert decoder_input.shape[0] == self.seq_length
    assert label.shape[0] == self.seq_length

  # TODO: Check the shapes of the return objects
    return {
        "encoder_input": encoder_input,
        "decoder_input": decoder_input,
        "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).type(torch.bool),
        "decoder_mask": (decoder_input != self.pad_token.view(1,-1) & causal_mask(decoder_input.shape[0])).type(torch.bool),
        "label": label,
        "src_text": src_text,
        "tgt_text": tgt_text
    }

def causal_mask(size):
  mask = torch.tril(torch.ones((1,size,size))).type(torch.bool)
  return mask

