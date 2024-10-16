import datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.tensorboard import SummaryWriter

import datetime
import os
import pickle
import time

from dataset import BilingualDataset, causal_mask
from config import get_config
from model import build_transformers


def get_sentences(ds, lang):
    for one_item in ds:
        yield one_item["translation"][lang]


def build_tokenizer(ds, lang):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(get_sentences(ds, lang), trainer=trainer)

    return tokenizer


def get_dataset(config):
    ds_raw = load_dataset(
        config["datasource"],
        f"{config['src_lang']}-{config['tgt_lang']}",
        split="train",
    )

    print(f"GPU {config['local_rank']} - Loading tokenizers...")

    tokenizer_path = config["tokenizer_path"]
    src_tokenizer_file = os.path.join(tokenizer_path, "src_tokenizer.pkl")
    tgt_tokenizer_file = os.path.join(tokenizer_path, "tgt_tokenizer.pkl")

    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    if os.path.exists(src_tokenizer_file) and os.path.exists(tgt_tokenizer_file):
        with open(src_tokenizer_file, "rb") as f:
            src_tokenizer = pickle.load(f)
        with open(tgt_tokenizer_file, "rb") as f:
            tgt_tokenizer = pickle.load(f)
        print("Loaded existing tokenizers")
    else:
        src_tokenizer = build_tokenizer(ds_raw, config["src_lang"])
        tgt_tokenizer = build_tokenizer(ds_raw, config["tgt_lang"])
        with open(src_tokenizer_file, "wb") as f:
            pickle.dump(src_tokenizer, f)
        with open(tgt_tokenizer_file, "wb") as f:
            pickle.dump(tgt_tokenizer, f)
        print("Generated and saved new tokenizers")

    train_ds_raw_size = int(0.9 * len(ds_raw))
    val_ds_raw_size = len(ds_raw) - train_ds_raw_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_raw_size, val_ds_raw_size]
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        src_tokenizer,
        tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_length"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        src_tokenizer,
        tgt_tokenizer,
        config["src_lang"],
        config["tgt_lang"],
        config["seq_length"],
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = src_tokenizer.encode(item["translation"][config["src_lang"]]).ids
        tgt_ids = tgt_tokenizer.encode(item["translation"][config["tgt_lang"]]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=DistributedSampler(train_ds, shuffle=True),
    )
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer


def greedy_decode(model, src, src_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    sos_idx = torch.tensor(tgt_tokenizer.token_to_id("[SOS]"), device=device)
    eos_idx = torch.tensor(tgt_tokenizer.token_to_id("[EOS]"), device=device)

    encoder_output = model.encode(src, src_mask)
    decoder_input = torch.empty(1, 1, dtype=src.dtype, device=device).fill_(sos_idx)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        output = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        proj = model.project(output[:, -1])
        next_word = torch.argmax(proj, dim=1)

        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input


def run_validation(
    model, val_dataloader, src_tokenizer, tgt_tokenizer, max_len, device, num_examples=2
):
    print("Running validation")
    model.eval()
    count = 0

    src_texts = []
    tgt_texts = []
    predicted_texts = []

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)

            output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                src_tokenizer,
                tgt_tokenizer,
                max_len,
                device,
            )

            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            model_output_text = tgt_tokenizer.decode(output.detach().cpu().numpy())

            src_texts.append(src_text)
            tgt_texts.append(tgt_text)
            predicted_texts.append(model_output_text)

            print(f"Source:     {src_text}")
            print(f"Target:     {tgt_text}")
            print(f"Predicted:  {model_output_text}")

            count += 1
            if count == num_examples:
                break

    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted_texts, tgt_texts)
    print(f"BLEU score: {bleu}")

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted_texts, tgt_texts)
    print(f"WER score: {wer}")


def load_checkpoint(checkpoint_path, model, optimizer):
    if not os.path.exists(checkpoint_path):
        return 0, None, 0, []

    checkpoints = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
    if not checkpoints:
        return 0, None, 0, []

    latest_checkpoint = max(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    checkpoint = torch.load(os.path.join(checkpoint_path, latest_checkpoint))

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    global_step = checkpoint["global_step"]
    initial_epoch = checkpoint["epoch"]
    loss_list = checkpoint["loss_list"]

    return global_step, latest_checkpoint, initial_epoch, loss_list


def train_model_distributed(config):

    assert torch.cuda.is_available(), "CUDA is not available, Training not supported"
    device = torch.device("cuda")

    print(f"Using device {device}")

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_dataset(config)

    model = build_transformers(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        src_seq_length=config["seq_length"],
        tgt_seq_length=config["seq_length"],
        N=6,
        d_model=config["d_model"],
        device=device,
    )
    model = DistributedDataParallel(model, device_ids=[config["local_rank"]])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tgt_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    num_epochs = config["num_epochs"]

    # Create a SummaryWriter instance
    log_dir = "runs/transformer_training"
    checkpoint_path = config["checkpoint_path"]
    writer = SummaryWriter(log_dir)

    # Check for existing checkpoints and load if available
    global_step, latest_checkpoint, initial_epoch, loss_list = load_checkpoint(
        checkpoint_path, model, optimizer
    )

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        print(f"GPU {config['local_rank']} - Starting training from scratch")
        os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(initial_epoch, num_epochs):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        loss_per_batch = 0
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )

            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tgt_tokenizer.get_vocab_size()), label.view(-1)
            )
            loss_per_batch += loss.item()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # Log the loss to TensorBoard
            writer.add_scalar("Training Loss", loss.item(), global_step)
            global_step += 1

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

        loss_per_batch /= len(train_dataloader)
        loss_list.append(loss_per_batch)

        if config["local_rank"] == 0:
            run_validation(
                model,
                val_dataloader,
                src_tokenizer,
                tgt_tokenizer,
                config["seq_length"],
                device,
                num_examples=1,
            )

            # Store the model checkpoint after each epoch
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "global_step": global_step,
                "loss_list": loss_list,
            }
            torch.save(checkpoint, f"{checkpoint_path}/model_epoch_{epoch+1}.pt")

    # Close the SummaryWriter
    writer.close()


@click.command()
@click.option("--batch_size", type=int, default=get_config()["batch_size"])
@click.option("--num_epochs", type=int, default=get_config()["num_epochs"])
@click.option("--lr", type=float, default=get_config()["lr"])
@click.option("--seq_length", type=int, default=get_config()["seq_length"])
@click.option("--d_model", type=int, default=get_config()["d_model"])
@click.option("--src_lang", type=str, default=get_config()["src_lang"])
@click.option("--tgt_lang", type=str, default=get_config()["tgt_lang"])
@click.option("--datasource", type=str, default=get_config()["datasource"])
@click.option("--tokenizer_path", type=str, default=get_config()["tokenizer_path"])
@click.option("--checkpoint_path", type=str, default=get_config()["checkpoint_path"])
def main():
    config = get_config()

    # Update config with command line arguments
    config.update(locals())
    config["local_rank"] = int(os.environ["LOCAL_RANK"])
    config["global_rank"] = int(os.environ["RANK"])

    assert config["local_rank"] != -1, "LOCAL_RANK is not set"
    assert config["global_rank"] != -1, "RANK is not set"

    if config["local_rank"] == 0:
        print("Using config: ")
        for k, v in config.items():
            print(f"{k}: {v}")

    init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(config["local_rank"])

    train_model_distributed(config)

    destroy_process_group()


if __name__ == "__main__":
    main()
