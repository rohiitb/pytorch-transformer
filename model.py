import torch
import torch.nn as nn
import math


class EmbeddingLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        # (batch_size, seq_length) --> (batch_size, seq_length, d_model)
        x = self.embedding(x)
        return x * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        seq_length: int,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        self.positional_encoding = torch.zeros(self.seq_length, d_model)
        position = torch.arange(0, seq_length, dtype=torch.float).reshape(seq_length, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.positional_encoding = self.positional_encoding.unsqueeze(0).to(device)

        self.register_buffer("pe", self.positional_encoding)

    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        x = x + self.positional_encoding.requires_grad_(False)
        x = self.dropout(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(self.d_model, self.d_ff)
        self.linear2 = nn.Linear(self.d_ff, self.d_model)

    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff) --> (batch_size, seq_length, d_model)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.num_heads = num_heads  # Number of heads
        self.dropout = nn.Dropout(dropout)

        # Make sure embedding_vector is divisible by num_heads
        assert self.d_model % self.num_heads == 0

        self.d_v = (
            self.d_model // self.num_heads
        )  # Dimension of vector seen by each head

        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

    @staticmethod
    def attention(q, k, v, mask, dropout):
        d_v = q.shape[-1]

        # (batch_size, num_heads, seq_length, d_v) --> (batch, num_heads, seq_length, seq_length)
        attention_scores = (q @ k.mT) / math.sqrt(d_v)
        if mask is not None:
            attention_scores.masked_fill_(mask, 1e-9)

        attention_scores = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch_size, num_heads, seq_length, seq_length) --> (batch_size, num_heads, seq_length, d_v)
        return (attention_scores @ v), attention_scores

    def forward(self, q, k, v, mask):
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # (batch_size, seq_length, d_model) -> (batch, seq_length, num_heads, d_v)
        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.d_v
        ).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_v).permute(
            0, 2, 1, 3
        )
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.d_v
        ).permute(0, 2, 1, 3)

        x, attention_scores = MultiheadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # (batch_size, seq_length, num_heads, d_v) --> (batch_size, seq_length, d_model)
        x = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_v)
        )

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        x = self.W_o(x)

        return x


class ResidualConnectionLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        # (batch_length, seq_length, d_model) --> (batch_length, seq_length, d_model)
        x = x + self.dropout(sub_layer(self.norm(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        self_attention_block: MultiheadAttention,
        feedforward_block: FeedForwardBlock,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connection_1 = ResidualConnectionLayer(self.d_model, dropout)
        self.residual_connection_2 = ResidualConnectionLayer(self.d_model, dropout)

    def forward(self, x, mask):
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, mask)
        )
        x = self.residual_connection_2(x, lambda x: self.feedforward_block(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        self_attention_block: MultiheadAttention,
        cross_attention_block: MultiheadAttention,
        feedforward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connection_1 = ResidualConnectionLayer(self.d_model, dropout)
        self.residual_connection_2 = ResidualConnectionLayer(self.d_model, dropout)
        self.residual_connection_3 = ResidualConnectionLayer(self.d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection_1(
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connection_2(
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, tgt_mask
            ),
        )
        x = self.residual_connection_3(x, lambda x: self.feedforward_block(x))
        return x


class Decoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: EmbeddingLayer,
        tgt_embedding: EmbeddingLayer,
        projection_layer: ProjectionLayer,
        src_pe: PositionalEncoding,
        tgt_pe: PositionalEncoding,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.src_pe = src_pe
        self.tgt_pe = tgt_pe
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch_size, seq_length) --> (batch_size, seq_length, d_model)
        src = self.src_embedding(src)
        src = self.src_pe(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (batch_size, seq_length) --> (batch_size, seq_length, d_model)
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pe(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, vocab_size)
        return self.projection_layer(x)


def build_transformers(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_length: int,
    tgt_seq_length: int,
    num_heads: int = 8,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    dropout: float = 0.1,
    device: torch.device = torch.device("cpu"),
):
    src_embedding = EmbeddingLayer(d_model, src_vocab_size)
    tgt_embedding = EmbeddingLayer(d_model, tgt_vocab_size)

    src_pe = PositionalEncoding(d_model, src_seq_length, dropout, device)
    tgt_pe = PositionalEncoding(d_model, tgt_seq_length, dropout, device)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadAttention(d_model, num_heads, dropout)
        encoder_feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, encoder_feedforward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadAttention(d_model, num_heads, dropout)
        decoder_cross_attention_block = MultiheadAttention(d_model, num_heads, dropout)
        decoder_feedforward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            d_model,
            decoder_self_attention_block,
            decoder_cross_attention_block,
            decoder_feedforward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder_list = nn.ModuleList(encoder_blocks)
    decoder_list = nn.ModuleList(decoder_blocks)

    encoder = Encoder(d_model, encoder_list)
    decoder = Decoder(d_model, decoder_list)

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(
        d_model,
        encoder,
        decoder,
        src_embedding,
        tgt_embedding,
        projection_layer,
        src_pe,
        tgt_pe,
    )

    for p in transformer.parameters():
        if p.dim() > 1:  # Check if the tensor has at least 2 dimensions
            nn.init.xavier_uniform_(p)

    return transformer.to(device)
