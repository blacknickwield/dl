import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math


class Config:
    def __init__(self, vocab_size: int, d_model: int, d_q: int, d_k: int, d_v: int, heads: int, num_hidden: int, num_encoder_block: int, num_decoder_block: int, dropout_prob) -> None:
        self.vocab_size = vocab_size
        self.d_moedl = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads
        self.num_hidden = num_hidden
        self.dropout_prob = dropout_prob
        self.encoder_blocks = num_encoder_block
        self.decoder_blocks = num_decoder_block


config = Config(vocab_size=100, d_model=512, d_q=64, d_k=64, d_v=64, heads=8, num_hidden=1024, num_encoder_block=5, num_decoder_block=5, dropout_prob=0.1)

class PositionEmbedding(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        seq_len, d_model = X.size(-2), X.size(-1)
        position = torch.zeros(seq_len, d_model)
        for i in range(d_model):
            position[:, i] = torch.sin(torch.arange(0, seq_len) / math.pow(1e4, 2 * i / d_model)) if i % 2 == 0 else torch.cos(torch.arange(0, seq_len) / math.pow(1e4, 2 * i / d_model))

        return position + X


class SelfAttention(nn.Module):
    def __init__(self, d_k: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_k = d_k
        self.scale = 1 / math.sqrt(self.d_k)

    '''
    Q: [..., seq_len, d_q]
    K: [..., seq_len, d_k]
    V: [..., seq_len, d_V]
    '''
    def forward(self, Q: Tensor, K: Tensor, V: Tensor):
        scores = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) * self.scale, dim=-1)
        attention = torch.matmul(scores, V)
        return V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, d_q: int, d_k: int, d_v: int, *args, **kwargs) -> None:
        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.d_model = d_model
        self.heads = heads
        self.w_q = nn.Linear(in_features=d_model, out_features=heads * d_q)
        self.w_k = nn.Linear(in_features=d_model, out_features=heads * d_k)
        self.w_v = nn.Linear(in_features=d_model, out_features=heads * d_v)
        self.attention = SelfAttention(d_k=d_k)
        
    '''
    X: [batch_size, seq_len, d_model]
    Q: [batch_size, seq_len, heads * d_q]
    K: [batch_size, seq_len, heads * d_k]
    V: [batch_size, seq_len, heads * d_v]
    '''
    def forward(self, X: Tensor, mask: Tensor=None, Y: Tensor=None):
        Q, K, V = self.w_q(X), self.w_k(X), self.w_v(X)
        if Y is not None:
            Q, K, V = self.w_q(X), self.w_k(Y), self.w_v(Y)

        Q, K, V = self._transpose_input(Q), self._transpose_input(K), self._transpose_input(V)
        attention = self.attention(Q, K, V)
        if mask is not None:
            pass
        attention = self._transpose_output(attention)
        return attention

    def _transpose_input(self, X: Tensor) -> Tensor:
        batch_size, seq_len, _ = X.shape
        X = X.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)
        return X


    def _transpose_output(self, X: Tensor) -> Tensor:
        batch_size, heads, seq_len, d = X.shape
        X = X.permute(0, 2, 1, 3).view(batch_size, seq_len, -1)
        return X

class FFN(nn.Module):
    def __init__(self, d_model: int, hiddens: int, dropout_prob: float, *args, **kwargs) -> None:
        super(FFN, self).__init__(*args, **kwargs)
        self.fc1 = nn.Linear(d_model, hiddens)
        self.fc2 = nn.Linear(hiddens, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: Tensor) -> Tensor:
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = self.fc2(X)

        return X        

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int, heads: int, hiddens: int, dropout_prob: float, *args, **kwargs) -> None:
        super(EncoderBlock, self).__init__(*args, **kwargs)
        self.mha = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, heads=heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, hiddens=hiddens, dropout_prob=dropout_prob)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, X: Tensor) -> Tensor:
        X = self.ln1(X + self.mha(X))
        X = self.ln2(X + self.ffn(X))
        return X

class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_q: int, d_k: int, d_v: int, heads: int, hiddens: int, dropout_prob: float, num_encoder: int, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding()
        self.encoder_blocks = nn.ModuleList(EncoderBlock(d_model=config.d_moedl, heads=config.heads, d_q=config.d_q, d_k=config.d_k, d_v=config.d_v, hiddens=config.num_hidden, dropout_prob=config.dropout_prob) for _ in range(num_encoder))


    def forward(self, X: Tensor) -> Tensor:
        X = self.embedding(X)
        X = self.pos_embedding(X)
        for block in self.encoder_blocks:
            X = block(X)

        return X

class DecoderBLock(nn.Module):
    def __init__(self, d_model: int, d_q: int, d_k: int, d_v: int, heads: int, hiddens: int, dropout_prob: float, *args, **kwargs) -> None:
        super(DecoderBLock, self).__init__(*args, **kwargs)
        self.mha1 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, heads=heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha2 = MultiHeadAttention(d_model=d_model, d_q=d_q, d_k=d_k, d_v=d_v, heads=heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model=d_model, hiddens=hiddens, dropout_prob=dropout_prob)
        self.ln3 = nn.LayerNorm(d_model)


    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        X = self.ln1(X + self.mha1(X))
        X = self.ln2(X + self.mha2(X, Y))
        X = self.ln3(X + self.ffn(X))

        return X

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_q: int, d_k: int, d_v: int, heads: int, hiddens: int, dropout_prob: float, num_decoder: int, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = PositionEmbedding()
        self.decoder_blocks = nn.ModuleList(DecoderBLock(d_model=config.d_moedl, heads=config.heads, d_q=config.d_q, d_k=config.d_k, d_v=config.d_v, hiddens=config.num_hidden, dropout_prob=config.dropout_prob) for _ in range(num_decoder))

    def forward(self, X: Tensor, Y: Tensor):
        X = self.embedding(X)
        X = self.pos_embedding(X)
        for block in self.decoder_blocks:
            X = block(X, Y)

        return X

class Transformer(nn.Module):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder(vocab_size=config.vocab_size, d_model=config.d_moedl, heads=config.heads, d_q=config.d_q, d_k=config.d_k, d_v=config.d_v, hiddens=config.num_hidden, dropout_prob=config.dropout_prob, num_encoder=config.encoder_blocks)
        self.decoder = Decoder(vocab_size=config.vocab_size, d_model=config.d_moedl, heads=config.heads, d_q=config.d_q, d_k=config.d_k, d_v=config.d_v, hiddens=config.num_hidden, dropout_prob=config.dropout_prob, num_decoder=config.decoder_blocks)


    def forward(self, X: Tensor):
        Y = self.encoder(X)
        Y = self.decoder(X, Y)

        return Y

if __name__ == '__main__':
    model = Transformer(config=config)
    model = model
    X = torch.randn(8, 64).type(torch.int).clamp(0, 100)
    Y = model(X)
    print(Y.shape)