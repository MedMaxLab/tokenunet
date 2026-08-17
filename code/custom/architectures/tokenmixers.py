import torch
import torch.nn as nn
import math

__all__ = [
    "MyMLP",
    "MyMLPMixer",
    "MyMLPMixerBlock",
    "MySelfAttention",
    "MyTransformerEncoderLayer",
    "MyTransformerEncoder"
]

class MyMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        
        self.Wa = nn.Parameter(torch.randn(dim_feedforward, d_model))
        self.Wb = nn.Parameter(torch.randn(d_model, dim_feedforward))
        self.act = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor([-2.5]))

    def forward(self, x):
        res = nn.functional.linear(x, self.Wa / math.sqrt(self.d_model))
        res = self.act(res)
        
        if self.dropout > 0.0:
            res = nn.functional.dropout(res, p=self.dropout, training=self.training)
            
        res = nn.functional.linear(res, self.Wb / math.sqrt(self.dim_feedforward / 2.))
        
        # Replaced deprecated nn.functional.sigmoid with torch.sigmoid
        alpha = torch.sigmoid(self.alpha)
        
        return alpha * res + (1. - alpha) * x

class MyMLPMixerBlock(nn.Module):
    def __init__(self, n_tokens, d_model, dim_feedforward, dropout=0.0):
        super().__init__()
        
        # Token mixing: operates across the token sequence length
        # Input to this MLP will be transposed to have n_tokens as the last dimension
        self.token_mix = MyMLP(
            d_model=n_tokens, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        
        # Channel mixing: operates across the feature dimension
        # Input to this MLP remains in standard (B, n_tokens, d_model) format
        self.channel_mix = MyMLP(
            d_model=d_model, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )

    def forward(self, x):
        # x shape: (B, n_tokens, d_model)
        
        # 1. Token Mixing
        x = x.transpose(1, 2)         # Shape: (B, d_model, n_tokens)
        x = self.token_mix(x)         # MyMLP handles its own residual internally
        x = x.transpose(1, 2)         # Shape back to: (B, n_tokens, d_model)
        
        # 2. Channel Mixing
        x = self.channel_mix(x)       # Shape: (B, n_tokens, d_model)
        
        return x

class MyMLPMixer(nn.Module):
    def __init__(self, n_tokens, d_model, dim_feedforward, n_blocks, dropout=0.0):
        super().__init__()
        
        # Fixed the `nhead` copy-paste artifact and passed the correct variables
        self.encoder = nn.ModuleList([
            MyMLPMixerBlock(
                n_tokens=n_tokens, 
                d_model=d_model, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
    def norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])
    def forward(self, x):
        # x shape: (B, n_tokens, d_model)
        x = self.norm(x)
        for block in self.encoder:
            x = block(x)
        return x

class MySelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model  = d_model
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        #self.norm_qk  = nn.RMSNorm(self.head_dim, elementwise_affine=False)
        #self.norm_o   = nn.RMSNorm(d_model, elementwise_affine=False)
        self.Wqkv = nn.Parameter(torch.randn(d_model * 3, d_model))
        self.Wo   = nn.Parameter(torch.randn(d_model, d_model))
        self.alpha = nn.Parameter(torch.tensor([-2.5]))
    
    def norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        q, k, v = (
            nn.functional.linear(x, self.Wqkv / math.sqrt(self.d_model))
            .unflatten(-1, (3, self.nhead, self.head_dim))
            .unbind(dim=2)
        )
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = key_padding_to_attn_mask(mask, q.dtype) if mask is not None else None

        o = nn.functional.scaled_dot_product_attention(
            self.norm(q), self.norm(k), v, attn_mask=attn_mask
        )

        o = self.norm(o.transpose(1, 2).flatten(-2))
        o = nn.functional.linear(o, self.Wo / math.sqrt(self.d_model))
        alpha = nn.functional.sigmoid(self.alpha)
        return alpha * o + (1. - alpha) * x


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MySelfAttention(d_model, nhead, dropout)
        self.mlp       = MyMLP(d_model, dim_feedforward, dropout)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.mlp(x)
        return x


class MyTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers=12, *args, **kwargs):
        super().__init__()
        #self.norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.encoder = nn.ModuleList([
            MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    def norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])

    def forward(self, x, mask=None):
        x = self.norm(x)
        for layer in self.encoder:
            x = layer(x, mask)
        return x