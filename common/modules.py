import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttnFuse(nn.Module):
    """
    v_a: [B, N]  (primary)
    v_b: [B, M]  (aux, less reliable)

    return: [B, N] (same as v_a)
    """
    def __init__(self, d_model=64, n_heads=4, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # 先把标量扩到通道维
        self.q_proj = nn.Linear(1, d_model)
        self.k_proj = nn.Linear(1, d_model)
        self.v_proj = nn.Linear(1, d_model)

        self.out_proj = nn.Linear(d_model, 1)  # 压回标量
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 可学习门控，初值 0 -> 刚开始基本等于恒等映射
        self.gate = nn.Parameter(torch.tensor(0.0))

    def _reshape_heads(self, x):  # [B, T, d_model] -> [B, h, T, d_head]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x):    # [B, h, T, d_head] -> [B, T, d_model]
        B, h, T, dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * dh)

    def forward(self, v_a, v_b, mask_b=None):
        """
        v_a: [B, N]
        v_b: [B, M]
        mask_b: [B, M]，无效位置为 False/0（可选）
        """
        B, N = v_a.shape
        M = v_b.shape[1]

        # 升维到通道
        qa = self.q_proj(v_a.unsqueeze(-1))  # [B, N, d]
        kb = self.k_proj(v_b.unsqueeze(-1))  # [B, M, d]
        vb = self.v_proj(v_b.unsqueeze(-1))  # [B, M, d]

        # 规范化后做注意力更稳
        qa = self.norm(qa)

        # 多头
        q = self._reshape_heads(qa)  # [B, h, N, dh]
        k = self._reshape_heads(kb)  # [B, h, M, dh]
        v = self._reshape_heads(vb)  # [B, h, M, dh]

        # 为了数值稳定，attention 在 fp32 做，再回到原 dtype
        dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()

        # 构造 mask（SDPA 里 True=可见，False/−inf=屏蔽）
        attn_mask = None
        if mask_b is not None:
            # mask_b: [B, M] (True 有效)，扩成 [B, h, N, M]
            attn_mask = mask_b[:, None, None, :].expand(B, self.n_heads, N, M)
            # PyTorch SDPA 期望的是加性掩码/布尔掩码，这里使用布尔即可：
            # True=keep, False=mask。F.scaled_dot_product_attention 接受布尔 mask。
        
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)  # [B, h, N, dh]
        ctx = ctx.to(dtype)
        ctx = self._merge_heads(ctx)          # [B, N, d]
        ctx = self.dropout(ctx)

        # 压回标量偏移量，并用门控控制来自 v_b 的影响强度
        delta = self.out_proj(ctx).squeeze(-1)  # [B, N]
        g = torch.sigmoid(self.gate)            # 标量 ∈ (0,1)

        # 残差：以 v_a 为主，微调一小步
        fused = v_a + g * delta                 # [B, N]
        return fused


class CrossFromScalars(nn.Module):
    """
    a = rhand_feat: [B, 21, 512] (Q)
    b = v_b:        [B, M]       (标量序列 -> K/V)
    out:            [B, 21, 512]
    """
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, use_pos_emb=True, max_M=512):
        super().__init__()
        self.b_proj = nn.Linear(1, d_model)            # 标量 -> 512
        self.attn   = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ffn    = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)
        self.gate   = nn.Parameter(torch.tensor(0.0))  # 可学习门控

        self.use_pos = use_pos_emb
        if use_pos_emb:
            self.pos_b = nn.Embedding(max_M, d_model)  # 给 b 的 M 个位置加位置编码

    def forward(self, a, b, mask_b=None):
        """
        a: [B, 21, 512]
        b: [B, M]
        mask_b: [B, M] (True=有效) 可选
        """
        B, M = b.shape
        # 标量 -> token
        b_tok = self.b_proj(b.unsqueeze(-1))           # [B, M, 512]
        if self.use_pos:
            idx = torch.arange(M, device=b.device)[None, :].expand(B, M)  # [B, M]
            b_tok = b_tok + self.pos_b(idx)            # 位置编码

        # cross-attn: Q=a, K/V=b_tok
        a_n = self.norm_q(a)
        key_padding_mask = (~mask_b) if mask_b is not None else None  # True=pad(屏蔽)
        ctx, _ = self.attn(a_n, b_tok, b_tok, key_padding_mask=key_padding_mask)  # [B,21,512]
        delta = self.ffn(ctx)
        out = a + torch.sigmoid(self.gate) * delta
        return self.norm_o(out)                         # [B,21,512]
