"""
HeatViT: Hardware-Efficient Adaptive Token Pruning Vision Transformer
Applied to Breast Histopathology (IDC) Detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HeadwiseLinear(nn.Module):
    def __init__(self, h, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.randn(h, i, o) * i ** -0.5)
        self.b = nn.Parameter(torch.zeros(h, o))

    def forward(self, x):
        return torch.einsum('bnhi,hio->bnho', x, self.w) + self.b

class TokenSelector(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h, self.d = heads, dim // heads
        d, h = self.d, heads
        self.local = HeadwiseLinear(h, d, d // 2)
        self.mlp1 = HeadwiseLinear(h, d, d // 2)
        self.mlp2 = HeadwiseLinear(h, d // 2, d // 4)
        self.mlp3 = HeadwiseLinear(h, d // 4, 2)
        self.att1, self.att2 = nn.Linear(h, h), nn.Linear(h, h)
        with torch.no_grad():
            self.mlp3.b[:, 0] = 3.0

    def forward(self, x, policy):
        B, N, D = x.shape
        xh = x.reshape(B, N, self.h, self.d)
        loc = F.gelu(self.local(xh))
        pol = policy[:, :, None, None].to(loc.dtype)
        glob_ = (loc * pol).sum(1, keepdim=True) / (pol.sum(1, keepdim=True) + 1e-6)
        e = torch.cat([loc, glob_.expand(-1, N, -1, -1)], -1)
        s = F.softmax(self.mlp3(F.gelu(self.mlp2(F.gelu(self.mlp1(e))))).float(), -1)
        a = torch.sigmoid(self.att2(F.gelu(self.att1(xh.mean(-1))))).float()
        return (s * a.unsqueeze(-1)).sum(2) / (a.sum(-1, keepdim=True) + 1e-6)

def attn_policy(attn, x, pol):
    B, T_, C = x.shape
    H = attn.num_heads
    q, k, v = attn.qkv(x).reshape(B, T_, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
    q = getattr(attn, 'q_norm', nn.Identity())(q)
    k = getattr(attn, 'k_norm', nn.Identity())(k)
    a = (q @ k.transpose(-2, -1)).float() * (C // H) ** -0.5
    a = a - a.amax(-1, keepdim=True)
    eye = torch.eye(T_, device=x.device, dtype=a.dtype)[None, None]
    p = pol[:, None, None, :].to(a.dtype)
    p = p + (1 - p) * eye
    e = a.exp() * p
    a = (e / (e.sum(-1, keepdim=True) + 1e-6)).to(v.dtype)
    out = (attn.attn_drop(a) @ v).transpose(1, 2).reshape(B, T_, C)
    return attn.proj_drop(attn.proj(out))

def block_policy(blk, x, pol):
    ident = nn.Identity()
    ls1, ls2 = getattr(blk, 'ls1', ident), getattr(blk, 'ls2', ident)
    dp1 = getattr(blk, 'drop_path1', getattr(blk, 'drop_path', ident))
    dp2 = getattr(blk, 'drop_path2', getattr(blk, 'drop_path', ident))
    x = x + dp1(ls1(attn_policy(blk.attn, blk.norm1(x), pol)))
    return x + dp2(ls2(blk.mlp(blk.norm2(x))))

def count_macs(tokens, D):
    return 12 * tokens * D * D + 2 * tokens * tokens * D

class HeatViT(nn.Module):
    """
    HeatViT: Multi-head learned token selector with progressive token packaging.
    """
    def __init__(self, backbone, sel_blocks=(3, 6, 9), keep=(0.90, 0.80, 0.60), use_package=True, mode='mh'):
        super().__init__()
        self.bb = backbone
        self.use_package = use_package
        self.mode = mode
        self.sel_blocks = list(sel_blocks)
        self.keep = list(keep)
        D, H = backbone.embed_dim, backbone.blocks[0].attn.num_heads
        self.D = D
        self.nc = backbone.num_classes
        if mode == 'static':
            self.selectors = nn.ModuleList()
        else:
            self.selectors = nn.ModuleList([TokenSelector(D, 1 if mode == 'sh' else H) for _ in self.sel_blocks])
        g = int(round(math.sqrt(backbone.patch_embed.num_patches)))
        yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing='ij')
        dist = ((yy - (g - 1) / 2) ** 2 + (xx - (g - 1) / 2) ** 2).float().sqrt().flatten()
        prior = 1 - dist / dist.max() * 0.999 + torch.arange(g * g).float() * 1e-6
        self.register_buffer('prior', prior)

    def _score(self, j, xp, policy, pos=None):
        if self.mode == 'static':
            pr = self.prior if pos is None else self.prior[pos]
            pr = pr.unsqueeze(0).expand(xp.shape[0], -1) if pr.dim() == 1 else pr
            return torch.stack([pr, 1 - pr], -1).float()
        return self.selectors[j](xp, policy)

    def forward(self, img):
        """Masked forward pass used during training and compute analysis."""
        bb = self.bb
        x = bb.patch_embed(img)
        x = torch.cat([bb.cls_token.expand(x.shape[0], -1, -1), x], 1) + bb.pos_embed
        x = bb.pos_drop(x)
        B, N = x.shape[0], x.shape[1] - 1
        policy = x.new_ones(B, N)
        extra, keep_masks, macs = 0, [], 0.
        for i, blk in enumerate(bb.blocks):
            if i in self.sel_blocks:
                j = self.sel_blocks.index(i)
                xp = x[:, 1:1 + N]
                score = self._score(j, xp, policy)
                p_keep = score[..., 0]
                if self.training:
                    k_soft = p_keep
                    pol_all = torch.cat([policy.new_ones(B, 1), policy * k_soft, policy.new_ones(B, extra)], 1)
                    policy = policy * k_soft
                    keep_masks.append(policy)
                else:
                    k = max(1, int(round(self.keep[j] * N)))
                    _, idx = (p_keep * policy).topk(k, dim=1)
                    hard = torch.zeros_like(p_keep).scatter_(1, idx, 1.0)
                    pol_all = torch.cat([policy.new_ones(B, 1), hard, policy.new_ones(B, extra + (1 if self.use_package else 0))], 1)
                    keep_masks.append(hard)
                    if self.use_package:
                        w = ((1 - hard) * p_keep * policy).unsqueeze(-1)
                        pkg = (xp * w).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1e-6)
                        x = torch.cat([x, pkg], 1)
                        extra += 1
                    policy = hard
            else:
                pol_all = torch.cat([policy.new_ones(B, 1), policy, policy.new_ones(B, extra)], 1)
            macs = macs + count_macs(pol_all.sum(1), self.D)
            x = block_policy(blk, x, pol_all)
        macs = macs + N * 768 * self.D + self.D * self.nc
        logits = bb.forward_head(bb.norm(x))
        return logits, dict(keep_masks=keep_masks, macs=macs)

    @torch.no_grad()
    def forward_dense(self, img, return_keep_indices=False):
        """Physically pruned forward pass: token sequence actually shrinks for maximum speed."""
        bb = self.bb
        x = bb.patch_embed(img)
        x = torch.cat([bb.cls_token.expand(x.shape[0], -1, -1), x], 1) + bb.pos_embed
        x = bb.pos_drop(x)
        B = x.shape[0]
        n_alive = x.shape[1] - 1
        pos = torch.arange(n_alive, device=x.device).unsqueeze(0).expand(B, -1)
        all_indices = []

        for i, blk in enumerate(bb.blocks):
            if i in self.sel_blocks:
                j = self.sel_blocks.index(i)
                xp = x[:, 1:1 + n_alive]
                score = self._score(j, xp, x.new_ones(B, n_alive), pos)
                p_keep = score[..., 0]
                r = self.keep[j] / (self.keep[j - 1] if j > 0 else 1.0)
                k = max(1, int(round(r * n_alive)))
                idx = p_keep.topk(k, dim=1).indices
                kept = xp.gather(1, idx.unsqueeze(-1).expand(-1, -1, xp.shape[-1]))
                old_extra = x[:, 1 + n_alive:]
                if self.use_package:
                    hard_k = torch.zeros_like(p_keep).scatter_(1, idx, 1.0)
                    w = (p_keep * (1 - hard_k)).unsqueeze(-1)
                    pkg = (xp.float() * w).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1e-6)
                    x = torch.cat([x[:, :1], kept, old_extra, pkg.to(x.dtype)], 1)
                else:
                    x = torch.cat([x[:, :1], kept, old_extra], 1)
                pos = pos.gather(1, idx)
                all_indices.append(pos)
                n_alive = k
            x = blk(x)
        logits = bb.forward_head(bb.norm(x))
        if return_keep_indices:
            return logits, all_indices
        return logits

def build_heatvit(num_classes=2, checkpoint_path=None, device='cpu'):
    """Helper to instantiate and optionally load pre-trained HeatViT weights."""
    backbone = timm.create_model('deit_tiny_patch16_224', pretrained=False, num_classes=num_classes)
    model = HeatViT(backbone, sel_blocks=(3, 6, 9), keep=(0.90, 0.80, 0.60)).to(device)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded HeatViT checkpoint from: {checkpoint_path}")
    model.eval()
    return model
