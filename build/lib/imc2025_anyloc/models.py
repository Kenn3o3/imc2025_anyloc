import os
import torch
from torch import nn
from torch.nn import functional as F
import fast_pytorch_kmeans as fpk
import numpy as np
import einops as ein

class VisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, num_classes=0):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation=F.gelu,
                batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.head = nn.Identity() if num_classes == 0 else nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class DinoV2ExtractFeatures:
    def __init__(self, model_path, layer, facet="value", use_cls=False, norm_descs=True, device="cpu"):
        self.device = torch.device(device)
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self._hook_out = None
        
        self.dino_model = VisionTransformer(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            num_classes=0
        )
        
        if not (0 <= layer < 24):
            raise ValueError(f"Layer index {layer} is out of range for dinov2_vitl14 (0 to 23)")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DINOv2 weights not found at {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.dino_model.load_state_dict(state_dict, strict=False)
        
        self.dino_model = self.dino_model.eval().to(self.device)
        
        try:
            if self.facet == "token":
                self.fh_handle = self.dino_model.blocks[self.layer].register_forward_hook(self._generate_forward_hook())
            else:
                self.fh_handle = self.dino_model.blocks[self.layer].self_attn.register_forward_hook(self._generate_forward_hook())
        except Exception as e:
            print(f"Failed to register forward hook: {e}")
            self.fh_handle = None
            raise RuntimeError(f"Cannot proceed without forward hook: {e}")
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            if isinstance(module, nn.MultiheadAttention):
                attn_output = output[0] if isinstance(output, tuple) else output
                embed_dim = attn_output.shape[-1]
                d_len = embed_dim // 3
                if self.facet == "query":
                    self._hook_out = attn_output[:, :, :d_len]
                elif self.facet == "key":
                    self._hook_out = attn_output[:, :, d_len:2*d_len]
                else:
                    self._hook_out = attn_output[:, :, 2*d_len:]
            else:
                self._hook_out = output
        return _forward_hook
    
    def __del__(self):
        if hasattr(self, 'fh_handle') and self.fh_handle is not None:
            self.fh_handle.remove()
    
    def __call__(self, img):
        if not hasattr(self, 'fh_handle') or self.fh_handle is None:
            raise RuntimeError("Forward hook not initialized")
        with torch.no_grad():
            res = self.dino_model(img)
            if self._hook_out is None:
                raise RuntimeError("Forward hook did not capture output")
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None
        return res

class VLAD:
    def __init__(self, num_clusters, desc_dim=None, intra_norm=True, norm_descs=True, dist_mode="cosine", vlad_mode="hard", soft_temp=1.0, cache_dir=None):
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.mode = dist_mode
        self.vlad_mode = vlad_mode.lower()
        assert self.vlad_mode in ['soft', 'hard']
        self.soft_temp = soft_temp
        self.cache_dir = cache_dir
        self.c_centers = None
        self.kmeans = None
    
    def fit(self, train_descs):
        self.kmeans = fpk.KMeans(self.num_clusters, mode=self.mode)
        if train_descs is None:
            raise ValueError("No training descriptors given")
        if isinstance(train_descs, np.ndarray):
            train_descs = torch.from_numpy(train_descs).to(torch.float32)
        if self.desc_dim is None:
            self.desc_dim = train_descs.shape[1]
        if self.norm_descs:
            train_descs = F.normalize(train_descs)
        self.kmeans.fit(train_descs)
        self.c_centers = self.kmeans.centroids
    
    def generate(self, query_descs, cache_id=None):
        residuals = self.generate_res_vec(query_descs, cache_id)
        un_vlad = torch.zeros(self.num_clusters * self.desc_dim)
        if self.vlad_mode == 'hard':
            labels = self.kmeans.predict(query_descs)
            used_clusters = set(labels.numpy())
            for k in used_clusters:
                cd_sum = residuals[labels==k, k].sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        else:
            cos_sims = F.cosine_similarity(
                ein.rearrange(query_descs, "q d -> q 1 d"), 
                ein.rearrange(self.c_centers, "c d -> 1 c d"), 
                dim=2)
            soft_assign = F.softmax(self.soft_temp * cos_sims, dim=1)
            for k in range(self.num_clusters):
                w = ein.rearrange(soft_assign[:, k], "q -> q 1 1")
                cd_sum = ein.rearrange(w * residuals, "q c d -> (q c) d").sum(dim=0)
                if self.intra_norm:
                    cd_sum = F.normalize(cd_sum, dim=0)
                un_vlad[k*self.desc_dim:(k+1)*self.desc_dim] = cd_sum
        return F.normalize(un_vlad, dim=0)
    
    def generate_res_vec(self, query_descs, cache_id=None):
        assert self.kmeans is not None and self.c_centers is not None
        if isinstance(query_descs, np.ndarray):
            query_descs = torch.from_numpy(query_descs).to(torch.float32)
        if self.norm_descs:
            query_descs = F.normalize(query_descs)
        residuals = ein.rearrange(query_descs, "q d -> q 1 d") - ein.rearrange(self.c_centers, "c d -> 1 c d")
        return residuals