import torch
import torch.nn as nn
import math

from .tokenmixers import *

__all__ = [
    "Block",
    "BlockUp",
    "Stage",
    "StageUp",
    "CNNEnc",
    "CNNDec",
    "SpatialAttentionMaskMaker3d",
    "TokenLearner3d",
    "TokenFuser3d",
    "TokenUNet"
]

class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        p=0.0,
        downsample=False
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, in_channels*2, kernel_size, stride=2 if downsample else 1, padding=1),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm3d(in_channels*2),
            nn.Conv3d(in_channels*2, out_channels, kernel_size=1,padding="same"),
        )            
        nn.init.kaiming_normal_(self.block[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.block[-1].weight, mode='fan_in', nonlinearity='relu')
        
        self.res_alpha = nn.Parameter(torch.tensor([-2.5]))
        adjust_volume = nn.AvgPool3d(2,2) if downsample else nn.Identity()
        adjust_channels = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.adjust = nn.Sequential(adjust_volume, adjust_channels)
       
    def forward(self, x):
        res = self.block(x)
        alpha = torch.sigmoid(self.res_alpha)
        x = self.adjust(x)
        x = (1.-alpha)*x + alpha*res
        return x

class BlockUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        p=0.0,
        upsample=False
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.Conv3d(in_channels, in_channels*2, kernel_size, padding=1 ),
            nn.LeakyReLU(0.1),
            nn.InstanceNorm3d(in_channels*2),
            nn.ConvTranspose3d(in_channels*2, out_channels, kernel_size=2, stride=2) if upsample else nn.Conv3d(in_channels*2, out_channels, kernel_size=1, stride=1),
        )            
        nn.init.kaiming_normal_(self.block[1].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.block[-1].weight, mode='fan_in', nonlinearity='relu')
        
        self.res_alpha = nn.Parameter(torch.tensor([-2.5]))
        adjust_volume = nn.Upsample(scale_factor=2, mode="trilinear") if upsample else nn.Identity()
        adjust_channels = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.adjust = nn.Sequential(adjust_volume, adjust_channels)
       
    def forward(self, x):
        res = self.block(x)
        alpha = torch.sigmoid(self.res_alpha)
        x = self.adjust(x)
        x = (1.-alpha)*x + alpha*res
        return x

class Stage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_blocks,
        downsample=True
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(in_channels, in_channels, kernel_size) for _ in range(n_blocks-1)]+
            [Block(in_channels, out_channels, kernel_size, downsample=downsample)]
        )
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class StageUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_blocks,
        upsample=True
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BlockUp(in_channels, in_channels, kernel_size) for _ in range(n_blocks-1)]+
            [BlockUp(in_channels, out_channels, kernel_size, upsample=upsample)]
        )
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CNNEnc(nn.Module):
    def __init__(
        self,
        in_channels,
        stage_channels,
        kernel_size,
        blocks_per_stage
    ):
        super().__init__()

        n_stages = len(stage_channels)

        # Normalise blocks_per_stage: accept a single int or a per-stage list
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * n_stages
        assert len(blocks_per_stage) == n_stages, (
            f"blocks_per_stage length ({len(blocks_per_stage)}) "
            f"must match the number of stages ({n_stages})"
        )

        self.norm = nn.InstanceNorm3d(in_channels)

        # Build stages; the output of stage i is the input of stage i+1
        in_ch = in_channels
        self.stages = nn.ModuleList()
        first = 0
        for out_ch, n_blocks in zip(stage_channels, blocks_per_stage):
            self.stages.append(
                Stage(in_ch, out_ch, kernel_size, n_blocks, downsample=first>0)#last!=n_stages)
            )
            first += 1
            in_ch = out_ch

    def forward(self, x):
        x = self.norm(x)
        maps_to_decoder = []
        for stage in self.stages:
            x = stage(x)
            maps_to_decoder.append(x)
            #print(x.shape)

        return maps_to_decoder

class CNNDec(nn.Module):
    def __init__(
        self,
        in_channels,
        stage_channels,
        kernel_size,
        blocks_per_stage
    ):
        super().__init__()

        n_stages = len(stage_channels)

        # Normalise blocks_per_stage: accept a single int or a per-stage list
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * n_stages
        assert len(blocks_per_stage) == n_stages, (
            f"blocks_per_stage length ({len(blocks_per_stage)}) "
            f"must match the number of stages ({n_stages})"
        )

        #self.norm = nn.InstanceNorm3d(in_channels)

        # Build stages; the output of stage i is the input of stage i+1
        in_ch = in_channels
        self.stages = nn.ModuleList()
        #last = 0
        for out_ch, n_blocks in zip(stage_channels, blocks_per_stage):
            self.stages.append(
                StageUp(in_ch, out_ch, kernel_size, n_blocks, upsample=True)
            )
            in_ch = out_ch
            #last += 1

    def forward(self, maps_to_decoder):
        x = 0.0
        for idx,stage in enumerate(self.stages):
            x = stage(maps_to_decoder[idx]+x)
            #print(maps_to_decoder[idx].shape)
        return x

class SpatialAttentionMaskMaker3d(nn.Module):
    """
    For each spatial location in the feature map, learns a direct linear mapping
    from the local feature vector to N token assignment scores (one per token).
    A sigmoid then turns each score into an independent soft weight in (0, 1).

    Input:  feature map of shape (B, C, H, W, D)
    Output: attention masks of shape (B, n_tokens, H, W, D), values in (0, 1)
    """

    def __init__(self, in_channels: int, n_tokens: int = 8, bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.n_tokens = n_tokens
        #self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        # Single linear layer: maps each voxel's C-dim feature vector directly
        # to n_tokens scalar logits. Acts like a learned 1x1x1 convolution.
        # (B, n_voxels, C) -> (B, n_voxels, n_tokens)
        self.linear = nn.Linear(in_features=in_channels, out_features=n_tokens, bias=bias)

    def norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])
    
    def forward(self, feature_map_bchwd: torch.Tensor):
        """
        Args:
            feature_map_bchwd: (B, C, H, W, D)

        Returns:
            attention_masks_bnhwd: (B, n_tokens, H, W, D)
        """
        B, C, H, W, D = feature_map_bchwd.shape
        # assert C == self.in_channels, f"Expected in_channels={self.in_channels}, got {C}"

        # Flatten spatial dims to treat all voxels uniformly.
        # (B, C, H, W, D) -> (B, C, n_voxels)
        n_voxels = H * W * D
        feature_map_bcv = feature_map_bchwd.reshape(B, C, n_voxels)

        # Move channels last so Linear operates on the feature dimension.
        # (B, C, n_voxels) -> (B, n_voxels, C)
        feature_map_bvc = feature_map_bcv.permute(0, 2, 1)
        feature_map_bvc = self.norm(feature_map_bvc)

        # Direct linear projection: each voxel's feature vector -> n_tokens scores.
        # (B, n_voxels, C) -> (B, n_voxels, n_tokens)
        token_scores_bvn = self.linear(feature_map_bvc)

        # Sigmoid: independent soft weight per (voxel, token) pair, no competition across voxels.
        # (B, n_voxels, n_tokens) -> same shape, values in (0, 1)
        attention_weights_bvn = torch.sigmoid(token_scores_bvn)

        # Restore spatial structure and move token dim before spatial dims.
        # (B, n_voxels, n_tokens) -> (B, n_tokens, n_voxels) -> (B, n_tokens, H, W, D)
        attention_masks_bnhwd = attention_weights_bvn.permute(0, 2, 1).reshape(
            B, self.n_tokens, H, W, D
        )

        return attention_masks_bnhwd

class TokenLearner3d(nn.Module):
    """
    TokenLearner: compresses a volumetric feature map into a small set of learned tokens.

    The key idea:
      1. SpatialAttentionMaskMaker produces N soft spatial masks  (B, N, H, W, D)
      2. Each mask is element-wise multiplied with the feature map  (B, C, H, W, D)
      3. The masked feature map is mean-pooled over (H, W, D) -> one C-dim token per mask
      4. Result: set of N tokens, shape (B, N, C), a compact set-like representation

    This replaces the global average pool (which treats all voxels equally) with
    N learned, content-adaptive pooling operations.

    Args:
        in_channels:    number of feature channels C of the input
        n_tokens:       number of output tokens N  (the compression factor)
        bias:           whether Linear layers use bias
        out_channels:   if set, project each token from C to out_channels via a Linear layer;
                        otherwise tokens keep their original C channels (Identity)
    """

    def __init__(
        self,
        in_channels: int,
        n_tokens: int = 8,
        bias: bool = True,
        out_channels: int = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_tokens = n_tokens


        #self.norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        self.mask_maker = SpatialAttentionMaskMaker3d(
            in_channels=in_channels,
            n_tokens=n_tokens,
            bias=bias,
        )

        # Optional linear projection applied independently to each token vector
        self.token_projector = (
            nn.Linear(in_channels, out_channels) if out_channels else nn.Identity()
        )
    def norm(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1) * math.sqrt(x.shape[-1])

    def forward(self, feature_map_bchwd: torch.Tensor):
        """
        Args:
            feature_map_bchwd: (B, C, H, W, D)  — volumetric feature map

        Returns:
            tokens_bnc:           (B, n_tokens, C_out)  — the learned token set
            attention_masks_bnhwd: (B, n_tokens, H, W, D) — masks for inspection / auxiliary loss
        """
        B, C, H, W, D = feature_map_bchwd.shape
        n_voxels = H * W * D

        # --- Step 1: build one spatial attention mask per token ---
        # Each mask encodes which voxels are relevant for that token.
        # (B, C, H, W, D) -> (B, n_tokens, H, W, D)
        attention_masks_bnhwd = self.mask_maker(feature_map_bchwd)
        attention_masks_flat_bnv = attention_masks_bnhwd.reshape(B, self.n_tokens, n_voxels)

        # --- Step 2: soft-mask the feature map for each token ---
        # Expand feature map along the token dimension so we can broadcast the mask.
        # (B, C, H, W, D) -> (B, 1, C, H, W, D)
        # feature_map_b1chwd = feature_map_bchwd.unsqueeze(1)
        feature_map_bcv = feature_map_bchwd.reshape(B, C, n_voxels)

        # (B, n_tokens, H, W, D) -> (B, n_tokens, 1, H, W, D)  for channel broadcast
        #attention_masks_bn1hwd = attention_masks_bnhwd.unsqueeze(2)

        # Element-wise product: each voxel's feature vector is scaled by its token weight.
        # (B, n_tokens, 1, H, W, D) * (B, 1, C, H, W, D) -> (B, n_tokens, C, H, W, D)
        #masked_features_bnchwd = attention_masks_bn1hwd * feature_map_b1chwd

        
        # --- Step 3: spatial mean-pooling -> one token vector per mask ---
        # Average over the three spatial dimensions (H, W, D).
        # (B, n_tokens, C, H, W, D) -> (B, n_tokens, C)
        #tokens_bnc = masked_features_bnchwd.mean(dim=(3, 4, 5))

        # --- True step 2-3 ---
        # We do not materialize the full (B, n_tokens, C, H, W, D) tensor!
        tokens_bnc = torch.bmm(
            attention_masks_flat_bnv / (attention_masks_flat_bnv.sum(dim=2, keepdims=True) + 1e-8), 
            #self.norm(feature_map_bcv.transpose(1,2))
            feature_map_bcv.transpose(1,2)
            ) #/ n_voxels
        
        # --- Step 4 (optional): project token channels ---
        # (B, n_tokens, C) -> (B, n_tokens, C_out)
        tokens_bnc = self.token_projector(tokens_bnc)

        return tokens_bnc, attention_masks_bnhwd

class TokenFuser3d(nn.Module):
    """
    This module weighted-averages N tokens and computes N spatial pertinence masks, 
    then broadcasts the tokens over the masks to update feature maps. 
    """
    def __init__(self, 
    n_tokens, 
    conv_dim, 
    token_dim,
    bias=True
    ):
        super().__init__()
        self.M = nn.Linear(n_tokens, n_tokens)
        self.Beta = SpatialAttentionMaskMaker3d(
            in_channels=conv_dim,
            n_tokens=n_tokens,
            bias=bias,
        )
        self.n_tokens = n_tokens
        self.conv_dim = conv_dim
        self.token_dim = token_dim
        if (token_dim != conv_dim):
            self.C = nn.Linear(token_dim, conv_dim)
        else:
            self.C = nn.Identity()
        
    def forward(self, tokens, feat_maps):
        B, C, H, W, D = feat_maps.shape # B,Cc,H,[W,[D]]
        n_voxels = H * W * D

        # Mix the tokens, and eventually map them to channel dimension of convolutional features
        mixed_tokens = self.C(self.M(tokens.transpose(1,2)).transpose(1,2)) # B, N, Ct -> B, N, Cc
        
        # Determine how much each voxel needs each token
        pertinence_masks_bnhwd = self.Beta(feat_maps) # B,Cc,H,[W,[D]]
        pertinence_masks_flat_bnv = pertinence_masks_bnhwd.reshape(B, self.n_tokens, n_voxels) # B, N, V=HWD

        token_broadcast_bcv = torch.bmm(mixed_tokens.transpose(1,2), pertinence_masks_flat_bnv) # B, (Cc, N) @ (N, V) -> (B,Ct,V)
        token_broadcast_bchwd = token_broadcast_bcv.reshape(B,C,H,W,D)
        out_maps = feat_maps + token_broadcast_bchwd 
        return out_maps      




class TokenUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        enc_stage_channels,
        kernel_size,
        blocks_per_stage,
        num_classes,
        n_tokens=8,
        token_dim=None,
        tokenize=True,
        attention=False,
        process_tokens=True,
        token_blocks=2,
        bias=True
    ):
        super().__init__()
        self.tokenize = tokenize
        self.process_tokens = process_tokens
        
        # Ensure token_dim is set (defaults to the channel size of the bottleneck)
        bottleneck_channels = enc_stage_channels[-1]
        dec_stage_channels = enc_stage_channels[::-1][1:]
        self.token_dim = token_dim if token_dim is not None else bottleneck_channels

        # 1. Encoder
        self.encoder = CNNEnc(
            in_channels=in_channels,
            stage_channels=enc_stage_channels,
            kernel_size=kernel_size,
            blocks_per_stage=blocks_per_stage
        )

        # 2. Tokenizer (Optional)
        if self.tokenize:
            self.token_learner = TokenLearner3d(
                in_channels=bottleneck_channels,
                n_tokens=n_tokens,
                bias=bias,
                out_channels=self.token_dim
            )
            
            # 3. Token Processor (Transformer / MLP Mixer) - Optional
            if self.process_tokens:
                if attention:
                    self.token_processor = MyTransformerEncoder(
                        d_model=self.token_dim, 
                        nhead=4, 
                        dim_feedforward=self.token_dim*2, 
                        dropout=0.0, 
                        num_layers=token_blocks
                    )
                else:
                    self.token_processor = MyMLPMixer(
                        n_tokens=n_tokens, 
                        d_model=self.token_dim, 
                        dim_feedforward=self.token_dim*2, 
                        n_blocks=token_blocks, 
                        dropout=0.0
                    )
            else:
                self.token_processor = nn.Identity()
            
            # 4. Token Fuser
            self.token_fuser = TokenFuser3d(
                n_tokens=n_tokens,
                conv_dim=bottleneck_channels,
                token_dim=self.token_dim,
                bias=bias
            )

        # 5. Decoder
        self.decoder = CNNDec(
            in_channels=bottleneck_channels, # Starts from the bottleneck size
            stage_channels=dec_stage_channels,
            kernel_size=kernel_size,
            blocks_per_stage=[1,]*len(dec_stage_channels)
        )

        # 6. Segmentation Head
        self.segmentation_head = nn.Conv3d(
            in_channels=dec_stage_channels[-1], 
            out_channels=num_classes, 
            kernel_size=1
        )

    def forward(self, x):
        # Forward through Encoder
        maps_to_decoder = self.encoder(x)
        
        # The bottleneck is the lowest resolution feature map (the last one)
        bottleneck = maps_to_decoder[-1]

        if self.tokenize:
            # Extract Tokens
            tokens, attention_masks = self.token_learner(bottleneck)
            
            # Process Tokens (Transformer/Mixer)
            if self.process_tokens:
                tokens = self.token_processor(tokens)
            
            # Fuse Tokens back into the bottleneck feature map
            bottleneck = self.token_fuser(tokens, bottleneck)
            
            # Update the bottleneck in our skip-connection list
            maps_to_decoder[-1] = bottleneck

        # CRITICAL: Reverse the maps so they match the Decoder's expected ascending resolutions
        # Example: if Enc outputs [128x128, 64x64, 32x32], Decoder needs [32x32, 64x64, 128x128]
        maps_to_decoder = maps_to_decoder[::-1]
        
        # Forward through Decoder
        out = self.segmentation_head(self.decoder(maps_to_decoder))
        
        return out