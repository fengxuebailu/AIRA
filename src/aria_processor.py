"""
ARIA: Architecture-aware Routing for Identity-preserving Attention.

Components:
  - LSI (Layer-Selective Injection): Restrict concept KV concat to single-stream blocks only.
    Joint-stream blocks propagate cross-subject contamination; single-stream blocks preserve fidelity.
  - GIA (Group Isolation Attention): Each subject's query tokens attend only to their own
    concept KV features, preventing cross-subject identity mixing.
  - CPD (Concept Position Decoupling): Assign each subject's concept tokens distinct RoPE
    positional regions, preventing shared positional encoding from biasing spatial overlap.
"""

from typing import List, Optional
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class ARIAAttnProcessor:
    """Single-subject ARIA attention processor.

    For single-subject, GIA is not needed (no cross-subject contamination).
    LSI is implemented by setting concept_process=True only on single-stream blocks.
    """

    def __init__(self, name: str, mask: torch.Tensor, device: torch.device,
                 tau: float = 0.90, concept_process: bool = False,
                 shift_mask: Optional[torch.Tensor] = None, img_dims: int = 4096):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        self.name = name
        self.device = device
        self.tau = tau
        self.concept_process = concept_process
        self.img_dims = img_dims
        self.mask = mask.view(img_dims).bool().to(device)
        self.shift_mask = shift_mask.view(img_dims).bool().to(device) if shift_mask is not None else self.mask

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
                 encoder_hidden_states: torch.FloatTensor = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                 image_rotary_emb: Optional[torch.Tensor] = None,
                 timestep=None) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]

        concept_process = self.concept_process
        t_flag = timestep > self.tau

        # Extract concept features from reference
        if encoder_hidden_states is not None:
            concept_feature = hidden_states[0, self.mask, :]
        else:
            concept_feature = hidden_states[0, 512:, :][self.mask, :]

        # Token replacement on target
        r_hidden_states = hidden_states
        if t_flag:
            if encoder_hidden_states is not None:
                r_hidden_states[1, self.shift_mask, :] = concept_feature
            else:
                text_hs = hidden_states[1, :512, :]
                img_hs = hidden_states[1, 512:, :]
                img_hs[self.shift_mask, :] = concept_feature
                r_hidden_states[1] = torch.cat([text_hs, img_hs], dim=0)

        # Q/K/V projection
        key = attn.to_k(r_hidden_states if t_flag else hidden_states)
        value = attn.to_v(r_hidden_states if t_flag else hidden_states)
        query = attn.to_q(r_hidden_states if t_flag else hidden_states)

        # Concept KV concatenation (LSI: only active in single-stream blocks)
        if concept_process:
            c_query = attn.to_q(concept_feature).repeat(query.shape[0], 1, 1)
            query = torch.cat([query, c_query], dim=1)
            c_key = attn.to_k(concept_feature).repeat(key.shape[0], 1, 1)
            c_value = attn.to_v(concept_feature).repeat(value.shape[0], 1, 1)
            key = torch.cat([key, c_key], dim=1)
            value = torch.cat([value, c_value], dim=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Joint-stream block: fuse text and image projections
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)

        # RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            if not concept_process:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            else:
                def extend_rotary_emb(emb, mask):
                    enc_emb = emb[:512, :]
                    hid_emb = emb[512:, :]
                    concept_emb = hid_emb[mask, :]
                    return torch.cat([enc_emb, hid_emb, concept_emb], dim=0)
                image_rotary_emb = (
                    extend_rotary_emb(image_rotary_emb[0], self.shift_mask),
                    extend_rotary_emb(image_rotary_emb[1], self.shift_mask),
                )
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            hidden_states = hidden_states[:, :self.img_dims, :]
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = hidden_states[:, :self.img_dims + 512, :]
            return hidden_states


class ARIAMultiAttnProcessor:
    """Multi-subject ARIA attention processor with LSI + GIA + CPD.

    Args:
        name: Layer name (used to determine joint vs single block).
        masks: List of binary masks, one per subject.
        device: Target device.
        tau: Timestep threshold for token replacement.
        concept_process: Whether to apply concept KV concat at this layer.
            Set True for single-stream blocks (LSI), False for joint-stream.
        shift_masks: Optional target masks (defaults to same as masks).
        img_dims: Number of image tokens (H * W in latent space).
        cpd_enabled: Enable Concept Position Decoupling.
        cpd_lambda: Scaling factor for concept keys when CPD is active.
        img_ids: (img_dims, 3) position IDs tensor for CPD.
        pos_embed_fn: FluxPosEmbed module for recomputing RoPE.
        latent_width: Width in latent space for CPD offset computation.
    """

    def __init__(self, name: str, masks: List[torch.Tensor], device: torch.device,
                 tau: float = 0.90, concept_process: bool = False,
                 shift_masks: Optional[List[torch.Tensor]] = None,
                 img_dims: int = 4096,
                 cpd_enabled: bool = False, cpd_lambda: float = 1.10,
                 img_ids: Optional[torch.Tensor] = None,
                 pos_embed_fn=None, latent_width: Optional[int] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")
        self.name = name
        self.device = device
        self.tau = tau
        self.concept_process = concept_process
        self.img_dims = img_dims
        self.masks = [m.view(img_dims).bool().to(device) for m in masks]
        if shift_masks is None:
            self.shift_masks = self.masks
        else:
            self.shift_masks = [m.view(img_dims).bool().to(device) for m in shift_masks]
        self.cpd_enabled = cpd_enabled
        self.cpd_lambda = cpd_lambda
        self.img_ids = img_ids
        self.pos_embed_fn = pos_embed_fn
        self.latent_width = latent_width

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor,
                 encoder_hidden_states: torch.FloatTensor = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                 image_rotary_emb: Optional[torch.Tensor] = None,
                 timestep=None) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
        concept_process = self.concept_process
        t_flag = timestep > self.tau

        # Token replacement
        concept_features = []
        r_hidden_states = hidden_states
        for idx, mask in enumerate(self.masks):
            if encoder_hidden_states is not None:
                concept_feature = hidden_states[idx, mask, :]
            else:
                concept_feature = hidden_states[idx, 512:, :][mask, :]
            shift_mask = self.shift_masks[idx]
            concept_features.append(concept_feature)
            if t_flag:
                if encoder_hidden_states is not None:
                    r_hidden_states[-1, shift_mask, :] = concept_feature
                else:
                    text_hs = r_hidden_states[-1, :512, :]
                    img_hs = r_hidden_states[-1, 512:, :]
                    img_hs[shift_mask, :] = concept_feature
                    r_hidden_states[-1] = torch.cat([text_hs, img_hs], dim=0)

        # Q/K/V
        key = attn.to_k(r_hidden_states if t_flag else hidden_states)
        value = attn.to_v(r_hidden_states if t_flag else hidden_states)
        query = attn.to_q(r_hidden_states if t_flag else hidden_states)

        # Concept KV concat (LSI: only in single-stream blocks)
        concept_kv_ranges = []
        if concept_process:
            offset = key.shape[1]
            for concept_feature in concept_features:
                n_tokens = concept_feature.shape[0]
                c_query = attn.to_q(concept_feature).repeat(query.shape[0], 1, 1)
                query = torch.cat([query, c_query], dim=1)
                c_key = attn.to_k(concept_feature).repeat(key.shape[0], 1, 1)
                c_value = attn.to_v(concept_feature).repeat(value.shape[0], 1, 1)
                if self.cpd_enabled:
                    c_key = c_key * self.cpd_lambda
                concept_kv_ranges.append((offset, offset + n_tokens))
                key = torch.cat([key, c_key], dim=1)
                value = torch.cat([value, c_value], dim=1)
                offset += n_tokens

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = attn.add_k_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)
            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)

        # RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            if not concept_process:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            else:
                if self.cpd_enabled and self.pos_embed_fn is not None and self.img_ids is not None:
                    image_rotary_emb = self._get_cpd_rotary_emb(image_rotary_emb)
                else:
                    def extend_rotary_emb(emb, shift_masks):
                        enc_emb = emb[:512, :]
                        hid_emb = emb[512:, :]
                        concept_embs = [hid_emb[m, :] for m in shift_masks]
                        concept_emb = torch.cat(concept_embs, dim=0) if concept_embs else torch.zeros(0, hid_emb.shape[1], device=hid_emb.device)
                        return torch.cat([enc_emb, hid_emb, concept_emb], dim=0)
                    image_rotary_emb = (
                        extend_rotary_emb(image_rotary_emb[0], self.shift_masks),
                        extend_rotary_emb(image_rotary_emb[1], self.shift_masks),
                    )
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # GIA or standard attention
        if concept_process and encoder_hidden_states is None and len(concept_kv_ranges) > 0:
            hidden_states = self._gia_attention(query, key, value, attn, concept_kv_ranges, batch_size, head_dim)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            hidden_states = hidden_states[:, :self.img_dims, :]
            return hidden_states, encoder_hidden_states
        else:
            hidden_states = hidden_states[:, :self.img_dims + 512, :]
            return hidden_states

    def _get_cpd_rotary_emb(self, image_rotary_emb):
        """Compute CPD-shifted RoPE: each subject's concept tokens get unique spatial positions."""
        ori_cos, ori_sin = image_rotary_emb
        enc_cos, hid_cos = ori_cos[:512, :], ori_cos[512:, :]
        enc_sin, hid_sin = ori_sin[:512, :], ori_sin[512:, :]
        txt_ids = torch.zeros(512, 3, device=self.device, dtype=self.img_ids.dtype)

        concept_cos_parts, concept_sin_parts = [], []
        for i, m in enumerate(self.shift_masks):
            shifted_ids = self.img_ids[m, :].clone()
            shifted_ids[:, 2] += self.latent_width * (i + 1)
            full_ids = torch.cat([txt_ids, shifted_ids], dim=0)
            shifted_cos, shifted_sin = self.pos_embed_fn(full_ids)
            concept_cos_parts.append(shifted_cos[512:, :])
            concept_sin_parts.append(shifted_sin[512:, :])

        if concept_cos_parts:
            concept_cos = torch.cat(concept_cos_parts, dim=0)
            concept_sin = torch.cat(concept_sin_parts, dim=0)
        else:
            concept_cos = torch.zeros(0, hid_cos.shape[1], device=self.device)
            concept_sin = torch.zeros(0, hid_sin.shape[1], device=self.device)

        return (
            torch.cat([enc_cos, hid_cos, concept_cos], dim=0),
            torch.cat([enc_sin, hid_sin, concept_sin], dim=0),
        )

    def _gia_attention(self, query, key, value, attn, concept_kv_ranges, batch_size, head_dim):
        """Group Isolation Attention: each subject's queries attend only to its own concept KV."""
        base_len = 512 + self.img_dims

        # Reference batches: standard attention
        if batch_size > 1:
            ref_out = F.scaled_dot_product_attention(
                query[:-1], key[:-1], value[:-1], dropout_p=0.0, is_causal=False)
        else:
            ref_out = None

        tgt_q = query[-1:]
        tgt_k = key[-1:]
        tgt_v = value[-1:]
        base_k = tgt_k[:, :, :base_len, :]
        base_v = tgt_v[:, :, :base_len, :]
        total_seq = tgt_q.shape[2]
        tgt_out = torch.zeros(1, attn.heads, total_seq, head_dim,
                              dtype=query.dtype, device=query.device)

        # Text tokens: attend to all KV
        text_q = tgt_q[:, :, :512, :]
        tgt_out[:, :, :512, :] = F.scaled_dot_product_attention(
            text_q, tgt_k, tgt_v, dropout_p=0.0, is_causal=False)

        # Per-subject isolation
        all_masks_combined = torch.zeros(self.img_dims, dtype=torch.bool, device=self.device)
        for m in self.masks:
            all_masks_combined |= m

        for subj_idx, (mask, (ckv_start, ckv_end)) in enumerate(zip(self.masks, concept_kv_ranges)):
            other_masks = torch.zeros_like(mask)
            for j, m in enumerate(self.masks):
                if j != subj_idx:
                    other_masks |= m

            # Exclusive region: attend to base + own concept KV only
            exclusive = mask & ~other_masks
            if exclusive.any():
                q_positions = 512 + torch.where(exclusive)[0]
                subj_q = tgt_q[:, :, q_positions, :]
                subj_k = torch.cat([base_k, tgt_k[:, :, ckv_start:ckv_end, :]], dim=2)
                subj_v = torch.cat([base_v, tgt_v[:, :, ckv_start:ckv_end, :]], dim=2)
                tgt_out[:, :, q_positions, :] = F.scaled_dot_product_attention(
                    subj_q, subj_k, subj_v, dropout_p=0.0, is_causal=False)

            # Overlap region: attend to all KV (fallback)
            overlap = mask & other_masks
            if overlap.any():
                overlap_positions = 512 + torch.where(overlap)[0]
                overlap_q = tgt_q[:, :, overlap_positions, :]
                tgt_out[:, :, overlap_positions, :] = F.scaled_dot_product_attention(
                    overlap_q, tgt_k, tgt_v, dropout_p=0.0, is_causal=False)

        # Background: attend to base KV only (no concept features)
        bg_mask = ~all_masks_combined
        if bg_mask.any():
            bg_positions = 512 + torch.where(bg_mask)[0]
            bg_q = tgt_q[:, :, bg_positions, :]
            tgt_out[:, :, bg_positions, :] = F.scaled_dot_product_attention(
                bg_q, base_k, base_v, dropout_p=0.0, is_causal=False)

        # Concept query positions
        if total_seq > base_len:
            concept_q = tgt_q[:, :, base_len:, :]
            tgt_out[:, :, base_len:, :] = F.scaled_dot_product_attention(
                concept_q, tgt_k, tgt_v, dropout_p=0.0, is_causal=False)

        if ref_out is not None:
            return torch.cat([ref_out, tgt_out], dim=0)
        return tgt_out


# =========================================================================
# Setup helpers
# =========================================================================

def setup_aria_single(pipe, mask, img_dims, tau=0.90):
    """Set up ARIA for single-subject personalization (LSI only)."""
    from attn_processor import set_flux_transformer_attn_processor

    def set_proc(name, hidden_size, cross_attention_dim, ori_attn_proc):
        is_single = name.startswith("single_transformer_blocks.")
        return ARIAAttnProcessor(
            name=name, mask=mask.clone(), device=torch.device("cuda"),
            tau=tau, img_dims=img_dims, concept_process=is_single)

    set_flux_transformer_attn_processor(pipe.transformer, set_proc)


def setup_aria_multi(pipe, masks, img_dims, tau=0.90,
                     cpd_enabled=False, cpd_lambda=1.10,
                     img_ids=None, pos_embed_fn=None, latent_width=None):
    """Set up ARIA for multi-subject personalization (LSI + GIA + optional CPD)."""
    from attn_processor import set_flux_transformer_attn_processor

    def set_proc(name, hidden_size, cross_attention_dim, ori_attn_proc):
        is_single = name.startswith("single_transformer_blocks.")
        return ARIAMultiAttnProcessor(
            name=name, masks=[m.clone() for m in masks], device=torch.device("cuda"),
            tau=tau, img_dims=img_dims, concept_process=is_single,
            cpd_enabled=cpd_enabled, cpd_lambda=cpd_lambda,
            img_ids=img_ids, pos_embed_fn=pos_embed_fn, latent_width=latent_width)

    set_flux_transformer_attn_processor(pipe.transformer, set_proc)
