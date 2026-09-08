# SPDX-License-Identifier: Apache-2.0
"""EXL3 quantization support for vLLM.

EXL3 (ExLlamaV3) uses trellis-coded quantization with Hadamard transforms.
Each quantized linear layer has:
  - trellis: int16 3D tensor (tiles_k, tiles_n, 16*bits) -- packed weights
  - suh: fp16 1D tensor (in_features,) -- input Hadamard sign-flip scales
  - svh: fp16 1D tensor (out_features,) -- output Hadamard sign-flip scales

The forward pass per projection is: had(x, suh) -> exl3_gemm -> had(out, svh)

IMPORTANT: Each projection has its own independent suh. Merged layers (QKV,
gate_up) CANNOT share a single suh — each sub-projection must be run
separately with its own input Hadamard transform.
"""

from typing import TYPE_CHECKING, Any, Union
import os

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# Env var to enable fused MoE GEMM+Had (disabled by default — regresses
# on RDNA3 due to VGPR spilling from 8-accumulator Had epilogue).
# Superseded by V7 compound op (EXL3_MOE_COMPOUND). Kept for testing.
_USE_FUSED_MOE_HAD = os.environ.get("EXL3_MOE_FUSED_HAD", "0") == "1"

# Env var to enable FP16 expert weight cache: dequant MoE experts at load
# time → use FP16 GEMM kernel (no trellis decode per K-tile).
# Cost: ~288MB/GPU extra. Default ON for EP configurations.
_USE_FP16_EXPERTS = os.environ.get("EXL3_FP16_EXPERTS", "0") == "1"

# Env var to enable fused gate+up compound op (1 graph node instead of 7).
# Default OFF — superseded by V2 compound op. Kept for legacy fallback path.
_USE_FUSED_GATE_UP = os.environ.get("EXL3_FUSED_GATE_UP", "0") == "1"

# V7: Compound Had→GEMM→Had op (1 graph node per projection, 3 GPU kernels).
# Back-to-back kernel launching — 4 graph nodes per MoE layer instead of 10.
# Default OFF — superseded by V2 compound op below.
_USE_MOE_COMPOUND = os.environ.get("EXL3_MOE_COMPOUND", "1") == "1"

# V2: Zero-allocation compound Had→GEMM→Had. Same 4 graph nodes as V7, but
# all scratch buffers allocated outside the opaque op (visible to torch.compile
# memory planner → CUDA graph memory pool, not real cudaMalloc on replay).
# Default ON when HIP batched Had is available.
# Default OFF: V2 compound ops still regress 4.9 tok/s (26.9 vs 31.8).
# The compound op wrapper adds more overhead than it saves in graph nodes.
# CUDA graphs already capture the same kernel launches regardless of
# whether they're grouped in 3 compound ops or 11 separate ops.
_USE_MOE_COMPOUND_V2 = os.environ.get("EXL3_MOE_COMPOUND_V2", "0") == "1"

# V10.5: Batched multi-GEMM for merged layers (QKV, gate_up).
# Groups matching-N sub-projections into one exl3_multi_gemm launch.
# Default ON. Set EXL3_MULTI_GEMM=0 to use sequential per-projection path.
_USE_MULTI_GEMM = os.environ.get("EXL3_MULTI_GEMM", "1") == "1"

# Prefill BLOCK_M=64: kernel-level dequant reuse for prefill.
# Automatically activated by the HIP kernel when M is large enough.
# No separate env var needed — controlled by EXL3_PREFILL_M64 in __init__.py.


# ---------------------------------------------------------------------------
# Custom weight loader
# ---------------------------------------------------------------------------

def _make_exl3_weight_loader(linear_weight_loader, output_sizes):
    """Create a weight loader that handles EXL3 custom parameters.

    For EXL3TrellisParameter and EXL3ScaleParameter, we handle the
    merged/QKV weight loading ourselves. For other params (e.g., bias),
    we fall back to the linear layer's weight_loader.
    """

    def exl3_weight_loader(param, loaded_weight, loaded_shard_id=None):
        if isinstance(param, EXL3TrellisParameter):
            _load_trellis(param, loaded_weight, loaded_shard_id,
                          output_sizes)
        elif isinstance(param, EXL3SuhParameter):
            _load_suh(param, loaded_weight, loaded_shard_id, output_sizes)
        elif isinstance(param, EXL3ScaleParameter):
            _load_svh(param, loaded_weight, loaded_shard_id, output_sizes)
        elif getattr(param, '_exl3_cb_dummy', False):
            # Codebook marker tensor (mcg/mul1): silently absorb, no-op
            pass
        else:
            if loaded_shard_id is not None:
                linear_weight_loader(param, loaded_weight, loaded_shard_id)
            else:
                linear_weight_loader(param, loaded_weight)

    return exl3_weight_loader


def _shard_idx(shard_id):
    """Convert shard_id to integer index.

    Returns int for single shard, or (start, stop) tuple for fused
    multi-shard weights (e.g. GDN in_proj_qkv -> (0, 3)).
    """
    if shard_id is None:
        return None
    if isinstance(shard_id, str):
        return {"q": 0, "k": 1, "v": 2}[shard_id]
    if isinstance(shard_id, tuple):
        return (shard_id[0], shard_id[-1] + 1)
    return shard_id


def _record_sub_bits(param, proj_idx: int, bits: int):
    """Record the ACTUAL bit width of a projection as observed in the
    checkpoint (trellis inner-dim // 16). For fused GDN projections the
    config-derived `exl3_per_shard_bits` has checkpoint-shard granularity
    (e.g. [in_proj_qkv, in_proj_z] = 2 entries) while the layer has
    per-projection granularity (q,k,v,z = 4) — the recorded values let
    `process_weights_after_loading` build the correct per-projection list."""
    d = getattr(param, "_exl3_sub_bits", None)
    if d is None:
        d = {}
        param._exl3_sub_bits = d
    d[proj_idx] = bits


def _load_trellis(param: "EXL3TrellisParameter", loaded_weight: torch.Tensor,
                  shard_id, output_sizes: list[int]):
    """Load trellis weight, handling merged column and QKV cases."""
    # Dequant path: param holds full-size weights, no TP slicing at load time.
    # `update_param_tp_status` overrides param.tp_size back to the layer TP
    # size after create_weights returns, so we use the sticky marker set by
    # `_create_weights_dequant`.
    if getattr(param, "_exl3_dequant_load", False):
        tp_size = 1
        tp_rank = 0
    else:
        tp_size = param.tp_size
        tp_rank = param.tp_rank

    if shard_id is None:
        # Non-merged: detect sharding dimension from shapes.
        # ColumnParallel (e.g. lm_head): dim 1 is TP-sharded (output tiles)
        # RowParallel (e.g. o_proj, down_proj): dim 0 is TP-sharded (input tiles)
        if tp_size > 1:
            if loaded_weight.shape[0] != param.data.shape[0]:
                shard_dim = 0  # RowParallel: input-sharded
            elif loaded_weight.shape[1] != param.data.shape[1]:
                shard_dim = 1  # ColumnParallel: output-sharded
            else:
                shard_dim = -1  # shapes already match
            if shard_dim >= 0:
                tp_tiles = param.data.shape[shard_dim]
                start = tp_rank * tp_tiles
                # Handle padded embeddings: param may be larger than
                # the per-TP share of loaded_weight (vocab padding).
                avail = loaded_weight.shape[shard_dim] - start
                if avail <= 0:
                    return  # entire share is padding (zeros)
                load_size = min(tp_tiles, avail)
                src = loaded_weight.narrow(shard_dim, start, load_size)
                if load_size < tp_tiles:
                    # Partial: copy into beginning, rest stays zeros
                    param.data.narrow(shard_dim, 0, load_size).copy_(src)
                    return
                loaded_weight = src
        assert param.data.shape == loaded_weight.shape, (
            f"Trellis shape mismatch (shard_id=None): "
            f"param={param.data.shape}, loaded={loaded_weight.shape}, "
            f"tp_size={tp_size}, tp_rank={tp_rank}"
        )
        param.data.copy_(loaded_weight)
        return

    idx = _shard_idx(shard_id)
    # Fused multi-shard: idx is (start, stop) tuple. Each sub-shard must
    # be TP-sliced independently along its own checkpoint offsets,
    # otherwise rank 0 gets full q+k and rank 1 full v on TP=2.
    if isinstance(idx, tuple):
        # output_sizes can be passed as either FULL sizes (pre-TP) or
        # per-TP-partitioned sizes depending on call site.  Detect by
        # comparing the fused sub-range sum against the checkpoint width.
        fused_sum = sum(output_sizes[idx[0]:idx[1]])
        ckpt_sub_width = loaded_weight.shape[1]
        # tiles-space equivalent: loaded_weight is stored as tiles (/16)
        if ckpt_sub_width * 16 == fused_sum:
            # output_sizes are FULL (checkpoint width matches sum)
            per_tp_sizes = [s // tp_size for s in output_sizes]
        elif ckpt_sub_width * 16 == fused_sum * tp_size:
            # output_sizes are per-TP (checkpoint = sum * tp_size)
            per_tp_sizes = list(output_sizes)
        else:
            raise RuntimeError(
                f"Cannot infer output_sizes layout: ckpt_width(tiles*16)="
                f"{ckpt_sub_width*16}, sum(output_sizes[{idx[0]}:{idx[1]}])="
                f"{fused_sum}, tp_size={tp_size}, output_sizes={output_sizes}"
            )
        param_off_tiles = sum(per_tp_sizes[:idx[0]]) // 16
        ckpt_off_tiles = 0
        for sub_idx in range(idx[0], idx[1]):
            sub_per_tp_tiles = per_tp_sizes[sub_idx] // 16
            sub_full_tiles = sub_per_tp_tiles * tp_size
            src_start = ckpt_off_tiles + tp_rank * sub_per_tp_tiles
            src = loaded_weight.narrow(1, src_start, sub_per_tp_tiles)
            dst = param.data.narrow(1, param_off_tiles, sub_per_tp_tiles)
            # Variable per-shard bits (same handling as the single-shard
            # path below): the fused param is allocated with max-bits, so a
            # sub-shard stored at fewer bits has a smaller inner dim.
            # Pad-load into the leading region.
            src_inner = src.shape[-1]
            dst_inner = dst.shape[-1]
            _record_sub_bits(param, sub_idx, src_inner // 16)
            if src_inner != dst_inner:
                assert src_inner < dst_inner, (
                    f"Trellis sub-shard inner-dim larger than allocated "
                    f"sub={sub_idx}: param={dst.shape}, ckpt_slice={src.shape}, "
                    f"loaded={loaded_weight.shape}, tp={tp_size}")
                assert dst.shape[:-1] == src.shape[:-1], (
                    f"Trellis sub-shard mismatch sub={sub_idx}: "
                    f"param={dst.shape}, ckpt_slice={src.shape}, "
                    f"per_tp_sizes={per_tp_sizes}, output_sizes={output_sizes}, "
                    f"loaded={loaded_weight.shape}, tp={tp_size}")
                dst[..., :src_inner].copy_(src)
            else:
                assert dst.shape == src.shape, (
                    f"Trellis sub-shard mismatch sub={sub_idx}: "
                    f"param={dst.shape}, ckpt_slice={src.shape}, "
                    f"per_tp_sizes={per_tp_sizes}, output_sizes={output_sizes}, "
                    f"loaded={loaded_weight.shape}, tp={tp_size}")
                dst.copy_(src)
            param_off_tiles += sub_per_tp_tiles
            ckpt_off_tiles += sub_full_tiles
        return

    shard_offset = sum(output_sizes[:idx])
    shard_size = output_sizes[idx]

    tile_offset = shard_offset // 16
    tile_size = shard_size // 16

    # output_sizes are already TP-divided (output_partition_sizes), so
    # tile_size and tile_offset are per-TP-rank values.  Index into the
    # full checkpoint tensor using the number of distinct slices it
    # contains (handles GQA where KV heads are replicated across ranks).
    n_distinct = loaded_weight.shape[1] // tile_size
    shard_rank = tp_rank * n_distinct // tp_size
    loaded_weight = loaded_weight.narrow(1, shard_rank * tile_size,
                                         tile_size)
    param_data = param.data.narrow(1, tile_offset, tile_size)
    _record_sub_bits(param, idx, loaded_weight.shape[-1] // 16)
    # Variable per-shard bits: loaded inner-dim may be SMALLER than param
    # inner-dim (allocated with max-bits). Pad-load into the leading region.
    loaded_inner = loaded_weight.shape[-1]
    param_inner = param_data.shape[-1]
    if loaded_inner != param_inner:
        assert loaded_inner < param_inner, (
            f"Trellis inner-dim larger than allocated: "
            f"param_slice={param_data.shape}, loaded={loaded_weight.shape}, "
            f"shard_id={shard_id}"
        )
        assert param_data.shape[:-1] == loaded_weight.shape[:-1], (
            f"Trellis non-inner shape mismatch: param_slice={param_data.shape}, "
            f"loaded={loaded_weight.shape}, shard_id={shard_id}"
        )
        param_data[..., :loaded_inner].copy_(loaded_weight)
        return
    assert param_data.shape == loaded_weight.shape, (
        f"Trellis shape mismatch: param_slice={param_data.shape}, "
        f"loaded={loaded_weight.shape}, shard_id={shard_id}"
    )
    param_data.copy_(loaded_weight)


def _load_suh(param: "EXL3SuhParameter", loaded_weight: torch.Tensor,
              shard_id, output_sizes: list[int]):
    """Load per-projection suh into the stacked suh tensor.

    For non-merged layers: suh is 1D (in_features,), num_projections=1.
    For merged layers: suh is 2D (num_projections, in_features), each
    projection gets its own row.
    """
    if shard_id is None:
        # Non-merged: RowParallel layers have TP-sharded input, so suh
        # (input scale) must be narrowed to the local partition.
        if getattr(param, "_exl3_dequant_load", False):
            tp_size = 1
            tp_rank = 0
        else:
            tp_size = param.tp_size
            tp_rank = param.tp_rank
        if tp_size > 1 and loaded_weight.shape[0] != param.data.shape[-1]:
            per_tp = param.data.shape[-1]
            loaded_weight = loaded_weight.narrow(
                0, tp_rank * per_tp, per_tp)
        assert param.data.shape[-1] == loaded_weight.shape[0], (
            f"suh shape mismatch: param={param.data.shape}, "
            f"loaded={loaded_weight.shape}"
        )
        if param.data.dim() == 1:
            param.data.copy_(loaded_weight)
        else:
            param.data[0].copy_(loaded_weight)
        return

    idx = _shard_idx(shard_id)
    assert param.data.dim() == 2, (
        f"Merged suh must be 2D, got {param.data.shape}"
    )
    # Fused multi-shard: idx is (start, stop)
    if isinstance(idx, tuple):
        n_shards = idx[1] - idx[0]
        if loaded_weight.dim() == 1:
            # Shared suh across fused projections — broadcast to all rows
            assert param.data.shape[1] == loaded_weight.shape[0], (
                f"suh dim mismatch: param={param.data.shape[1]}, "
                f"loaded={loaded_weight.shape[0]}")
            for i in range(idx[0], idx[1]):
                param.data[i].copy_(loaded_weight)
        else:
            assert param.data.shape[1] == loaded_weight.shape[1], (
                f"suh dim mismatch: param={param.data.shape[1]}, "
                f"loaded={loaded_weight.shape[1]}")
            param.data[idx[0]:idx[1]].copy_(loaded_weight)
    else:
        assert param.data.shape[1] == loaded_weight.shape[0], (
            f"suh dim mismatch: param row={param.data.shape[1]}, "
            f"loaded={loaded_weight.shape[0]}"
        )
        param.data[idx].copy_(loaded_weight)


def _load_svh(param: "EXL3ScaleParameter", loaded_weight: torch.Tensor,
              shard_id, output_sizes: list[int]):
    """Load svh (output scale), handling merged column and QKV cases."""
    if getattr(param, "_exl3_dequant_load", False):
        tp_size = 1
        tp_rank = 0
    else:
        tp_size = param.tp_size
        tp_rank = param.tp_rank

    if shard_id is None:
        # ColumnParallel: svh (output scale) is TP-sharded, narrow it.
        # RowParallel: svh is full output size, no narrowing needed.
        if tp_size > 1 and loaded_weight.shape[0] != param.data.shape[0]:
            per_tp = param.data.shape[0]
            start = tp_rank * per_tp
            # Handle padded embeddings (vocab padding)
            avail = loaded_weight.shape[0] - start
            if avail <= 0:
                return
            load_size = min(per_tp, avail)
            loaded_weight = loaded_weight.narrow(0, start, load_size)
            if load_size < per_tp:
                param.data[:load_size].copy_(loaded_weight)
                return
        assert param.data.shape == loaded_weight.shape, (
            f"svh shape mismatch (shard_id=None): "
            f"param={tuple(param.data.shape)}, "
            f"loaded={tuple(loaded_weight.shape)}, "
            f"tp_size={tp_size}, tp_rank={tp_rank}, "
            f"output_sizes={output_sizes}, "
            f"prefix={getattr(param, 'prefix', '?')}"
        )
        param.data.copy_(loaded_weight)
        return

    idx = _shard_idx(shard_id)
    # Fused multi-shard: idx is (start, stop) tuple — TP-slice each sub
    # independently along its own checkpoint offsets.
    if isinstance(idx, tuple):
        # Detect FULL vs per-TP output_sizes (same logic as _load_trellis).
        fused_sum = sum(output_sizes[idx[0]:idx[1]])
        ckpt_width = loaded_weight.shape[0]
        if ckpt_width == fused_sum:
            per_tp_sizes = [s // tp_size for s in output_sizes]
        elif ckpt_width == fused_sum * tp_size:
            per_tp_sizes = list(output_sizes)
        else:
            raise RuntimeError(
                f"svh: cannot infer output_sizes layout: ckpt_width="
                f"{ckpt_width}, fused_sum={fused_sum}, tp={tp_size}, "
                f"output_sizes={output_sizes}"
            )
        param_off = sum(per_tp_sizes[:idx[0]])
        ckpt_off = 0
        for sub_idx in range(idx[0], idx[1]):
            sub_per_tp = per_tp_sizes[sub_idx]
            sub_full = sub_per_tp * tp_size
            src_start = ckpt_off + tp_rank * sub_per_tp
            src = loaded_weight.narrow(0, src_start, sub_per_tp)
            dst = param.data.narrow(0, param_off, sub_per_tp)
            assert dst.shape == src.shape, (
                f"svh sub-shard mismatch sub={sub_idx}: "
                f"param={dst.shape}, ckpt_slice={src.shape}, "
                f"per_tp_sizes={per_tp_sizes}, output_sizes={output_sizes}")
            dst.copy_(src)
            param_off += sub_per_tp
            ckpt_off += sub_full
        return

    shard_offset = sum(output_sizes[:idx])
    shard_size = output_sizes[idx]

    # output_sizes are already TP-divided (output_partition_sizes), so
    # shard_size and shard_offset are per-TP-rank values.  Index into
    # the full checkpoint tensor using the number of distinct slices
    # (handles GQA where KV heads are replicated across ranks).
    n_distinct = loaded_weight.shape[0] // shard_size
    shard_rank = tp_rank * n_distinct // tp_size
    loaded_weight = loaded_weight.narrow(0, shard_rank * shard_size,
                                         shard_size)
    param_data = param.data.narrow(0, shard_offset, shard_size)
    assert param_data.shape == loaded_weight.shape, (
        f"svh shape mismatch: param_slice={param_data.shape}, "
        f"loaded={loaded_weight.shape}"
    )
    param_data.copy_(loaded_weight)


# ---------------------------------------------------------------------------
# Custom parameter classes
# ---------------------------------------------------------------------------


def _dbg(msg):
    import os, time
    with open("/tmp/vllm_debug.log", "a") as f:
        f.write(f"{time.time():.3f} pid={os.getpid()} {msg}\n")
        f.flush()


class EXL3TrellisParameter(BasevLLMParameter):
    """3D trellis parameter (tiles_k, tiles_n, words_per_tile)."""

    def __init__(self, data: torch.Tensor, weight_loader, **kwargs):
        super().__init__(data=data, weight_loader=weight_loader)


class EXL3SuhParameter(BasevLLMParameter):
    """Per-projection suh (input Hadamard scale).

    For non-merged layers: 1D (in_features,)
    For merged layers: 2D (num_projections, in_features)
    """

    def __init__(self, data: torch.Tensor, weight_loader, **kwargs):
        super().__init__(data=data, weight_loader=weight_loader)


class EXL3ScaleParameter(BasevLLMParameter):
    """1D svh output scale parameter."""

    def __init__(self, data: torch.Tensor, weight_loader, **kwargs):
        super().__init__(data=data, weight_loader=weight_loader)


# ---------------------------------------------------------------------------
# EXL3 quantization config
# ---------------------------------------------------------------------------

class EXL3Config(QuantizationConfig):
    """Config class for EXL3 (ExLlamaV3) trellis-coded quantization."""

    # Codebook name → cb integer mapping
    _CODEBOOK_MAP = {"3inst": 0, "mcg": 1, "mul1": 2}

    def __init__(
        self,
        weight_bits: int,
        head_bits: int,
        tensor_storage: dict[str, Any],
        cb: int = 0,
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.head_bits = head_bits
        self.tensor_storage = tensor_storage
        self.cb = cb

        # Build lookup: prefix -> bits_per_weight (only for EXL3 layers)
        self._layer_bits: dict[str, int] = {}
        self._unquantized_layers: list[str] = []

        for prefix, info in tensor_storage.items():
            if info.get("quant_format") == "exl3":
                self._layer_bits[prefix] = info["bits_per_weight"]
            else:
                self._unquantized_layers.append(prefix)

        # Build normalized lookup for VL and other multi-nested models.
        # tensor_storage keys may use "model.language_model.X" but vLLM
        # prefixes use "language_model.model.X" due to hf_to_vllm_mapper.
        self._layer_bits_normalized: dict[str, int] = {}
        for prefix, bits in self._layer_bits.items():
            norm = self._normalize_prefix(prefix)
            self._layer_bits_normalized[norm] = bits

    def __repr__(self) -> str:
        return (
            f"EXL3Config(weight_bits={self.weight_bits}, "
            f"head_bits={self.head_bits}, cb={self.cb})"
        )

    def get_name(self) -> "QuantizationMethods":
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EXL3Config":
        weight_bits = int(config.get("bits", 4))
        head_bits = int(config.get("head_bits", 6))
        tensor_storage = config.get("tensor_storage", {})
        codebook_name = config.get("codebook", "3inst")
        cb = cls._CODEBOOK_MAP.get(codebook_name, 0)
        if codebook_name not in cls._CODEBOOK_MAP:
            logger.warning(
                "EXL3: unknown codebook '%s', defaulting to cb=0 (3inst)",
                codebook_name)
        else:
            logger.info("EXL3: codebook=%s (cb=%d)", codebook_name, cb)
        return cls(weight_bits, head_bits, tensor_storage, cb=cb)

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        """Strip model nesting prefixes to get canonical layer path.

        VL models have checkpoint paths like 'model.language_model.layers.0...'
        but vLLM module paths like 'language_model.model.layers.0...'.
        Normalize both to 'layers.0...' for matching.
        """
        for p in ("model.language_model.", "language_model.model.",
                  "language_model.", "model."):
            if prefix.startswith(p):
                return prefix[len(p):]
        return prefix

    def _get_moe_layer_bits(self, prefix: str) -> int | None:
        """Look up bits_per_weight for a MoE expert layer.

        FusedMoE prefix looks like "model.layers.0.mlp.experts" but
        tensor_storage keys are per-projection like
        "layers.0.mlp.experts.5.gate_proj". Search for any key that
        starts with this prefix (with "model." stripped).
        """
        if not self._layer_bits:
            return self.weight_bits

        stripped = prefix.removeprefix("model.")
        norm = self._normalize_prefix(prefix)
        for key, bits in self._layer_bits.items():
            key_norm = self._normalize_prefix(key)
            if (key.startswith(stripped + ".")
                    or key.startswith(prefix + ".")
                    or key_norm.startswith(norm + ".")):
                return bits
        return None

    def _get_layer_bits(self, prefix: str) -> int | None:
        """Look up bits_per_weight for a layer prefix."""
        if self._layer_bits:
            # Try exact match first
            if prefix in self._layer_bits:
                return self._layer_bits[prefix]

            # Try without "model." prefix (tensor_storage keys may omit it)
            stripped = prefix.removeprefix("model.")
            if stripped in self._layer_bits:
                return self._layer_bits[stripped]

            # Try packed module mapping (qkv_proj -> q_proj, etc.)
            proj_name = prefix.split(".")[-1]
            if proj_name in self.packed_modules_mapping:
                first_shard = self.packed_modules_mapping[proj_name][0]
                unfused_prefix = prefix.replace(proj_name, first_shard)
                if unfused_prefix in self._layer_bits:
                    return self._layer_bits[unfused_prefix]
                unfused_stripped = unfused_prefix.removeprefix("model.")
                if unfused_stripped in self._layer_bits:
                    return self._layer_bits[unfused_stripped]

            # Try normalized prefix matching for VL and multi-nested models.
            # E.g. tensor_storage has "model.language_model.layers.0.mlp.q_proj"
            # but vLLM prefix is "language_model.model.layers.0.mlp.qkv_proj".
            norm = self._normalize_prefix(prefix)
            if norm in self._layer_bits_normalized:
                return self._layer_bits_normalized[norm]
            # Also try packed module on normalized prefix
            if proj_name in self.packed_modules_mapping:
                first_shard = self.packed_modules_mapping[proj_name][0]
                norm_unfused = norm.rsplit(".", 1)[0] + "." + first_shard
                if norm_unfused in self._layer_bits_normalized:
                    return self._layer_bits_normalized[norm_unfused]

            return None
        else:
            # Fallback when tensor_storage not available
            if "lm_head" in prefix:
                return self.head_bits
            return self.weight_bits

    def _lookup_bits_for_unfused(self, unfused_prefix: str) -> int | None:
        """Lookup bits for a single, unfused projection prefix.

        Used by `_get_per_shard_bits_for_merged` to resolve per-shard bits
        without re-triggering packed-module fallback.
        """
        if unfused_prefix in self._layer_bits:
            return self._layer_bits[unfused_prefix]
        stripped = unfused_prefix.removeprefix("model.")
        if stripped in self._layer_bits:
            return self._layer_bits[stripped]
        norm = self._normalize_prefix(unfused_prefix)
        if norm in self._layer_bits_normalized:
            return self._layer_bits_normalized[norm]
        return None

    def _get_per_shard_bits_for_merged(self, prefix: str) -> list[int] | None:
        """For merged projections (qkv_proj, gate_up_proj), return list of
        per-shard bits in shard order. Returns None when not merged, when
        any shard can't be resolved, or when all shards have identical bits
        (in which case the regular `_get_layer_bits` path is sufficient).

        Handles the variable-bits-per-projection case in EXL3 quants where
        the quantizer chose different bit rates for Q/K/V (or gate/up).
        """
        if not self._layer_bits:
            return None
        proj_name = prefix.split(".")[-1]
        if proj_name not in self.packed_modules_mapping:
            return None
        shard_names = self.packed_modules_mapping[proj_name]
        if len(shard_names) <= 1:
            return None
        bits_list = []
        for shard in shard_names:
            unfused = prefix.replace(proj_name, shard)
            b = self._lookup_bits_for_unfused(unfused)
            if b is None:
                return None
            bits_list.append(b)
        if len(set(bits_list)) == 1:
            return None  # uniform — caller can use _get_layer_bits as-is
        return bits_list

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        if isinstance(layer, RoutedExperts):
            bits = self._get_moe_layer_bits(prefix)
            if bits is not None:
                return EXL3FusedMoEMethod(self, bits, layer.moe_config)
            return None
        if isinstance(layer, LinearBase):
            bits = self._get_layer_bits(prefix)
            if bits is not None:
                # Detect variable per-shard bits for merged QKV/gate_up
                per_shard_bits = self._get_per_shard_bits_for_merged(prefix)
                return EXL3LinearMethod(self, bits, per_shard_bits)
            return UnquantizedLinearMethod()
        if isinstance(layer, ParallelLMHead):
            # Return EXL3EmbeddingMethod for all ParallelLMHead instances.
            # For tied models (tie_word_embeddings=True), the weight loader
            # skips lm_head.* keys, so trellis stays zeros. After tying,
            # lm_head.weight points to embed_tokens. We detect this in
            # process_weights_after_loading and fall back to FP16 matmul.
            bits = self._get_layer_bits(prefix)
            if bits is not None:
                return EXL3EmbeddingMethod(self, bits)
        return None


# ---------------------------------------------------------------------------
# EXL3 linear method
# ---------------------------------------------------------------------------

class EXL3LinearMethod(LinearMethodBase):
    """Linear method for EXL3 trellis-coded quantization.

    Each projection has its own suh (input Hadamard scale). For merged
    layers (QKV, gate_up), we store per-projection suh and run each
    sub-projection separately in apply().

    When the quantizer chose different bit rates per shard of a merged
    projection, `per_shard_bits` is a list of bits (one per shard); the
    trellis is allocated with max-bits inner-dim and per-shard regions
    are pad-loaded (smaller shards leave trailing zeros in the inner-dim).
    The apply()/dequant paths pass the per-shard bits to the EXL3 kernel
    so each shard reads only its own data.
    """

    def __init__(self, quant_config: EXL3Config, bits: int,
                 per_shard_bits: list[int] | None = None):
        self.quant_config = quant_config
        self.bits = bits
        self.per_shard_bits = per_shard_bits
        if per_shard_bits and len(set(per_shard_bits)) > 1:
            self.max_bits = max(per_shard_bits)
        else:
            self.max_bits = bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        num_projections = len(output_partition_sizes)
        original_weight_loader = extra_weight_attrs.get("weight_loader")

        # Check if TP-sharded dimensions are Had-128 compatible.
        # Had-128 requires all dimensions to be multiples of 128.
        # When TP sharding makes dimensions too small (e.g. shared expert
        # with intermediate_size=512 and TP=8 → 64), we must dequantize
        # to FP16 at load time.
        had_incompatible = (
            input_size_per_partition % 128 != 0
            or any(s % 128 != 0 for s in output_partition_sizes)
        )

        # Performance-based dequant: for shapes where Triton kernel launch
        # overhead (~70µs) dominates, rocBLAS FP16 via F.linear is faster.
        # Threshold: K*N < 8M elements (~lm_head is 78M, stays EXL3).
        _ROCBLAS_DEQUANT_THRESHOLD = int(os.environ.get(
            "EXL3_DEQUANT_THRESHOLD", "8000000"))
        if (not had_incompatible
                and _ROCBLAS_DEQUANT_THRESHOLD > 0
                and input_size_per_partition * output_size_per_partition
                    < _ROCBLAS_DEQUANT_THRESHOLD):
            # Skip for GQA layers where K/V heads are replicated across
            # TP ranks — _create_weights_dequant computes full sizes as
            # per_rank * TP, which overcounts replicated KV shards
            # (checkpoint K SVH is 512 but full_sizes says 1024).
            has_kv_replication = (
                getattr(layer, 'num_kv_head_replicas', 1) > 1
            )
            if not has_kv_replication:
                logger.info(
                    "EXL3: dequant %s (%d×%d = %dK < %dM threshold)"
                    " to FP16 for rocBLAS",
                    layer.__class__.__name__,
                    input_size_per_partition, output_size_per_partition,
                    input_size_per_partition * output_size_per_partition
                    // 1024,
                    _ROCBLAS_DEQUANT_THRESHOLD // 1_000_000,
                )
                had_incompatible = True

        if had_incompatible:
            self._create_weights_dequant(
                layer, input_size_per_partition, output_partition_sizes,
                input_size, output_size, params_dtype,
                original_weight_loader, num_projections,
            )
            return

        exl3_loader = _make_exl3_weight_loader(
            original_weight_loader, output_partition_sizes
        )

        # Trellis: 3D (tiles_k, tiles_n_total, words_per_tile)
        # For variable per-shard bits, allocate using MAX bits inner-dim
        # so smaller-bits shards can be pad-loaded (leading region filled,
        # trailing zeros). The EXL3 kernel reads bits*16 words per tile,
        # so per-shard bits determines what's actually consumed.
        tiles_k = input_size_per_partition // 16
        tiles_n = output_size_per_partition // 16
        words_per_tile = 16 * self.max_bits

        trellis = EXL3TrellisParameter(
            data=torch.zeros(
                (tiles_k, tiles_n, words_per_tile),
                dtype=torch.int16,
            ),
            weight_loader=exl3_loader,
        )

        # suh: per-projection input Hadamard scale
        # For merged layers (num_projections > 1): 2D (num_projections, K)
        # For non-merged layers: 1D (K,)
        if num_projections > 1:
            suh = EXL3SuhParameter(
                data=torch.ones(
                    (num_projections, input_size_per_partition),
                    dtype=torch.float16,
                ),
                weight_loader=exl3_loader,
            )
        else:
            suh = EXL3SuhParameter(
                data=torch.ones(input_size_per_partition,
                                dtype=torch.float16),
                weight_loader=exl3_loader,
            )

        # svh: output Hadamard scale (concatenated for merged layers)
        svh = EXL3ScaleParameter(
            data=torch.ones(output_size_per_partition, dtype=torch.float16),
            weight_loader=exl3_loader,
        )

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)
        # Stamp the owning module prefix for diagnostics (misroute tracing)
        _dbg_prefix = getattr(layer, "prefix", "?")
        svh.prefix = _dbg_prefix
        suh.prefix = _dbg_prefix
        trellis.prefix = _dbg_prefix

        # Register codebook marker dummy param (mcg/mul1) so vLLM's
        # weight loader doesn't crash on the checkpoint tensor.
        self._register_cb_dummy(layer, exl3_loader)

        layer.exl3_bits = self.bits
        layer.exl3_max_bits = self.max_bits
        layer.exl3_per_shard_bits = self.per_shard_bits
        layer.exl3_cb = self.quant_config.cb
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer._exl3_dequant = False

    def _register_cb_dummy(self, layer, weight_loader):
        """Register a codebook marker dummy parameter if cb != 0.

        MCG models (cb=1) have a per-layer 'mcg' scalar int32 tensor
        in the checkpoint. MUL1 models (cb=2) have 'mul1'. We register
        a dummy parameter so the weight loader can absorb it silently.
        """
        cb = self.quant_config.cb
        cb_names = {1: "mcg", 2: "mul1"}
        if cb in cb_names:
            name = cb_names[cb]
            dummy = BasevLLMParameter(
                data=torch.zeros(1, dtype=torch.int32),
                weight_loader=weight_loader,
            )
            dummy._exl3_cb_dummy = True
            layer.register_parameter(name, dummy)

    def _create_weights_dequant(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        original_weight_loader,
        num_projections: int,
    ):
        """Create full-size EXL3 params for layers that need dequantization.

        When TP sharding makes dimensions incompatible with Had-128 (e.g.
        shared expert with intermediate=512 and TP=8 → 64 per rank),
        we allocate trellis/suh/svh at FULL size, load without TP sharding,
        then dequantize to FP16 and TP-shard in process_weights_after_loading.
        """
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
        )
        tp_size = get_tensor_model_parallel_world_size()

        # Detect sharding type:
        # ColumnParallel: output is TP-sharded, input is full
        # RowParallel: input is TP-sharded, output is full
        # Replicated (e.g. ReplicatedLinear like the QSA index_qk_proj):
        #   neither is sharded — output_partition_sizes are ALREADY full.
        # `is_row_parallel` alone mis-classifies Replicated as ColumnParallel
        # (input not sharded) and scales output by TP, over-allocating svh
        # (e.g. index_qk_proj: 640 * TP4 = 2560 vs checkpoint 640).
        # Use `output_size` (the full logical output vLLM passes) as ground
        # truth: when per-rank output already equals it, the output is not
        # TP-sharded (RowParallel or Replicated) and must not be scaled.
        is_row_parallel = (input_size_per_partition != input_size)
        per_rank_output = sum(output_partition_sizes)
        output_already_full = is_row_parallel or per_rank_output == output_size

        if output_already_full:
            full_output_partition_sizes = list(output_partition_sizes)
        else:
            # ColumnParallel: output_partition_sizes are per-TP-rank
            full_output_partition_sizes = [
                s * tp_size for s in output_partition_sizes
            ]
        full_output_size = sum(full_output_partition_sizes)

        # Use full-size loader (no TP sharding for EXL3 params)
        exl3_loader = _make_exl3_weight_loader(
            original_weight_loader, full_output_partition_sizes
        )

        # Full-size trellis — use max_bits for variable per-shard inner-dim
        tiles_k = input_size // 16
        tiles_n = full_output_size // 16
        words_per_tile = 16 * self.max_bits

        trellis = EXL3TrellisParameter(
            data=torch.zeros(
                (tiles_k, tiles_n, words_per_tile),
                dtype=torch.int16,
            ),
            weight_loader=exl3_loader,
        )
        # Override TP attributes so weight loader doesn't TP-shard.
        # NOTE: ColumnParallelLinear.update_param_tp_status() resets these
        # to the layer TP size after create_weights returns, so we also set
        # `_exl3_dequant_load = True` as a sticky marker the loader checks.
        trellis.tp_size = 1
        trellis.tp_rank = 0
        trellis._exl3_dequant_load = True

        # Full-size suh
        if num_projections > 1:
            suh = EXL3SuhParameter(
                data=torch.ones(
                    (num_projections, input_size),
                    dtype=torch.float16,
                ),
                weight_loader=exl3_loader,
            )
        else:
            suh = EXL3SuhParameter(
                data=torch.ones(input_size, dtype=torch.float16),
                weight_loader=exl3_loader,
            )
        suh.tp_size = 1
        suh.tp_rank = 0
        suh._exl3_dequant_load = True

        # Full-size svh
        svh = EXL3ScaleParameter(
            data=torch.ones(full_output_size, dtype=torch.float16),
            weight_loader=exl3_loader,
        )
        svh.tp_size = 1
        svh.tp_rank = 0
        svh._exl3_dequant_load = True

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)

        # Register codebook marker dummy param (mcg/mul1)
        self._register_cb_dummy(layer, exl3_loader)

        layer.exl3_bits = self.bits
        layer.exl3_max_bits = self.max_bits
        layer.exl3_per_shard_bits = self.per_shard_bits
        layer.exl3_cb = self.quant_config.cb
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer._exl3_dequant = True
        layer._exl3_dequant_full_output_sizes = full_output_partition_sizes
        layer._exl3_dequant_input_size = input_size
        layer._exl3_dequant_tp_size = tp_size
        layer._exl3_dequant_is_row_parallel = is_row_parallel
        # Replicated (neither dim sharded): post-load must keep the full
        # output on every rank instead of N-sharding it (see _dequant_layer).
        layer._exl3_dequant_output_already_full = output_already_full
        layer._exl3_dequant_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Canonicalize per-shard bits to PER-PROJECTION granularity before
        # anything reads them. The config-derived list follows
        # packed_modules_mapping (checkpoint shards), which for fused GDN
        # projections (in_proj_qkvz -> [in_proj_qkv, in_proj_z]) is shorter
        # than the layer's projection count (q,k,v,z) and would index out of
        # range. The loader recorded the actual bits per projection from the
        # checkpoint tensor shapes — authoritative, so prefer them.
        suh_data = layer.suh.data if hasattr(layer, 'suh') else None
        sub_bits = getattr(layer.trellis, "_exl3_sub_bits", None) \
            if hasattr(layer, 'trellis') else None
        if (sub_bits and suh_data is not None and suh_data.dim() == 2):
            num_proj = suh_data.shape[0]
            if set(sub_bits.keys()) == set(range(num_proj)):
                bits_list = [sub_bits[i] for i in range(num_proj)]
                if (len(set(bits_list)) > 1
                        or getattr(layer, 'exl3_per_shard_bits', None)):
                    layer.exl3_per_shard_bits = bits_list

        if getattr(layer, '_exl3_dequant', False):
            _dbg(f"EXL3.pwl: dequant path")
            self._process_weights_dequant(layer)
            _dbg(f"EXL3.pwl: dequant DONE")
            return

        _dbg(f"EXL3.pwl: start normal path, bits={layer.exl3_bits}")
        trellis_data = layer.trellis.data
        layer.trellis = torch.nn.Parameter(trellis_data, requires_grad=False)
        layer.suh = torch.nn.Parameter(layer.suh.data, requires_grad=False)
        layer.svh = torch.nn.Parameter(layer.svh.data, requires_grad=False)

        # Precompute int32 view for the Triton kernel
        layer.trellis_i32 = layer.trellis.data.view(torch.int32)
        _dbg(f"EXL3.pwl: trellis_i32 done")

        # Eagerly populate device-side caches so that no CPU→CUDA copies
        # happen during CUDA graph capture.
        device = trellis_data.device
        from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
            get_bit_tables,
        )
        from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
            _get_had128,
        )
        _dbg(f"EXL3.pwl: before get_bit_tables")
        get_bit_tables(layer.exl3_bits, device)
        _dbg(f"EXL3.pwl: before _get_had128")
        _get_had128(device)
        _dbg(f"EXL3.pwl: caches done")

        # Pre-compute batched weight groups for merged layers (QKV, gate_up).
        # Groups sub-projections by matching (N, bits) so they can be dispatched
        # as a single multi-GEMM launch when their bits match; variable per-shard
        # bits projections are forced to singletons with own bits+inner-dim slice.
        suh = layer.suh.data
        if suh.dim() == 2:
            output_partition_sizes = layer.exl3_output_partition_sizes
            trellis_i32 = layer.trellis_i32
            num_proj = suh.shape[0]
            per_shard_bits = getattr(layer, 'exl3_per_shard_bits', None)
            default_bits = layer.exl3_bits

            # Build per-projection metadata with per-shard bits and inner-dim
            proj_infos = []
            tile_offset = 0
            out_offset = 0
            for i in range(num_proj):
                proj_size = output_partition_sizes[i]
                proj_tiles = proj_size // 16
                shard_bits = per_shard_bits[i] if per_shard_bits else default_bits
                proj_infos.append({
                    'index': i,
                    'proj_size': proj_size,
                    'proj_tiles': proj_tiles,
                    'tile_offset': tile_offset,
                    'out_offset': out_offset,
                    'bits': shard_bits,
                    'words_i16': 16 * shard_bits,
                    'words_i32': 8 * shard_bits,
                })
                tile_offset += proj_tiles
                out_offset += proj_size

            # Group by matching (N, bits) — shards with different bits never batch
            from collections import defaultdict
            groups_by_key = defaultdict(list)
            for pi in proj_infos:
                groups_by_key[(pi['proj_tiles'], pi['bits'])].append(pi)

            batched_groups = []
            singleton_groups = []
            for (proj_tiles, group_bits), members in groups_by_key.items():
                if len(members) >= 2:
                    # Stack trellis slices (sliced to inner-dim of group_bits)
                    slices = []
                    indices = []
                    out_offsets = []
                    for m in members:
                        s = trellis_i32[:, m['tile_offset']:
                                        m['tile_offset'] + m['proj_tiles'],
                                        :m['words_i32']]
                        slices.append(s.contiguous())
                        indices.append(m['index'])
                        out_offsets.append(m['out_offset'])
                    B_stacked = torch.stack(slices, dim=0)
                    batched_groups.append({
                        'indices': indices,
                        'B_stacked_i32': B_stacked,
                        'n_batch': len(members),
                        'proj_N': members[0]['proj_size'],
                        'out_offsets': out_offsets,
                        'bits': group_bits,
                    })
                else:
                    m = members[0]
                    s = trellis_i32[:, m['tile_offset']:
                                    m['tile_offset'] + m['proj_tiles'],
                                    :m['words_i32']]
                    singleton_groups.append({
                        'index': m['index'],
                        'trellis_i32': s.contiguous(),
                        'trellis': layer.trellis.data[
                            :, m['tile_offset']:
                            m['tile_offset'] + m['proj_tiles'],
                            :m['words_i16']],
                        'out_offset': m['out_offset'],
                        'proj_N': m['proj_size'],
                        'bits': m['bits'],
                    })

            layer.exl3_batched_groups = batched_groups
            layer.exl3_singleton_groups = singleton_groups

    def _process_weights_dequant(self, layer: torch.nn.Module) -> None:
        """Dequantize EXL3 weights to FP16 for Had-128-incompatible layers.

        Runs the full had→gemm→had pipeline on an identity matrix to extract
        the FP16 weight matrix, then TP-shards the result.
        """
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            had_r_128,
        )

        trellis = layer.trellis.data
        suh = layer.suh.data
        svh = layer.svh.data
        bits = layer.exl3_bits
        per_shard_bits = getattr(layer, 'exl3_per_shard_bits', None)
        cb = getattr(layer, 'exl3_cb', 0)
        full_output_sizes = layer._exl3_dequant_full_output_sizes
        tp_size = layer._exl3_dequant_tp_size
        tp_rank = get_tensor_model_parallel_rank()
        output_partition_sizes = layer.exl3_output_partition_sizes
        is_row_parallel = layer._exl3_dequant_is_row_parallel
        K = layer._exl3_dequant_input_size

        device = trellis.device
        num_proj = suh.shape[0] if suh.dim() == 2 else 1

        # Dequant each projection by passing identity batches through the
        # had→gemm→had pipeline (processes BLOCK_SIZE rows at a time to
        # limit temporary memory).
        BLOCK = 128
        fp16_columns = []
        tile_offset = 0
        out_offset = 0
        for i in range(num_proj):
            proj_suh = suh[i] if suh.dim() == 2 else suh
            proj_n = full_output_sizes[i]
            proj_tiles = proj_n // 16
            # Variable per-shard bits: use this shard's bits for the kernel.
            # The merged trellis allocates max-bits inner-dim; the kernel
            # only reads `shard_bits * 16` words per tile (rest is zeros).
            shard_bits = per_shard_bits[i] if per_shard_bits else bits
            shard_words = 16 * shard_bits
            proj_trellis = trellis.narrow(
                1, tile_offset, proj_tiles).narrow(2, 0, shard_words)
            proj_trellis_i32 = proj_trellis.contiguous().view(torch.int32)
            proj_svh = svh[out_offset:out_offset + proj_n]

            # Process in blocks of BLOCK rows of the identity matrix
            weight_rows = []
            for start in range(0, K, BLOCK):
                end = min(start + BLOCK, K)
                bs = end - start
                eye_block = torch.zeros(
                    (bs, K), dtype=torch.float16, device=device
                )
                eye_block[:, start:end] = torch.eye(
                    bs, dtype=torch.float16, device=device
                )

                # Input Hadamard
                xh = torch.empty_like(eye_block)
                had_r_128(eye_block, xh, proj_suh, None, 1.0)

                # GEMM
                proj_out = exl3_gemm(
                    xh, proj_trellis, bits=shard_bits, cb=cb,
                    B_i32=proj_trellis_i32,
                )

                # Output Hadamard
                proj_out_h = torch.empty_like(proj_out)
                had_r_128(proj_out, proj_out_h, None, proj_svh, 1.0)

                weight_rows.append(proj_out_h)

            # Full dequanted weight: (K, N_full_proj)
            full_weight = torch.cat(weight_rows, dim=0)

            if is_row_parallel:
                # RowParallel: TP-shard the input (K) dimension
                k_per_tp = K // tp_size
                tp_start = tp_rank * k_per_tp
                sharded_weight = full_weight[tp_start:tp_start + k_per_tp, :]
            elif getattr(layer, '_exl3_dequant_output_already_full', False):
                # Replicated (e.g. QSA index_qk_proj): output is full on
                # every rank. N-sharding here handed ranks > 0 an empty
                # slice past the weight (tp_rank * N), producing width-0
                # outputs downstream.
                sharded_weight = full_weight
            else:
                # ColumnParallel: TP-shard the output (N) dimension
                n_per_tp = output_partition_sizes[i]
                tp_start = tp_rank * n_per_tp
                sharded_weight = full_weight[:, tp_start:tp_start + n_per_tp]
            fp16_columns.append(sharded_weight)

            tile_offset += proj_tiles
            out_offset += proj_n

        # Assemble the TP-sharded weight
        # F.linear expects (out_features, in_features), so transpose
        fp16_weight = torch.cat(fp16_columns, dim=1).T.contiguous()

        # Replace EXL3 params with dequanted weight
        # Cast to params_dtype for torch.compile compatibility (Inductor
        # expects weight dtype to match activation dtype).
        target_dtype = getattr(layer, '_exl3_dequant_params_dtype',
                               torch.float16)
        if fp16_weight.dtype != target_dtype:
            fp16_weight = fp16_weight.to(target_dtype)
        del layer.trellis, layer.suh, layer.svh
        layer.weight = torch.nn.Parameter(fp16_weight, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Dequanted layers use standard matmul (cast x to match weight dtype
        # to avoid mixed-dtype errors in torch.compile/Inductor)
        if getattr(layer, '_exl3_dequant', False):
            return torch.nn.functional.linear(
                x.to(layer.weight.dtype), layer.weight, bias
            )

        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            exl3_multi_gemm,
            had_r_128,
        )

        trellis = layer.trellis
        suh = layer.suh
        svh = layer.svh
        bits = layer.exl3_bits
        cb = getattr(layer, 'exl3_cb', 0)
        output_partition_sizes = layer.exl3_output_partition_sizes

        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x_2d = x.reshape(-1, K).half()
        M = x_2d.shape[0]

        if suh.dim() == 1:
            # Non-merged layer: single projection
            xh = torch.empty_like(x_2d)
            had_r_128(x_2d, xh, suh, None, 1.0)

            out = exl3_gemm(
                xh, trellis, bits=bits, cb=cb,
                B_i32=layer.trellis_i32,
            )

            out_h = torch.empty_like(out)
            had_r_128(out, out_h, None, svh, 1.0)
        else:
            # Merged layer
            num_proj = suh.shape[0]
            N_total = sum(output_partition_sizes)
            out_h = torch.empty((M, N_total), dtype=x_2d.dtype,
                                device=x_2d.device)

            per_shard_bits = getattr(layer, 'exl3_per_shard_bits', None)
            if (_USE_MULTI_GEMM
                    and hasattr(layer, 'exl3_batched_groups')):
                # V10.5: group-by-(N,bits) batched dispatch
                # Singleton groups (unique key): sequential Had→GEMM→Had
                for sg in layer.exl3_singleton_groups:
                    i = sg['index']
                    sg_bits = sg.get('bits', bits)
                    xh = torch.empty_like(x_2d)
                    had_r_128(x_2d, xh, suh[i], None, 1.0)

                    proj_out = exl3_gemm(
                        xh, sg['trellis'], bits=sg_bits, cb=cb,
                        B_i32=sg['trellis_i32'],
                    )

                    proj_svh = svh.narrow(
                        0, sg['out_offset'], sg['proj_N'])
                    proj_out_h = out_h.narrow(
                        1, sg['out_offset'], sg['proj_N'])
                    had_r_128(proj_out, proj_out_h, None, proj_svh, 1.0)

                # Batched groups (matching N+bits): multi-GEMM
                for bg in layer.exl3_batched_groups:
                    n_batch = bg['n_batch']
                    bg_bits = bg.get('bits', bits)

                    # Input Hadamard per sub-projection
                    xh_batched = torch.empty(
                        (n_batch, M, K), dtype=x_2d.dtype,
                        device=x_2d.device)
                    for j, i in enumerate(bg['indices']):
                        had_r_128(
                            x_2d, xh_batched[j], suh[i], None, 1.0)

                    # Single batched GEMM launch
                    out_batched = exl3_multi_gemm(
                        xh_batched, bg['B_stacked_i32'],
                        n_batch, bits=bg_bits, cb=cb,
                    )

                    # Output Hadamard per sub-projection
                    for j, i in enumerate(bg['indices']):
                        o_off = bg['out_offsets'][j]
                        proj_svh = svh.narrow(0, o_off, bg['proj_N'])
                        proj_out_h = out_h.narrow(
                            1, o_off, bg['proj_N'])
                        had_r_128(
                            out_batched[j], proj_out_h,
                            None, proj_svh, 1.0)
            else:
                # Sequential fallback (EXL3_MULTI_GEMM=0)
                tile_offset = 0
                out_offset = 0
                for i in range(num_proj):
                    proj_suh = suh[i]
                    proj_size = output_partition_sizes[i]
                    proj_tiles = proj_size // 16
                    shard_bits = per_shard_bits[i] if per_shard_bits else bits
                    shard_words = 16 * shard_bits

                    xh = torch.empty_like(x_2d)
                    had_r_128(x_2d, xh, proj_suh, None, 1.0)

                    proj_trellis = trellis.narrow(
                        1, tile_offset, proj_tiles).narrow(
                        2, 0, shard_words)
                    proj_trellis_i32 = proj_trellis.contiguous().view(
                        torch.int32)

                    proj_out = exl3_gemm(
                        xh, proj_trellis, bits=shard_bits, cb=cb,
                        B_i32=proj_trellis_i32,
                    )

                    proj_svh = svh.narrow(0, out_offset, proj_size)
                    proj_out_h = out_h.narrow(1, out_offset, proj_size)
                    had_r_128(
                        proj_out, proj_out_h, None, proj_svh, 1.0)

                    tile_offset += proj_tiles
                    out_offset += proj_size

        if bias is not None:
            out_h = out_h + bias

        return out_h.reshape(*orig_shape, -1)


# ---------------------------------------------------------------------------
# EXL3 embedding method (for quantized lm_head)
# ---------------------------------------------------------------------------

class EXL3EmbeddingMethod(QuantizeMethodBase):
    """Quantize method for EXL3 quantized lm_head (ParallelLMHead).

    Uses the same had -> gemm -> had forward pass as EXL3LinearMethod.
    The lm_head is never merged, so suh is always 1D.
    """

    def __init__(self, quant_config: EXL3Config, bits: int):
        self.quant_config = quant_config
        self.bits = bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        original_weight_loader = extra_weight_attrs.get("weight_loader")

        # Check if TP-sharded output is Had-128 compatible.
        # lm_head is ColumnParallel: output is TP-sharded.
        had_incompatible = (
            input_size_per_partition % 128 != 0
            or output_size_per_partition % 128 != 0
        )

        if had_incompatible:
            self._create_weights_dequant(
                layer, input_size_per_partition, output_partition_sizes,
                input_size, output_size, params_dtype,
                original_weight_loader,
            )
            return

        exl3_loader = _make_exl3_weight_loader(
            original_weight_loader, output_partition_sizes
        )

        tiles_k = input_size_per_partition // 16
        tiles_n = output_size_per_partition // 16
        words_per_tile = 16 * self.bits

        trellis = EXL3TrellisParameter(
            data=torch.zeros(
                (tiles_k, tiles_n, words_per_tile),
                dtype=torch.int16,
            ),
            weight_loader=exl3_loader,
        )

        suh = EXL3SuhParameter(
            data=torch.ones(input_size_per_partition, dtype=torch.float16),
            weight_loader=exl3_loader,
        )

        svh = EXL3ScaleParameter(
            data=torch.ones(output_size_per_partition, dtype=torch.float16),
            weight_loader=exl3_loader,
        )

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)

        # Register codebook marker dummy param (mcg/mul1)
        self._register_cb_dummy(layer, exl3_loader)

        layer.exl3_bits = self.bits
        layer.exl3_cb = self.quant_config.cb
        layer._exl3_dequant = False

    def _register_cb_dummy(self, layer, weight_loader):
        """Register a codebook marker dummy parameter if cb != 0."""
        cb = self.quant_config.cb
        cb_names = {1: "mcg", 2: "mul1"}
        if cb in cb_names:
            name = cb_names[cb]
            dummy = BasevLLMParameter(
                data=torch.zeros(1, dtype=torch.int32),
                weight_loader=weight_loader,
            )
            dummy._exl3_cb_dummy = True
            layer.register_parameter(name, dummy)

    def _create_weights_dequant(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        original_weight_loader,
    ):
        """Create full-size EXL3 params for Had-128-incompatible lm_head."""
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
        )
        tp_size = get_tensor_model_parallel_world_size()

        # lm_head is ColumnParallel: output is TP-sharded
        full_output_partition_sizes = [
            s * tp_size for s in output_partition_sizes
        ]
        full_output_size = sum(full_output_partition_sizes)

        exl3_loader = _make_exl3_weight_loader(
            original_weight_loader, full_output_partition_sizes
        )

        tiles_k = input_size // 16
        tiles_n = full_output_size // 16
        words_per_tile = 16 * self.bits

        trellis = EXL3TrellisParameter(
            data=torch.zeros(
                (tiles_k, tiles_n, words_per_tile),
                dtype=torch.int16,
            ),
            weight_loader=exl3_loader,
        )
        trellis.tp_size = 1
        trellis.tp_rank = 0

        suh = EXL3SuhParameter(
            data=torch.ones(input_size, dtype=torch.float16),
            weight_loader=exl3_loader,
        )
        suh.tp_size = 1
        suh.tp_rank = 0

        svh = EXL3ScaleParameter(
            data=torch.ones(full_output_size, dtype=torch.float16),
            weight_loader=exl3_loader,
        )
        svh.tp_size = 1
        svh.tp_rank = 0

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)

        # Register codebook marker dummy param (mcg/mul1)
        self._register_cb_dummy(layer, exl3_loader)

        layer.exl3_bits = self.bits
        layer.exl3_cb = self.quant_config.cb
        layer._exl3_dequant = True
        layer._exl3_dequant_full_output_sizes = full_output_partition_sizes
        layer._exl3_dequant_input_size = input_size
        layer._exl3_dequant_tp_size = tp_size
        layer._exl3_dequant_params_dtype = params_dtype
        layer.exl3_output_partition_sizes = output_partition_sizes

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, '_exl3_dequant', False):
            self._process_weights_dequant(layer)
            return

        # Detect if this is a tied lm_head (tie_word_embeddings=True).
        # In that case, trellis was never loaded (stays zeros) and
        # layer.weight was set to embed_tokens.weight by tie_weights().
        is_tied = (layer.trellis.data.count_nonzero() == 0
                   and hasattr(layer, 'weight'))
        layer._exl3_is_tied = is_tied

        if is_tied:
            # Remove EXL3 parameters to save memory; we'll use weight
            del layer.trellis
            del layer.suh
            del layer.svh
        else:
            layer.trellis = torch.nn.Parameter(
                layer.trellis.data, requires_grad=False)
            layer.suh = torch.nn.Parameter(
                layer.suh.data, requires_grad=False)
            layer.svh = torch.nn.Parameter(
                layer.svh.data, requires_grad=False)
            layer.trellis_i32 = layer.trellis.data.view(torch.int32)

            device = layer.trellis.data.device
            from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
                get_bit_tables,
            )
            from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
                _get_had128,
            )
            get_bit_tables(layer.exl3_bits, device)
            _get_had128(device)

    def _process_weights_dequant(self, layer: torch.nn.Module) -> None:
        """Dequantize lm_head to FP16 when output is Had-128-incompatible."""
        from vllm.distributed import get_tensor_model_parallel_rank
        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            had_r_128,
        )

        trellis = layer.trellis.data
        suh = layer.suh.data
        svh = layer.svh.data
        bits = layer.exl3_bits
        cb = getattr(layer, 'exl3_cb', 0)
        full_output_sizes = layer._exl3_dequant_full_output_sizes
        tp_size = layer._exl3_dequant_tp_size
        tp_rank = get_tensor_model_parallel_rank()
        output_partition_sizes = layer.exl3_output_partition_sizes
        K = layer._exl3_dequant_input_size

        device = trellis.device
        BLOCK = 128
        trellis_i32 = trellis.contiguous().view(torch.int32)

        # Process in blocks of BLOCK rows of the identity matrix
        weight_rows = []
        for start in range(0, K, BLOCK):
            end = min(start + BLOCK, K)
            bs = end - start
            eye_block = torch.zeros(
                (bs, K), dtype=torch.float16, device=device
            )
            eye_block[:, start:end] = torch.eye(
                bs, dtype=torch.float16, device=device
            )

            xh = torch.empty_like(eye_block)
            had_r_128(eye_block, xh, suh, None, 1.0)

            proj_out = exl3_gemm(
                xh, trellis, bits=bits, cb=cb, B_i32=trellis_i32,
            )

            proj_out_h = torch.empty_like(proj_out)
            had_r_128(proj_out, proj_out_h, None, svh, 1.0)

            weight_rows.append(proj_out_h)

        # Full weight: (K, N_full)
        full_weight = torch.cat(weight_rows, dim=0)

        # TP-shard the output (ColumnParallel)
        n_per_tp = output_partition_sizes[0]
        tp_start = tp_rank * n_per_tp
        avail = full_weight.shape[1] - tp_start
        n_copy = min(n_per_tp, avail)
        if n_copy < n_per_tp:
            # Last TP rank may have fewer columns (vocab padding)
            sharded = torch.zeros(
                (K, n_per_tp), dtype=torch.float16, device=device
            )
            sharded[:, :n_copy] = full_weight[:, tp_start:tp_start + n_copy]
        else:
            sharded = full_weight[:, tp_start:tp_start + n_per_tp]

        # F.linear expects (out_features, in_features)
        fp16_weight = sharded.T.contiguous()

        target_dtype = getattr(layer, '_exl3_dequant_params_dtype',
                               torch.float16)
        if fp16_weight.dtype != target_dtype:
            fp16_weight = fp16_weight.to(target_dtype)

        del layer.trellis, layer.suh, layer.svh
        layer.weight = torch.nn.Parameter(fp16_weight, requires_grad=False)
        layer._exl3_is_tied = False  # not tied, but dequanted

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, '_exl3_dequant', False):
            return torch.nn.functional.linear(
                x.to(layer.weight.dtype), layer.weight, bias
            )

        if layer._exl3_is_tied:
            # Tied lm_head: use FP16 matmul with embed_tokens weight
            out = torch.nn.functional.linear(x, layer.weight, bias)
            return out

        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            had_r_128,
        )

        orig_shape = x.shape[:-1]
        K = x.shape[-1]
        x_2d = x.reshape(-1, K).half()

        xh = torch.empty_like(x_2d)
        had_r_128(x_2d, xh, layer.suh, None, 1.0)

        out = exl3_gemm(
            xh, layer.trellis, bits=layer.exl3_bits,
            cb=getattr(layer, 'exl3_cb', 0),
            B_i32=layer.trellis_i32,
        )

        out_h = torch.empty_like(out)
        had_r_128(out, out_h, None, layer.svh, 1.0)

        if bias is not None:
            out_h = out_h + bias

        return out_h.reshape(*orig_shape, -1)


# ---------------------------------------------------------------------------
# EXL3 FusedMoE method (for MoE expert layers)
# ---------------------------------------------------------------------------

def _make_exl3_moe_weight_loader(
    param_name: str,
    hidden_size: int,
    intermediate_size_per_partition: int,
    tp_size: int,
    tp_rank: int,
    moe_layer: torch.nn.Module,
):
    """Create a weight loader closure for EXL3 MoE parameters.

    Called as: weight_loader(param, loaded_weight, name,
                             shard_id=..., expert_id=...)

    shard_id: "w1" (gate_proj), "w2" (down_proj), "w3" (up_proj)
    expert_id: integer expert index (global, mapped to local for EP)
    """

    def _loader(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        **kwargs,
    ):
        # Map global expert_id to local for expert parallelism
        expert_map = getattr(moe_layer, "_expert_map", None)
        if expert_map is not None:
            local_id = expert_map[expert_id].item()
            if local_id == -1:
                return False  # Expert not local to this rank
            expert_id = local_id

        # Determine TP slice for each weight type.
        # w1 (gate), w3 (up): K=hidden_size, N=intermediate_size
        #   → TP shards N (output dim) → trellis narrow dim=1, svh narrow dim=0
        #   → suh is full hidden_size (no TP shard)
        # w2 (down): K=intermediate_size, N=hidden_size
        #   → TP shards K (input dim) → trellis narrow dim=0, suh narrow dim=0
        #   → svh is full hidden_size (no TP shard)

        if param_name.endswith("_trellis"):
            _load_moe_trellis(
                param, loaded_weight, expert_id, shard_id,
                intermediate_size_per_partition, tp_size, tp_rank,
            )
        elif param_name.endswith("_suh"):
            _load_moe_suh(
                param, loaded_weight, expert_id, shard_id,
                intermediate_size_per_partition, tp_size, tp_rank,
            )
        elif param_name.endswith("_svh"):
            _load_moe_svh(
                param, loaded_weight, expert_id, shard_id,
                intermediate_size_per_partition, tp_size, tp_rank,
            )
        return True

    return _loader


def _load_moe_trellis(
    param, loaded_weight, expert_id, shard_id,
    intermediate_size_per_partition, tp_size, tp_rank,
):
    """Load trellis into [E, tiles_k, tiles_n, wpt] with TP sharding."""
    # loaded_weight: [tiles_k_full, tiles_n_full, wpt]
    if shard_id in ("w1", "w3"):
        # gate/up: TP shards output (tiles_n dim=1)
        tiles_n_per_tp = intermediate_size_per_partition // 16
        loaded_weight = loaded_weight.narrow(
            1, tp_rank * tiles_n_per_tp, tiles_n_per_tp,
        )
        # w13_trellis: [E, tiles_k, 2*inter_tiles_n, wpt]
        # w1 → first half, w3 → second half
        half = param.data.shape[2] // 2
        if shard_id == "w1":
            param.data[expert_id, :, :half, :].copy_(loaded_weight)
        else:
            param.data[expert_id, :, half:, :].copy_(loaded_weight)
    else:
        # w2 (down): TP shards input (tiles_k dim=0)
        tiles_k_per_tp = intermediate_size_per_partition // 16
        loaded_weight = loaded_weight.narrow(
            0, tp_rank * tiles_k_per_tp, tiles_k_per_tp,
        )
        param.data[expert_id].copy_(loaded_weight)


def _load_moe_suh(
    param, loaded_weight, expert_id, shard_id,
    intermediate_size_per_partition, tp_size, tp_rank,
):
    """Load suh (input Hadamard scale) with TP sharding."""
    # loaded_weight: [K_full] (1D)
    if shard_id in ("w1", "w3"):
        # gate/up: input is hidden_size — NOT TP-sharded, full copy
        # w13_suh: [E, 2, hidden_size]
        idx = 0 if shard_id == "w1" else 1
        param.data[expert_id, idx].copy_(loaded_weight)
    else:
        # w2 (down): input is intermediate_size — TP-sharded
        per_tp = intermediate_size_per_partition
        loaded_weight = loaded_weight.narrow(0, tp_rank * per_tp, per_tp)
        param.data[expert_id].copy_(loaded_weight)


def _load_moe_svh(
    param, loaded_weight, expert_id, shard_id,
    intermediate_size_per_partition, tp_size, tp_rank,
):
    """Load svh (output Hadamard scale) with TP sharding."""
    # loaded_weight: [N_full] (1D)
    if shard_id in ("w1", "w3"):
        # gate/up: output is intermediate_size — TP-sharded
        per_tp = intermediate_size_per_partition
        loaded_weight = loaded_weight.narrow(0, tp_rank * per_tp, per_tp)
        # w13_svh: [E, 2*intermediate_size_per_partition]
        half = param.data.shape[1] // 2
        if shard_id == "w1":
            param.data[expert_id, :half].copy_(loaded_weight)
        else:
            param.data[expert_id, half:].copy_(loaded_weight)
    else:
        # w2 (down): output is hidden_size — NOT TP-sharded
        param.data[expert_id].copy_(loaded_weight)


def _dequant_expert_weights(trellis, trellis_i32, suh, svh, K, N, bits,
                            cb=0):
    """Dequant all experts: trellis → (E, K, N) FP16 via identity-matrix pipeline.

    Args:
        trellis: (E, tiles_k, tiles_n, wpt) int16
        trellis_i32: (E, tiles_k, tiles_n, wpt//2) int32 view
        suh: (E, K) fp16 — input Hadamard scale per expert
        svh: (E, N) fp16 — output Hadamard scale per expert
        K: input dimension
        N: output dimension
        bits: quantization bits
        cb: codebook index (0=3inst, 1=mcg, 2=mul1)

    Returns:
        (E, K, N) fp16 row-major weight tensor
    """
    from vllm.model_executor.layers.quantization.exl3_kernels import (
        exl3_gemm,
        had_r_128,
    )

    E = trellis.shape[0]
    device = trellis.device
    BLOCK = 128

    experts_fp16 = []
    for e in range(E):
        rows = []
        for start in range(0, K, BLOCK):
            bs = min(BLOCK, K - start)
            eye = torch.zeros(bs, K, dtype=torch.float16, device=device)
            eye[:, start:start + bs] = torch.eye(
                bs, dtype=torch.float16, device=device)

            # Input Hadamard
            xh = torch.empty_like(eye)
            had_r_128(eye, xh, suh[e], None, 1.0)

            # GEMM
            proj = exl3_gemm(
                xh, trellis[e], bits=bits, cb=cb,
                B_i32=trellis_i32[e],
            )

            # Output Hadamard
            proj_h = torch.empty_like(proj)
            had_r_128(proj, proj_h, None, svh[e], 1.0)

            rows.append(proj_h)
        experts_fp16.append(torch.cat(rows, dim=0))

    return torch.stack(experts_fp16, dim=0)  # (E, K, N)


class EXL3FusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for EXL3 trellis-coded quantization.

    Each expert has independent trellis, suh, svh per projection.
    gate_proj (w1) and up_proj (w3) are merged into w13 tensors.
    down_proj (w2) is stored separately.

    The forward pass per expert is:
      gate = had(x, suh[0]) -> gemm(w1_trellis) -> had(out, svh[:N])
      up   = had(x, suh[1]) -> gemm(w3_trellis) -> had(out, svh[N:])
      act  = SiLU(gate) * up
      down = had(act, w2_suh) -> gemm(w2_trellis) -> had(out, w2_svh)
    """

    def __init__(
        self,
        quant_config: EXL3Config,
        bits: int,
        moe: "FusedMoEConfig",
    ):
        super().__init__(moe)
        self.quant_config = quant_config
        self.bits = bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # rc4's RoutedExperts has no .tp_size/.tp_rank attrs — TP lives in
        # moe_config.moe_parallel_config (proxied by moe_config.tp_size/tp_rank).
        # getattr(layer, "tp_size", 1) silently returned 1/0 and mis-sized the
        # per-TP expert params.
        moe_cfg = getattr(layer, "moe_config", None)
        if moe_cfg is not None:
            tp_size = moe_cfg.tp_size
            tp_rank = moe_cfg.tp_rank
        else:
            tp_size = getattr(layer, "tp_size", 1)
            tp_rank = getattr(layer, "tp_rank", 0)

        # Validate Had-128 compatibility
        if (intermediate_size_per_partition % 128 != 0
                or hidden_size % 128 != 0):
            raise ValueError(
                f"EXL3 MoE requires dimensions divisible by 128 for "
                f"Hadamard-128 transforms. Got "
                f"intermediate_size_per_partition="
                f"{intermediate_size_per_partition}, "
                f"hidden_size={hidden_size}. "
                f"If using tensor parallelism with small expert sizes, "
                f"enable expert parallelism with "
                f"'--enable-expert-parallel' to avoid TP-sharding "
                f"expert weights."
            )

        bits = self.bits
        wpt = 16 * bits  # words per tile

        hidden_tiles_k = hidden_size // 16
        inter_tiles_n = intermediate_size_per_partition // 16
        inter_tiles_k = intermediate_size_per_partition // 16
        hidden_tiles_n = hidden_size // 16

        # --- w13 (gate + up merged) ---
        w13_trellis = torch.nn.Parameter(
            torch.zeros(
                (num_experts, hidden_tiles_k, 2 * inter_tiles_n, wpt),
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_trellis", w13_trellis)
        set_weight_attrs(w13_trellis, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w13_trellis", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
            # Marker: trellis weights are inherently 3D PER EXPERT; rc4's
            # RoutedExperts.load_weights must NOT treat them as fused-all-
            # experts tensors and unbind their leading (tile) dimension.
            "exl3_per_expert_3d": True,
        })

        w13_suh = torch.nn.Parameter(
            torch.ones(
                (num_experts, 2, hidden_size),
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_suh", w13_suh)
        set_weight_attrs(w13_suh, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w13_suh", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
        })

        w13_svh = torch.nn.Parameter(
            torch.ones(
                (num_experts, 2 * intermediate_size_per_partition),
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_svh", w13_svh)
        set_weight_attrs(w13_svh, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w13_svh", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
        })

        # --- w2 (down) ---
        w2_trellis = torch.nn.Parameter(
            torch.zeros(
                (num_experts, inter_tiles_k, hidden_tiles_n, wpt),
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_trellis", w2_trellis)
        set_weight_attrs(w2_trellis, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w2_trellis", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
            "exl3_per_expert_3d": True,
        })

        w2_suh = torch.nn.Parameter(
            torch.ones(
                (num_experts, intermediate_size_per_partition),
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_suh", w2_suh)
        set_weight_attrs(w2_suh, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w2_suh", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
        })

        w2_svh = torch.nn.Parameter(
            torch.ones(
                (num_experts, hidden_size),
                dtype=torch.float16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_svh", w2_svh)
        set_weight_attrs(w2_svh, {
            "weight_loader": _make_exl3_moe_weight_loader(
                "w2_svh", hidden_size,
                intermediate_size_per_partition, tp_size, tp_rank, layer,
            ),
        })

        layer.exl3_bits = bits
        layer.exl3_cb = self.quant_config.cb

        # Register codebook marker dummy params (mcg/mul1) for MoE.
        # Expert mapping converts "experts.5.gate_proj." → "experts.w13_",
        # so "experts.5.gate_proj.mcg" → "experts.w13_mcg". We need both
        # w13_<cb> and w2_<cb> registered.
        cb = self.quant_config.cb
        cb_names = {1: "mcg", 2: "mul1"}
        if cb in cb_names:
            name = cb_names[cb]
            def _make_cb_dummy_loader():
                def _cb_loader(param, loaded_weight, weight_name,
                               shard_id, expert_id, **kwargs):
                    pass  # Silently absorb
                return _cb_loader
            for prefix in ("w13_", "w2_"):
                dummy = torch.nn.Parameter(
                    torch.zeros(1, dtype=torch.int32),
                    requires_grad=False,
                )
                layer.register_parameter(f"{prefix}{name}", dummy)
                set_weight_attrs(dummy, {
                    "weight_loader": _make_cb_dummy_loader(),
                })

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        bits = layer.exl3_bits

        # Pre-split w13 trellis into gate/up halves for contiguous access
        w13 = layer.w13_trellis.data
        half_n = w13.shape[2] // 2
        layer.w1_trellis = torch.nn.Parameter(
            w13[:, :, :half_n, :].contiguous(), requires_grad=False,
        )
        layer.w3_trellis = torch.nn.Parameter(
            w13[:, :, half_n:, :].contiguous(), requires_grad=False,
        )
        del layer.w13_trellis

        # Int32 views for Triton kernel
        layer.w1_trellis_i32 = layer.w1_trellis.data.view(torch.int32)
        layer.w3_trellis_i32 = layer.w3_trellis.data.view(torch.int32)
        layer.w2_trellis_i32 = layer.w2_trellis.data.view(torch.int32)

        # Pre-split w13 svh into gate/up halves
        w13_svh = layer.w13_svh.data
        half_svh = w13_svh.shape[1] // 2
        layer.w1_svh = torch.nn.Parameter(
            w13_svh[:, :half_svh].contiguous(), requires_grad=False,
        )
        layer.w3_svh = torch.nn.Parameter(
            w13_svh[:, half_svh:].contiguous(), requires_grad=False,
        )
        del layer.w13_svh

        # V5: Pre-split w13_suh gate/up slices to contiguous for dual Had.
        # w13_suh is (E, 2, dim) — [:, 0, :] has stride (2*dim, 1), not
        # contiguous. Pre-materializing avoids .contiguous() per forward.
        layer.w13_suh_gate = torch.nn.Parameter(
            layer.w13_suh.data[:, 0, :].contiguous(), requires_grad=False,
        )
        layer.w13_suh_up = torch.nn.Parameter(
            layer.w13_suh.data[:, 1, :].contiguous(), requires_grad=False,
        )

        # Eagerly populate device-side caches
        device = layer.w1_trellis.data.device
        from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
            _get_had128,
        )
        from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
            get_bit_tables,
        )
        get_bit_tables(bits, device)
        _get_had128(device)

        # Pre-compile Triton kernels for M=1 MoE shapes so the autotuner
        # doesn't run during CUDA graph capture (which would crash HIP).
        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            had_r_128,
        )
        K = layer.w13_suh.shape[-1]     # hidden_size (input to gate/up)
        N = layer.w1_svh.shape[-1]      # intermediate_size (output of gate)
        dummy_x = torch.zeros(1, K, dtype=torch.float16, device=device)
        dummy_h = torch.empty_like(dummy_x)
        dummy_suh = layer.w13_suh[0, 0]  # (K,)
        had_r_128(dummy_x, dummy_h, dummy_suh, None, 1.0)
        # Gate/Up GEMM: (1, K) -> (1, N)
        cb = layer.exl3_cb
        exl3_gemm(dummy_h, layer.w1_trellis[0], bits=bits, cb=cb,
                  B_i32=layer.w1_trellis_i32[0])
        # Down GEMM: (1, N) -> (1, K)
        dummy_n = torch.zeros(1, N, dtype=torch.float16, device=device)
        dummy_nh = torch.empty_like(dummy_n)
        had_r_128(dummy_n, dummy_nh, layer.w2_suh[0], None, 1.0)
        exl3_gemm(dummy_nh, layer.w2_trellis[0], bits=bits, cb=cb,
                  B_i32=layer.w2_trellis_i32[0])
        # Had on output shapes
        dummy_gate = torch.zeros(1, N, dtype=torch.float16, device=device)
        had_r_128(dummy_gate, torch.empty_like(dummy_gate),
                  None, layer.w1_svh[0], 1.0)
        dummy_down = torch.zeros(1, K, dtype=torch.float16, device=device)
        had_r_128(dummy_down, torch.empty_like(dummy_down),
                  None, layer.w2_svh[0], 1.0)

        # --- FP16 expert dequant (Phase 4) ---
        # Dequant MoE expert weights to FP16 at load time for FP16 GEMM kernel.
        # Uses the same identity-matrix pipeline as the dense dequant path.
        if _USE_FP16_EXPERTS:
            K = layer.w13_suh.shape[-1]     # hidden_size
            N_gate = layer.w1_svh.shape[-1]  # intermediate_size_per_partition

            logger.info(
                "EXL3: dequanting MoE experts to FP16 "
                "(E=%d, gate/up %dx%d, down %dx%d)",
                layer.w1_trellis.shape[0], K, N_gate, N_gate, K)

            # Gate: (E, K) -> (E, K, N_gate)
            cb = layer.exl3_cb
            layer.w1_fp16 = torch.nn.Parameter(
                _dequant_expert_weights(
                    layer.w1_trellis, layer.w1_trellis_i32,
                    layer.w13_suh[:, 0, :], layer.w1_svh,
                    K, N_gate, bits, cb=cb,
                ),
                requires_grad=False,
            )
            # Up: (E, K) -> (E, K, N_gate)
            layer.w3_fp16 = torch.nn.Parameter(
                _dequant_expert_weights(
                    layer.w3_trellis, layer.w3_trellis_i32,
                    layer.w13_suh[:, 1, :], layer.w3_svh,
                    K, N_gate, bits, cb=cb,
                ),
                requires_grad=False,
            )
            # Down: (E, N_gate) -> (E, N_gate, K)
            layer.w2_fp16 = torch.nn.Parameter(
                _dequant_expert_weights(
                    layer.w2_trellis, layer.w2_trellis_i32,
                    layer.w2_suh, layer.w2_svh,
                    N_gate, K, bits, cb=cb,
                ),
                requires_grad=False,
            )

            fp16_bytes = (
                layer.w1_fp16.numel() + layer.w3_fp16.numel()
                + layer.w2_fp16.numel()
            ) * 2
            logger.info(
                "EXL3: FP16 expert cache: %.1f MB/layer on %s",
                fp16_bytes / 1024 / 1024, device)

        # Warmup HIP rocWMMA fused MoE GEMM kernel (Phase 3)
        try:
            from vllm.model_executor.layers.quantization.exl3_kernels import (
                _HAS_HIP_MOE,
            )
            if _HAS_HIP_MOE:
                from vllm.model_executor.layers.quantization.exl3_kernels import (
                    _get_hip_moe_splitk_buf,
                    _hip_moe_auto_split_k,
                )
                # Pre-allocate split-K buffers for decode shapes
                BLOCK_M = 16
                top_k = 10  # Qwen3-Next default
                EM_warmup = top_k * BLOCK_M
                num_m_blocks_warmup = EM_warmup // BLOCK_M
                # Gate/Up shape
                tiles_k_gu = K // 16
                sk_gu = _hip_moe_auto_split_k(num_m_blocks_warmup, tiles_k_gu,
                                               is_decode=True)
                if sk_gu > 1:
                    _get_hip_moe_splitk_buf(sk_gu, EM_warmup, N, device)
                # Down shape
                tiles_k_dn = N // 16
                sk_dn = _hip_moe_auto_split_k(num_m_blocks_warmup, tiles_k_dn,
                                               is_decode=True)
                if sk_dn > 1:
                    _get_hip_moe_splitk_buf(sk_dn, EM_warmup, K, device)
                logger.info(
                    "EXL3 HIP MoE warmup: gate/up sk=%d, down sk=%d, EM=%d",
                    sk_gu, sk_dn, EM_warmup)
        except ImportError:
            pass

        # Warmup fused MoE GEMM+Had kernels (pre-compile + auto-tune)
        if _USE_FUSED_MOE_HAD:
            from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
                _get_h16,
                exl3_fused_moe_gemm_had as _fused_moe_had_impl,
            )
            _get_h16(device)
            # Build minimal MoE-shaped inputs for one expert
            BLOCK_M = 16
            EM_warmup = BLOCK_M  # 1 block
            dummy_sorted = torch.arange(
                EM_warmup, device=device, dtype=torch.int32)
            dummy_eid = torch.zeros(1, device=device, dtype=torch.int32)
            dummy_npp = torch.tensor(
                [EM_warmup], device=device, dtype=torch.int32)
            # Gate/Up: (1, K) -> (EM, N_gate)
            dummy_a_k = torch.zeros(
                EM_warmup, K, dtype=torch.float16, device=device)
            _fused_moe_had_impl(
                dummy_a_k, layer.w1_trellis,
                dummy_sorted, dummy_eid, dummy_npp,
                layer.w1_svh, EM_max=EM_warmup, bits=bits, cb=cb,
                B_i32=layer.w1_trellis_i32)
            # Down: (1, N) -> (EM, K_out)
            dummy_a_n = torch.zeros(
                EM_warmup, N, dtype=torch.float16, device=device)
            _fused_moe_had_impl(
                dummy_a_n, layer.w2_trellis,
                dummy_sorted, dummy_eid, dummy_npp,
                layer.w2_svh, EM_max=EM_warmup, bits=bits, cb=cb,
                B_i32=layer.w2_trellis_i32)

    def get_fused_moe_quant_config(self, layer):
        return None

    def _apply_expert(self, layer, token, eid, bits, cb, exl3_gemm, had_r_128):
        """Run token(s) through one expert using Python int eid indexing.

        Used in eager path and all-experts graph capture path where eid
        is a Python int (basic indexing = views with fixed offsets).
        NOT safe for graph capture with GPU tensor eid.
        """
        # Gate: had -> gemm -> had
        xh = torch.empty_like(token)
        had_r_128(token, xh, layer.w13_suh[eid, 0], None, 1.0)
        gate = exl3_gemm(
            xh, layer.w1_trellis[eid], bits=bits, cb=cb,
            B_i32=layer.w1_trellis_i32[eid],
        )
        gate_h = torch.empty_like(gate)
        had_r_128(gate, gate_h, None, layer.w1_svh[eid], 1.0)

        # Up: had -> gemm -> had
        xh_up = torch.empty_like(token)
        had_r_128(token, xh_up, layer.w13_suh[eid, 1], None, 1.0)
        up = exl3_gemm(
            xh_up, layer.w3_trellis[eid], bits=bits, cb=cb,
            B_i32=layer.w3_trellis_i32[eid],
        )
        up_h = torch.empty_like(up)
        had_r_128(up, up_h, None, layer.w3_svh[eid], 1.0)

        # Activation
        hidden = F.silu(gate_h) * up_h

        # Down: had -> gemm -> had
        xh_down = torch.empty_like(hidden)
        had_r_128(hidden, xh_down, layer.w2_suh[eid], None, 1.0)
        down = exl3_gemm(
            xh_down, layer.w2_trellis[eid], bits=bits, cb=cb,
            B_i32=layer.w2_trellis_i32[eid],
        )
        down_h = torch.empty_like(down)
        had_r_128(down, down_h, None, layer.w2_svh[eid], 1.0)
        return down_h

    def _apply_expert_gathered(self, token, w13_suh_e, w1_trellis_e,
                               w1_svh_e, w3_trellis_e, w3_svh_e,
                               w2_suh_e, w2_trellis_e, w2_svh_e,
                               bits, cb, exl3_gemm, had_r_128):
        """Run token(s) through one expert using pre-gathered weights.

        Used in per-slot graph capture path. Weights are already gathered
        via index_select (graph-safe CUDA kernel), so no GPU tensor
        indexing needed here — only Python int indexing on the suh dim.
        """
        # Gate: had -> gemm -> had
        xh = torch.empty_like(token)
        had_r_128(token, xh, w13_suh_e[0], None, 1.0)
        gate = exl3_gemm(
            xh, w1_trellis_e, bits=bits, cb=cb,
            B_i32=w1_trellis_e.view(torch.int32),
        )
        gate_h = torch.empty_like(gate)
        had_r_128(gate, gate_h, None, w1_svh_e, 1.0)

        # Up: had -> gemm -> had
        xh_up = torch.empty_like(token)
        had_r_128(token, xh_up, w13_suh_e[1], None, 1.0)
        up = exl3_gemm(
            xh_up, w3_trellis_e, bits=bits, cb=cb,
            B_i32=w3_trellis_e.view(torch.int32),
        )
        up_h = torch.empty_like(up)
        had_r_128(up, up_h, None, w3_svh_e, 1.0)

        # Activation
        hidden = F.silu(gate_h) * up_h

        # Down: had -> gemm -> had
        xh_down = torch.empty_like(hidden)
        had_r_128(hidden, xh_down, w2_suh_e, None, 1.0)
        down = exl3_gemm(
            xh_down, w2_trellis_e, bits=bits, cb=cb,
            B_i32=w2_trellis_e.view(torch.int32),
        )
        down_h = torch.empty_like(down)
        had_r_128(down, down_h, None, w2_svh_e, 1.0)
        return down_h

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "torch.nn.Module | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # rc4 added `shared_experts` before `shared_experts_input` in the base
        # apply() signature. Both are unused here (Qwen3.5 MoE has no shared
        # experts; Qwen3-Next handles its shared expert in the model module,
        # not in the quant method).
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
            moe_align_block_size,
        )
        from vllm.model_executor.layers.quantization.exl3_kernels import (
            batched_had_r_128,
            exl3_fused_moe_gate_up,
            exl3_fused_moe_gemm,
            exl3_fused_moe_gemm_had,
            exl3_moe_had_gemm_had,
            exl3_moe_had_gemm_had_v2,
            exl3_trellis_to_fp16,
        )

        bits = layer.exl3_bits
        cb = getattr(layer, 'exl3_cb', 0)
        M, K = x.shape[0], x.shape[-1]
        top_k = topk_ids.shape[1]
        num_experts = layer.w1_trellis.shape[0]
        N_out = layer.w2_svh.shape[1]  # hidden_size
        total = M * top_k
        BLOCK_M = 16

        expert_map = getattr(layer, "expert_map", None)

        # Global expert count for moe_align_block_size (topk_ids use
        # global IDs; expert_map maps global→local).
        if expert_map is not None:
            global_num_experts = expert_map.shape[0]
        else:
            global_num_experts = num_experts

        # === UNIFIED FUSED PATH (works in both eager and graph capture) ===
        # One kernel launch per projection (gate/up/down) instead of
        # per-expert loop. All tensor shapes are Python-int determined
        # (EM_max, total, N_out). num_post_pad_t stays as tensor — kernel
        # loads it for runtime bounds checking. No .item(), no
        # GPU-data-dependent branches.

        x_expanded = x.unsqueeze(1).expand(
            -1, top_k, -1).reshape(total, K)

        sorted_token_ids, expert_ids, num_post_pad_t = \
            moe_align_block_size(
                topk_ids, BLOCK_M, global_num_experts, expert_map,
                pad_sorted_ids=True)

        if M == 1:
            # Decode: tight fixed bound for graph capture compatibility
            EM_max = top_k * BLOCK_M
        else:
            # Prefill/warmup: use sorted_token_ids tensor size as upper bound.
            # This is a Python int (tensor shape) — no .item() needed, so it's
            # safe during CUDA graph capture. The kernel skips excess blocks
            # via the num_tokens_post_padded runtime check.
            EM_max = sorted_token_ids.shape[0]

        # Gather: fixed-size EM_max rows (no .item() needed for decode)
        num_m_blocks = EM_max // BLOCK_M
        tid_clamped = sorted_token_ids[:EM_max].clamp(
            max=total - 1).long()
        x_sorted = x_expanded[tid_clamped]

        eid_blocks = expert_ids[:num_m_blocks]
        eid_per_token = eid_blocks.repeat_interleave(
            BLOCK_M).clamp(min=0)

        identity_ids = torch.arange(
            EM_max, device=x.device, dtype=sorted_token_ids.dtype)

        # Get FP16 expert weights if available (Phase 4)
        w1_fp16 = getattr(layer, 'w1_fp16', None)
        w3_fp16 = getattr(layer, 'w3_fp16', None)
        w2_fp16 = getattr(layer, 'w2_fp16', None)

        # When FP16 experts are available, the had→gemm→had pipeline is
        # baked into the FP16 weights — no separate Had transforms needed.
        # Just do: x_sorted @ W_fp16 (via FP16 MoE GEMM kernel).
        if w1_fp16 is not None:
            # FP16 path: skip Had transforms, use pre-dequanted weights
            gate_h = exl3_fused_moe_gemm(
                x_sorted, layer.w1_trellis,
                identity_ids, eid_blocks,
                num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                B_i32=layer.w1_trellis_i32,
                B_fp16=w1_fp16)

            up_h = exl3_fused_moe_gemm(
                x_sorted, layer.w3_trellis,
                identity_ids, eid_blocks,
                num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                B_i32=layer.w3_trellis_i32,
                B_fp16=w3_fp16)

            hidden = F.silu(gate_h) * up_h

            down_h = exl3_fused_moe_gemm(
                hidden, layer.w2_trellis,
                identity_ids, eid_blocks,
                num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                B_i32=layer.w2_trellis_i32,
                B_fp16=w2_fp16)
        elif _USE_MOE_COMPOUND_V2 and cb == 0:
            # V2: Zero-allocation compound Had→GEMM→Had (cb=0 only — HIP).
            # 4 graph nodes per MoE layer (gate, up, silu_mul, down).
            # ALL buffers allocated here (visible to torch.compile memory
            # planner → CUDA graph memory pool) and passed in via mutates_args.
            from vllm.model_executor.layers.quantization.exl3_kernels import (
                _get_moe_scratch_buf, _hip_moe_auto_split_k,
            )

            N_inter = layer.w1_svh.shape[1]  # intermediate_size_per_partition
            num_m_blocks = eid_blocks.shape[0]

            # Compute split-K for gate/up (K_in=K → N=N_inter)
            num_k_tiles_gu = K // 16
            split_k_gu = _hip_moe_auto_split_k(num_m_blocks, num_k_tiles_gu,
                                               is_decode=False)
            split_k_gu = min(split_k_gu, num_k_tiles_gu)

            # Compute split-K for down (K_in=N_inter → N=K)
            num_k_tiles_dn = N_inter // 16
            split_k_dn = _hip_moe_auto_split_k(num_m_blocks, num_k_tiles_dn,
                                               is_decode=False)
            split_k_dn = min(split_k_dn, num_k_tiles_dn)

            # Scratch buffers for gate/up (reusable — sequential, not concurrent)
            buf_xh_gu = _get_moe_scratch_buf("xh_gu", EM_max, K, x.device)
            buf_C_gu = _get_moe_scratch_buf("C_gu", EM_max, N_inter, x.device)
            gate_out = _get_moe_scratch_buf("gate_out", EM_max, N_inter,
                                            x.device)
            if split_k_gu > 1:
                buf_Cp_gu = _get_moe_scratch_buf(
                    "Cp_gu", split_k_gu * EM_max, N_inter, x.device
                ).view(split_k_gu, EM_max, N_inter)
                buf_Cp_gu.zero_()
            else:
                buf_Cp_gu = torch.empty(
                    1, 1, 1, dtype=torch.float16, device=x.device)

            # Gate projection: Had→GEMM→Had
            exl3_moe_had_gemm_had_v2(
                x_sorted, layer.w13_suh_gate,
                layer.w1_trellis, layer.w1_svh,
                eid_per_token, eid_blocks,
                num_post_pad_t, buf_xh_gu, buf_C_gu, buf_Cp_gu, gate_out,
                EM_max=EM_max, bits=bits, split_k=split_k_gu,
                B_i32=layer.w1_trellis_i32)

            # Up projection: reuse buf_xh_gu, buf_C_gu, buf_Cp_gu (gate done)
            up_out = _get_moe_scratch_buf("up_out", EM_max, N_inter, x.device)

            exl3_moe_had_gemm_had_v2(
                x_sorted, layer.w13_suh_up,
                layer.w3_trellis, layer.w3_svh,
                eid_per_token, eid_blocks,
                num_post_pad_t, buf_xh_gu, buf_C_gu, buf_Cp_gu, up_out,
                EM_max=EM_max, bits=bits, split_k=split_k_gu,
                B_i32=layer.w3_trellis_i32)

            # Activation: silu(gate) * up
            hidden = F.silu(gate_out) * up_out

            # Down projection: different dimensions (N_inter → K)
            buf_xh_dn = _get_moe_scratch_buf("xh_dn", EM_max, N_inter,
                                              x.device)
            buf_C_dn = _get_moe_scratch_buf("C_dn", EM_max, N_out, x.device)
            down_out = _get_moe_scratch_buf("down_out", EM_max, N_out,
                                            x.device)
            if split_k_dn > 1:
                buf_Cp_dn = _get_moe_scratch_buf(
                    "Cp_dn", split_k_dn * EM_max, N_out, x.device
                ).view(split_k_dn, EM_max, N_out)
                buf_Cp_dn.zero_()
            else:
                buf_Cp_dn = torch.empty(
                    1, 1, 1, dtype=torch.float16, device=x.device)

            exl3_moe_had_gemm_had_v2(
                hidden, layer.w2_suh,
                layer.w2_trellis, layer.w2_svh,
                eid_per_token, eid_blocks,
                num_post_pad_t, buf_xh_dn, buf_C_dn, buf_Cp_dn, down_out,
                EM_max=EM_max, bits=bits, split_k=split_k_dn,
                B_i32=layer.w2_trellis_i32)

            down_h = down_out

        elif _USE_MOE_COMPOUND and M > 1 and cb == 0:
            # V7/V8: Compound Had→GEMM→Had — prefill only (M>1), cb=0 only (HIP).
            # Reduces Python dispatch overhead: 2 ops/layer vs 10+.
            # Decode (M=1) uses legacy path below for CUDA graph compatibility.
            hidden = exl3_fused_moe_gate_up(
                x_sorted,
                layer.w13_suh_gate, layer.w13_suh_up,
                layer.w1_trellis_i32, layer.w3_trellis_i32,
                layer.w1_svh, layer.w3_svh,
                eid_per_token, identity_ids, eid_blocks,
                num_post_pad_t, EM_max=EM_max, bits=bits)

            down_h = exl3_moe_had_gemm_had(
                hidden, layer.w2_suh,
                layer.w2_trellis, layer.w2_svh,
                eid_per_token, eid_blocks,
                num_post_pad_t, EM_max=EM_max, bits=bits,
                B_i32=layer.w2_trellis_i32)
        else:
            # Legacy path: separate Had + GEMM ops (10 graph nodes per layer)
            # V5: Dual Had for gate+up input (1 kernel instead of 2)
            # Uses pre-split contiguous suh slices (no .contiguous() per call)
            if _USE_FUSED_GATE_UP and not _USE_FUSED_MOE_HAD:
                xh_gate, xh_up = torch.ops.vllm.batched_dual_had_r_128(
                    x_sorted,
                    layer.w13_suh_gate,
                    layer.w13_suh_up,
                    eid_per_token, 1)  # pre=True
            else:
                xh_gate = batched_had_r_128(
                    x_sorted, layer.w13_suh_gate,
                    eid_per_token, pre=True)
                xh_up = batched_had_r_128(
                    x_sorted, layer.w13_suh_up,
                    eid_per_token, pre=True)

            # Gate projection: GEMM(+Had)
            if _USE_FUSED_MOE_HAD:
                gate_h = exl3_fused_moe_gemm_had(
                    xh_gate, layer.w1_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, layer.w1_svh,
                    EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w1_trellis_i32)
            else:
                gate = exl3_fused_moe_gemm(
                    xh_gate, layer.w1_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w1_trellis_i32)
                gate_h = batched_had_r_128(
                    gate, layer.w1_svh,
                    eid_per_token, pre=False)

            # Up projection: GEMM(+Had)
            if _USE_FUSED_MOE_HAD:
                up_h = exl3_fused_moe_gemm_had(
                    xh_up, layer.w3_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, layer.w3_svh,
                    EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w3_trellis_i32)
            else:
                up = exl3_fused_moe_gemm(
                    xh_up, layer.w3_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w3_trellis_i32)
                up_h = batched_had_r_128(
                    up, layer.w3_svh,
                    eid_per_token, pre=False)

            hidden = F.silu(gate_h) * up_h

            # Down projection: Had → GEMM(+Had)
            xh_down = batched_had_r_128(
                hidden, layer.w2_suh,
                eid_per_token, pre=True)
            if _USE_FUSED_MOE_HAD:
                down_h = exl3_fused_moe_gemm_had(
                    xh_down, layer.w2_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, layer.w2_svh,
                    EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w2_trellis_i32)
            else:
                down = exl3_fused_moe_gemm(
                    xh_down, layer.w2_trellis,
                    identity_ids, eid_blocks,
                    num_post_pad_t, EM_max=EM_max, bits=bits, cb=cb,
                    B_i32=layer.w2_trellis_i32)
                down_h = batched_had_r_128(
                    down, layer.w2_svh,
                    eid_per_token, pre=False)

        # Scatter back to token order (padding → discard row)
        output_buf = torch.zeros(
            total + 1, N_out, dtype=x.dtype, device=x.device)
        scatter_ids = sorted_token_ids[:EM_max].clamp(
            min=0, max=total).long()
        scatter_ids_2d = scatter_ids.unsqueeze(1).expand(
            -1, N_out)
        output_buf.scatter_(
            0, scatter_ids_2d, down_h.to(x.dtype))
        output = output_buf[:total]

        # Apply routing weights and sum over top_k
        output = output.reshape(M, top_k, N_out)
        output = (output * topk_weights.unsqueeze(-1)).sum(dim=1)

        # topk_weights is float32 (all vLLM routers cast to float32), which
        # promotes output to float32. Cast back to match hidden_states dtype.
        return output.to(x.dtype)
