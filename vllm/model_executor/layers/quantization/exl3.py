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
        else:
            if loaded_shard_id is not None:
                linear_weight_loader(param, loaded_weight, loaded_shard_id)
            else:
                linear_weight_loader(param, loaded_weight)

    return exl3_weight_loader


def _shard_idx(shard_id):
    """Convert shard_id to integer index."""
    if shard_id is None:
        return None
    if isinstance(shard_id, str):
        return {"q": 0, "k": 1, "v": 2}[shard_id]
    return shard_id


def _load_trellis(param: "EXL3TrellisParameter", loaded_weight: torch.Tensor,
                  shard_id, output_sizes: list[int]):
    """Load trellis weight, handling merged column and QKV cases."""
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
    assert param.data.shape[1] == loaded_weight.shape[0], (
        f"suh dim mismatch: param row={param.data.shape[1]}, "
        f"loaded={loaded_weight.shape[0]}"
    )
    param.data[idx].copy_(loaded_weight)


def _load_svh(param: "EXL3ScaleParameter", loaded_weight: torch.Tensor,
              shard_id, output_sizes: list[int]):
    """Load svh (output scale), handling merged column and QKV cases."""
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
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)
        return

    idx = _shard_idx(shard_id)
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

    def __init__(
        self,
        weight_bits: int,
        head_bits: int,
        tensor_storage: dict[str, Any],
    ) -> None:
        super().__init__()
        self.weight_bits = weight_bits
        self.head_bits = head_bits
        self.tensor_storage = tensor_storage

        # Build lookup: prefix -> bits_per_weight (only for EXL3 layers)
        self._layer_bits: dict[str, int] = {}
        self._unquantized_layers: list[str] = []

        for prefix, info in tensor_storage.items():
            if info.get("quant_format") == "exl3":
                self._layer_bits[prefix] = info["bits_per_weight"]
            else:
                self._unquantized_layers.append(prefix)

    def __repr__(self) -> str:
        return (
            f"EXL3Config(weight_bits={self.weight_bits}, "
            f"head_bits={self.head_bits})"
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
        return cls(weight_bits, head_bits, tensor_storage)

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
        for key, bits in self._layer_bits.items():
            if key.startswith(stripped + ".") or key.startswith(prefix + "."):
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

            return None
        else:
            # Fallback when tensor_storage not available
            if "lm_head" in prefix:
                return self.head_bits
            return self.weight_bits

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Union["LinearMethodBase", "QuantizeMethodBase"] | None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
        )

        if isinstance(layer, FusedMoE):
            bits = self._get_moe_layer_bits(prefix)
            if bits is not None:
                return EXL3FusedMoEMethod(self, bits, layer.moe_config)
            return None
        if isinstance(layer, LinearBase):
            bits = self._get_layer_bits(prefix)
            if bits is not None:
                return EXL3LinearMethod(self, bits)
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

        layer.exl3_bits = self.bits
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer._exl3_dequant = False

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
        is_row_parallel = (input_size_per_partition != input_size)

        if is_row_parallel:
            # RowParallel: output_partition_sizes are already full size
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

        # Full-size trellis
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
        # Override TP attributes so weight loader doesn't TP-shard
        trellis.tp_size = 1
        trellis.tp_rank = 0

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

        # Full-size svh
        svh = EXL3ScaleParameter(
            data=torch.ones(full_output_size, dtype=torch.float16),
            weight_loader=exl3_loader,
        )
        svh.tp_size = 1
        svh.tp_rank = 0

        layer.register_parameter("trellis", trellis)
        layer.register_parameter("suh", suh)
        layer.register_parameter("svh", svh)

        layer.exl3_bits = self.bits
        layer.exl3_output_partition_sizes = output_partition_sizes
        layer._exl3_dequant = True
        layer._exl3_dequant_full_output_sizes = full_output_partition_sizes
        layer._exl3_dequant_input_size = input_size
        layer._exl3_dequant_tp_size = tp_size
        layer._exl3_dequant_is_row_parallel = is_row_parallel
        layer._exl3_dequant_params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, '_exl3_dequant', False):
            self._process_weights_dequant(layer)
            return

        trellis_data = layer.trellis.data
        layer.trellis = torch.nn.Parameter(trellis_data, requires_grad=False)
        layer.suh = torch.nn.Parameter(layer.suh.data, requires_grad=False)
        layer.svh = torch.nn.Parameter(layer.svh.data, requires_grad=False)

        # Precompute int32 view for the Triton kernel
        layer.trellis_i32 = layer.trellis.data.view(torch.int32)

        # Eagerly populate device-side caches so that no CPU→CUDA copies
        # happen during CUDA graph capture.
        device = trellis_data.device
        from vllm.model_executor.layers.quantization.exl3_kernels.triton_kernel import (
            get_bit_tables,
        )
        from vllm.model_executor.layers.quantization.exl3_kernels.hadamard import (
            _get_had128,
        )
        get_bit_tables(layer.exl3_bits, device)
        _get_had128(device)

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
            proj_trellis = trellis.narrow(1, tile_offset, proj_tiles)
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
                    xh, proj_trellis, bits=bits, cb=0,
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
            had_r_128,
        )

        trellis = layer.trellis
        suh = layer.suh
        svh = layer.svh
        bits = layer.exl3_bits
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
                xh, trellis, bits=bits, cb=0,
                B_i32=layer.trellis_i32,
            )

            out_h = torch.empty_like(out)
            had_r_128(out, out_h, None, svh, 1.0)
        else:
            # Merged layer: run each projection separately with its own suh
            num_proj = suh.shape[0]
            N_total = sum(output_partition_sizes)
            out_h = torch.empty((M, N_total), dtype=x_2d.dtype,
                                device=x_2d.device)

            tile_offset = 0
            out_offset = 0
            for i in range(num_proj):
                proj_suh = suh[i]
                proj_size = output_partition_sizes[i]
                proj_tiles = proj_size // 16

                # Input Hadamard with this projection's suh
                xh = torch.empty_like(x_2d)
                had_r_128(x_2d, xh, proj_suh, None, 1.0)

                # Slice trellis for this projection
                proj_trellis = trellis.narrow(1, tile_offset, proj_tiles)
                proj_trellis_i32 = proj_trellis.contiguous().view(torch.int32)

                # GEMM
                proj_out = exl3_gemm(
                    xh, proj_trellis, bits=bits, cb=0,
                    B_i32=proj_trellis_i32,
                )

                # Output Hadamard with this projection's svh slice
                proj_svh = svh.narrow(0, out_offset, proj_size)
                proj_out_h = out_h.narrow(1, out_offset, proj_size)
                had_r_128(proj_out, proj_out_h, None, proj_svh, 1.0)

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

        layer.exl3_bits = self.bits
        layer._exl3_dequant = False

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

        layer.exl3_bits = self.bits
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
                xh, trellis, bits=bits, cb=0, B_i32=trellis_i32,
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
            xh, layer.trellis, bits=layer.exl3_bits, cb=0,
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
    ):
        # Map global expert_id to local for expert parallelism
        expert_map = getattr(moe_layer, "_expert_map", None)
        if expert_map is not None:
            local_id = expert_map[expert_id].item()
            if local_id == -1:
                return  # Expert not local to this rank
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
        exl3_gemm(dummy_h, layer.w1_trellis[0], bits=bits, cb=0,
                  B_i32=layer.w1_trellis_i32[0])
        # Down GEMM: (1, N) -> (1, K)
        dummy_n = torch.zeros(1, N, dtype=torch.float16, device=device)
        dummy_nh = torch.empty_like(dummy_n)
        had_r_128(dummy_n, dummy_nh, layer.w2_suh[0], None, 1.0)
        exl3_gemm(dummy_nh, layer.w2_trellis[0], bits=bits, cb=0,
                  B_i32=layer.w2_trellis_i32[0])
        # Had on output shapes
        dummy_gate = torch.zeros(1, N, dtype=torch.float16, device=device)
        had_r_128(dummy_gate, torch.empty_like(dummy_gate),
                  None, layer.w1_svh[0], 1.0)
        dummy_down = torch.zeros(1, K, dtype=torch.float16, device=device)
        had_r_128(dummy_down, torch.empty_like(dummy_down),
                  None, layer.w2_svh[0], 1.0)

    def get_fused_moe_quant_config(self, layer):
        return None

    def _apply_expert(self, layer, token, eid, bits, exl3_gemm, had_r_128):
        """Run token(s) through one expert using Python int eid indexing.

        Used in eager path and all-experts graph capture path where eid
        is a Python int (basic indexing = views with fixed offsets).
        NOT safe for graph capture with GPU tensor eid.
        """
        # Gate: had -> gemm -> had
        xh = torch.empty_like(token)
        had_r_128(token, xh, layer.w13_suh[eid, 0], None, 1.0)
        gate = exl3_gemm(
            xh, layer.w1_trellis[eid], bits=bits, cb=0,
            B_i32=layer.w1_trellis_i32[eid],
        )
        gate_h = torch.empty_like(gate)
        had_r_128(gate, gate_h, None, layer.w1_svh[eid], 1.0)

        # Up: had -> gemm -> had
        xh_up = torch.empty_like(token)
        had_r_128(token, xh_up, layer.w13_suh[eid, 1], None, 1.0)
        up = exl3_gemm(
            xh_up, layer.w3_trellis[eid], bits=bits, cb=0,
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
            xh_down, layer.w2_trellis[eid], bits=bits, cb=0,
            B_i32=layer.w2_trellis_i32[eid],
        )
        down_h = torch.empty_like(down)
        had_r_128(down, down_h, None, layer.w2_svh[eid], 1.0)
        return down_h

    def _apply_expert_gathered(self, token, w13_suh_e, w1_trellis_e,
                               w1_svh_e, w3_trellis_e, w3_svh_e,
                               w2_suh_e, w2_trellis_e, w2_svh_e,
                               bits, exl3_gemm, had_r_128):
        """Run token(s) through one expert using pre-gathered weights.

        Used in per-slot graph capture path. Weights are already gathered
        via index_select (graph-safe CUDA kernel), so no GPU tensor
        indexing needed here — only Python int indexing on the suh dim.
        """
        # Gate: had -> gemm -> had
        xh = torch.empty_like(token)
        had_r_128(token, xh, w13_suh_e[0], None, 1.0)
        gate = exl3_gemm(
            xh, w1_trellis_e, bits=bits, cb=0,
            B_i32=w1_trellis_e.view(torch.int32),
        )
        gate_h = torch.empty_like(gate)
        had_r_128(gate, gate_h, None, w1_svh_e, 1.0)

        # Up: had -> gemm -> had
        xh_up = torch.empty_like(token)
        had_r_128(token, xh_up, w13_suh_e[1], None, 1.0)
        up = exl3_gemm(
            xh_up, w3_trellis_e, bits=bits, cb=0,
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
            xh_down, w2_trellis_e, bits=bits, cb=0,
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
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.exl3_kernels import (
            exl3_gemm,
            had_r_128,
        )

        bits = layer.exl3_bits
        M, K = x.shape[0], x.shape[-1]
        top_k = topk_ids.shape[1]
        num_experts = layer.w1_trellis.shape[0]
        N_out = layer.w2_svh.shape[1]  # hidden_size
        total = M * top_k

        flat_topk_ids = topk_ids.reshape(-1)  # (total,)
        expert_map = getattr(layer, "expert_map", None)

        # Two paths: eager (profiling/compilation) and graph capture (decode).
        # Eager can use .item() and dynamic shapes; graph capture cannot.
        capturing = torch.cuda.is_current_stream_capturing()

        if not capturing:
            # === EAGER PATH: sort by expert, process only active experts ===
            # Efficient for large M (prefill/profiling). Uses .item() for
            # expert boundaries — OK because not in graph capture.
            x_expanded = x.unsqueeze(1).expand(
                -1, top_k, -1).reshape(total, K)

            # Map to local expert IDs
            if expert_map is not None:
                local_ids = expert_map[flat_topk_ids]
            else:
                local_ids = flat_topk_ids

            # Sort by expert for coalesced access
            sorted_indices = local_ids.argsort()
            sorted_ids = local_ids[sorted_indices]
            sorted_x = x_expanded[sorted_indices]

            # Compute expert boundaries. Non-local tokens (id=-1) sort
            # to the front; skip them by starting offsets after them.
            num_invalid = (sorted_ids < 0).sum()
            expert_counts = torch.zeros(
                num_experts, dtype=torch.int64, device=x.device)
            valid = sorted_ids >= 0
            safe_ids = sorted_ids.clamp(min=0)
            expert_counts.scatter_add_(
                0, safe_ids,
                valid.to(torch.int64),
            )
            expert_offsets = torch.zeros(
                num_experts + 1, dtype=torch.int64, device=x.device)
            expert_offsets[0] = num_invalid
            torch.cumsum(expert_counts, dim=0, out=expert_offsets[1:])
            expert_offsets[1:] += num_invalid

            sorted_out = torch.zeros(
                total, N_out, dtype=x.dtype, device=x.device)

            for eid in range(num_experts):
                count = expert_counts[eid].item()
                if count == 0:
                    continue
                offset = expert_offsets[eid].item()
                tokens = sorted_x[offset:offset + count]
                down_h = self._apply_expert(
                    layer, tokens, eid, bits, exl3_gemm, had_r_128)
                sorted_out[offset:offset + count] = down_h

            # Unsort
            output = torch.empty_like(sorted_out)
            output[sorted_indices] = sorted_out

        else:
            # === GRAPH CAPTURE PATH ===
            # No .item(), no boolean indexing, fixed shapes only.
            # Two sub-paths based on M size:
            #   Small M (decode): per-slot with index_select
            #   Large M (prefill): all-experts with torch.where masking
            if expert_map is not None:
                local_ids = expert_map[flat_topk_ids]
            else:
                local_ids = flat_topk_ids

            local_ids_safe = local_ids.clamp(min=0)

            x_expanded = x.unsqueeze(1).expand(
                -1, top_k, -1).reshape(total, K)
            output = torch.zeros(
                total, N_out, dtype=x.dtype, device=x.device)

            if total <= num_experts:
                # -- Small M (decode): per-slot with index_select --
                # total = M * top_k; for M=1 decode, total=10.
                # index_select is a proper CUDA kernel that reads the
                # index tensor on GPU at runtime — graph-safe. squeeze(0)
                # and Python int indexing ([0], [1]) are views — also safe.
                for k in range(total):
                    idx = local_ids_safe[k:k+1]  # (1,) tensor
                    token = x_expanded[k:k+1]
                    down_h = self._apply_expert_gathered(
                        token,
                        torch.index_select(
                            layer.w13_suh, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w1_trellis, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w1_svh, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w3_trellis, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w3_svh, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w2_suh, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w2_trellis, 0, idx).squeeze(0),
                        torch.index_select(
                            layer.w2_svh, 0, idx).squeeze(0),
                        bits, exl3_gemm, had_r_128,
                    )
                    if expert_map is not None:
                        valid = (local_ids[k] >= 0).reshape(1, 1)
                        down_h = torch.where(
                            valid, down_h, torch.zeros_like(down_h))
                    output[k:k+1] = down_h
            else:
                # -- Large M (prefill): all-experts loop --
                # Fixed num_experts iterations (64 for EP=8).
                # Each iteration runs ALL tokens through one expert.
                # torch.where selects valid results, ignoring NaN
                # from wrong-expert fp16 overflow (IEEE 754:
                # torch.where picks output when mask=False, so NaN
                # in down_h never propagates to non-matching slots).
                for eid in range(num_experts):
                    mask = (local_ids == eid).unsqueeze(1)
                    down_h = self._apply_expert(
                        layer, x_expanded, eid, bits,
                        exl3_gemm, had_r_128)
                    output = torch.where(mask, down_h, output)

        # Apply routing weights and sum over top_k
        output = output.reshape(M, top_k, N_out)
        output = (output * topk_weights.unsqueeze(-1)).sum(dim=1)

        # topk_weights is float32 (all vLLM routers cast to float32), which
        # promotes output to float32. Cast back to match hidden_states dtype.
        return output.to(x.dtype)
