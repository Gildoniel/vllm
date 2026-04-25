# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""

# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

import typing

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm.env_override  # noqa: F401

# Install a permissive shim on torch.ops._C so module-level lookups for ops
# missing in older locally-built _C.abi3.so don't crash imports. Fusion
# passes often do `OP = torch.ops._C.<name>.default` at module load; if the
# kernel isn't compiled into the local build, the lookup raises and breaks
# the whole import chain. With this shim, missing ops resolve to a stub
# whose `.default` is None — fusion passes that reference None at pattern
# definition time silently skip themselves, and runtime call sites that
# actually need the op still raise (the shim's __call__ does).
def _install_ops_C_compat():
    import torch
    try:
        ns = torch.ops._C
    except Exception:
        return
    orig_getattr = type(ns).__getattr__

    class _MissingOp:
        """Stand-in for ops not present in the locally-built _C.abi3.so.

        Module-level lookups (`OP = torch.ops._C.<name>.default`) succeed
        with `OP is None`, so existing `if OP is not None` guards work.
        Actual call attempts raise so that we don't silently emit garbage.
        Any overload-name access (.default, .out, etc.) returns None.
        """
        def __init__(self, name):
            self._name = name
        def __getattr__(self, attr):
            # .default, .out, etc. → None (so module-level guards see None)
            return None
        def __call__(self, *args, **kwargs):
            raise AttributeError(
                f"vllm._C missing op '{self._name}' at runtime "
                f"(local _C.abi3.so doesn't export it)"
            )
        def __bool__(self):
            return False
        def __eq__(self, other):
            return other is None or other is self

    def patched_getattr(self, name):
        try:
            return orig_getattr(self, name)
        except (AttributeError, RuntimeError):
            return _MissingOp(name)

    type(ns).__getattr__ = patched_getattr


_install_ops_C_compat()
del _install_ops_C_compat

MODULE_ATTRS = {
    "AsyncEngineArgs": ".engine.arg_utils:AsyncEngineArgs",
    "EngineArgs": ".engine.arg_utils:EngineArgs",
    "AsyncLLMEngine": ".engine.async_llm_engine:AsyncLLMEngine",
    "LLMEngine": ".engine.llm_engine:LLMEngine",
    "LLM": ".entrypoints.llm:LLM",
    "initialize_ray_cluster": ".v1.executor.ray_utils:initialize_ray_cluster",
    "PromptType": ".inputs:PromptType",
    "TextPrompt": ".inputs:TextPrompt",
    "TokensPrompt": ".inputs:TokensPrompt",
    "ModelRegistry": ".model_executor.models:ModelRegistry",
    "SamplingParams": ".sampling_params:SamplingParams",
    "PoolingParams": ".pooling_params:PoolingParams",
    "ClassificationOutput": ".outputs:ClassificationOutput",
    "ClassificationRequestOutput": ".outputs:ClassificationRequestOutput",
    "CompletionOutput": ".outputs:CompletionOutput",
    "EmbeddingOutput": ".outputs:EmbeddingOutput",
    "EmbeddingRequestOutput": ".outputs:EmbeddingRequestOutput",
    "PoolingOutput": ".outputs:PoolingOutput",
    "PoolingRequestOutput": ".outputs:PoolingRequestOutput",
    "RequestOutput": ".outputs:RequestOutput",
    "ScoringOutput": ".outputs:ScoringOutput",
    "ScoringRequestOutput": ".outputs:ScoringRequestOutput",
}

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine
    from vllm.entrypoints.llm import LLM
    from vllm.inputs import PromptType, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry
    from vllm.outputs import (
        ClassificationOutput,
        ClassificationRequestOutput,
        CompletionOutput,
        EmbeddingOutput,
        EmbeddingRequestOutput,
        PoolingOutput,
        PoolingRequestOutput,
        RequestOutput,
        ScoringOutput,
        ScoringRequestOutput,
    )
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
    from vllm.v1.executor.ray_utils import initialize_ray_cluster
else:

    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(f"module {__package__} has no attribute {name}")


__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
