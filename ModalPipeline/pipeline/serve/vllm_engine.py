from __future__ import annotations

import inspect
import time
from typing import Any

from pipeline.generate.grammar import build_guided_decoding_params


def build_llm_kwargs(model: dict[str, Any], cache_dir: str) -> dict[str, Any]:
    vllm_cfg = model.get("vllm", {})
    kwargs: dict[str, Any] = {
        "model": model["hf_id"],
        "download_dir": cache_dir,
        "trust_remote_code": True,
        "tensor_parallel_size": vllm_cfg.get("tensor_parallel_size", 1),
        "max_model_len": vllm_cfg.get("max_model_len", 32768),
        "gpu_memory_utilization": vllm_cfg.get("gpu_memory_utilization", 0.90),
        "enable_expert_parallel": vllm_cfg.get("enable_expert_parallel", True),
    }
    quant = vllm_cfg.get("quantization")
    if quant:
        kwargs["quantization"] = quant
    return kwargs


def _add_guided_json_sampling_param(
    sampling_kwargs: dict[str, Any],
    sampling_params_cls: type,
    guided_json: dict[str, Any],
) -> None:
    params = inspect.signature(sampling_params_cls).parameters

    if "structured_outputs" in params:
        from vllm.sampling_params import StructuredOutputsParams

        sampling_kwargs["structured_outputs"] = StructuredOutputsParams(json=guided_json)
        return

    if "guided_decoding" in params:
        from vllm.sampling_params import GuidedDecodingParams

        try:
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(json=guided_json)
        except TypeError:
            sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
                json_schema=guided_json
            )
        return

    if "guided_json" in params:
        sampling_kwargs["guided_json"] = guided_json
        return

    raise RuntimeError(
        "This vLLM SamplingParams version does not expose structured_outputs, "
        "guided_json, or guided_decoding, so xgrammar_on cannot be used with "
        "this image."
    )


class VllmGenerator:
    def __init__(self, model: dict[str, Any], cache_dir: str) -> None:
        from vllm import LLM

        self.model = model
        self.llm = LLM(**build_llm_kwargs(model, cache_dir))

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        output_profile: str,
        xgrammar_enabled: bool,
    ) -> dict[str, Any]:
        from vllm import SamplingParams

        prompt = self._format_chat(messages)
        guided = build_guided_decoding_params(output_profile, xgrammar_enabled)

        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }

        if guided and guided.get("guided_json"):
            _add_guided_json_sampling_param(
                sampling_kwargs,
                SamplingParams,
                guided["guided_json"],
            )

        params = SamplingParams(**sampling_kwargs)
        t0 = time.perf_counter()
        outputs = self.llm.generate([prompt], params)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        out = outputs[0].outputs[0]
        text = out.text
        completion_tokens = len(out.token_ids)
        prompt_tokens = len(outputs[0].prompt_token_ids)

        tps = (
            completion_tokens / (elapsed_ms / 1000)
            if elapsed_ms > 0
            else 0.0
        )

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_latency_ms": elapsed_ms,
            "tokens_per_second": tps,
        }

    def _format_chat(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}")
            else:
                parts.append(f"<|im_start|>{role}\n{content}")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
