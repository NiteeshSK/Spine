"""Batched generation, shared by the eval harness and the DPO candidate sampler.

Two backends:

* ``hf``   -- transformers + optional PEFT adapter. Works everywhere, including
              CPU with a tiny model for the smoke test.
* ``vllm`` -- ~10x faster for the 800-example test set. Optional import; the
              module must stay importable on a machine with no CUDA.

Both expose the same ``generate(prompts, ...) -> list[str]`` surface so the
eval harness never branches on backend.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class GenConfig:
    max_new_tokens: int = 320
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    seed: int = 0
    batch_size: int = 8
    stop: list[str] = field(default_factory=list)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class HFBackend:
    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        *,
        load_in_4bit: bool = True,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        merge_adapter: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for correct batched decode

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        has_cuda = torch.cuda.is_available()
        if load_in_4bit and has_cuda:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = device_map
        else:
            kwargs["torch_dtype"] = getattr(torch, dtype) if has_cuda else torch.float32
            kwargs["device_map"] = device_map if has_cuda else None

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            if merge_adapter and not load_in_4bit:
                self.model = self.model.merge_and_unload()
        self.model.eval()
        if not has_cuda:
            self.model.to("cpu")

    def generate(self, chats: Sequence[list[dict]], cfg: GenConfig) -> list[list[str]]:
        torch = self.torch
        outputs: list[list[str]] = []
        for start in range(0, len(chats), cfg.batch_size):
            batch = chats[start : start + cfg.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in batch
            ]
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": cfg.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "num_return_sequences": cfg.n,
            }
            if cfg.greedy:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs.update(do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            trimmed = out[:, enc["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)
            for i in range(len(batch)):
                outputs.append([d.strip() for d in decoded[i * cfg.n : (i + 1) * cfg.n]])
        return outputs

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class VLLMBackend:
    def __init__(self, model_id: str, adapter: str | None = None, *, max_model_len: int = 4096, gpu_memory_utilization: float = 0.90):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.adapter = adapter
        self.llm = LLM(
            model=model_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter is not None,
            max_lora_rank=64,
            trust_remote_code=True,
        )

    def generate(self, chats: Sequence[list[dict]], cfg: GenConfig) -> list[list[str]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=cfg.n,
            temperature=0.0 if cfg.greedy else cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
            stop=cfg.stop or None,
        )
        prompts = [
            self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chats
        ]
        lora = None
        if self.adapter:
            from vllm.lora.request import LoRARequest

            lora = LoRARequest("spine", 1, self.adapter)
        results = self.llm.generate(prompts, params, lora_request=lora)
        return [[o.text.strip() for o in r.outputs] for r in results]

    def close(self) -> None:  # pragma: no cover
        pass


def build_backend(backend: str, model_id: str, adapter: str | None = None, **kwargs):
    if backend == "vllm":
        return VLLMBackend(model_id, adapter, **{k: v for k, v in kwargs.items() if k in ("max_model_len", "gpu_memory_utilization")})
    return HFBackend(model_id, adapter, **{k: v for k, v in kwargs.items() if k in ("load_in_4bit", "dtype", "device_map", "merge_adapter")})
