"""LLM interface for local model support using HuggingFace transformers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed
)


@dataclass
class LLMOutput:
    """Output from LLM generation."""
    text: str
    tokens_used: int
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: Optional[List[float]] = None


class LLMInterface:
    """Interface for local LLM models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda:0",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        seed: Optional[int] = None
    ):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        if seed is not None:
            set_seed(seed)

        # Configure quantization
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
            )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_name,
            quantization_config=quantization_config,
            device_map=device if device != "cpu" else "auto",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )

        # Default generation parameters (sampling toggled per call)
        self.default_max_new_tokens = max_new_tokens
        self.default_temperature = temperature

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        return_logprobs: bool = False,
    ) -> LLMOutput:
        """Generate text from prompt."""
        start_time = time.time()

        # Apply chat template for instruct/chat models so Llama-3-8B-Instruct
        # responds as an assistant instead of auto-completing the raw prompt.
        # Without this, EM drops 10-30 points across all datasets.
        # Detect via name tag OR tokenizer's chat_template attribute (Qwen etc).
        mn = (getattr(self, "model_name", "") or "").lower()
        name_chat = any(tag in mn for tag in ("instruct", "-it", "chat", "qwen"))
        tok_chat = bool(getattr(self.tokenizer, "chat_template", None))
        is_chat = name_chat or tok_chat
        if is_chat and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                tmpl_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if "qwen" in mn:
                    tmpl_kwargs["enable_thinking"] = False
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **tmpl_kwargs,
                )
                prompt = templated
            except Exception:
                pass

        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        # Resolve generation parameters
        max_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        do_sample = temp is not None and temp > 0
        top_p_val = top_p if top_p is not None else 1.0

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            output_scores=return_logprobs,
            return_dict_in_generate=True,
        )

        # Only pass temperature when sampling is enabled
        if do_sample:
            generation_kwargs["temperature"] = temp
            generation_kwargs["top_p"] = top_p_val

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        # P0: Add guards for empty outputs
        import os
        debug = os.environ.get("TRIDENT_DEBUG_LLM_CERT", "0") == "1"

        if not hasattr(outputs, 'sequences') or outputs.sequences is None:
            error_msg = (
                f"Model generate returned no sequences!\n"
                f"  Prompt length: {len(prompt)}\n"
                f"  max_new_tokens: {max_tokens}"
            )
            if debug:
                print(f"[MODEL ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        if len(outputs.sequences) == 0:
            error_msg = (
                f"Model generated empty sequences list!\n"
                f"  Prompt length: {len(prompt)}"
            )
            if debug:
                print(f"[MODEL ERROR] {error_msg}")
            # Return empty output rather than crashing
            return LLMOutput(text="", tokens_used=input_length, latency_ms=(time.time() - start_time) * 1000)

        # Decode output
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # P0: Warn if empty
        if debug and not generated_text:
            print(f"[MODEL WARNING] Generated empty text")
            print(f"  Prompt length: {len(prompt)}")
            print(f"  Generated IDs length: {len(generated_ids)}")

        if stop:
            for s in stop:
                idx = generated_text.find(s)
                if idx != -1:
                    generated_text = generated_text[:idx]
                    break

        # Calculate token usage
        prompt_tokens = input_length
        completion_tokens = len(generated_ids)
        tokens_used = prompt_tokens + completion_tokens

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Extract logprobs if requested
        logprobs = None
        if return_logprobs and hasattr(outputs, 'scores'):
            logprobs = []
            for score in outputs.scores:
                probs = torch.softmax(score[0], dim=-1)
                logprobs.append(probs.max().item())

        return LLMOutput(
            text=generated_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            logprobs=logprobs
        )

    def build_rag_prompt(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        instructions: Optional[str] = None
    ) -> str:
        """Build a RAG prompt with retrieved passages."""
        if instructions is None:
            instructions = (
                "Answer the following question based on the provided context. "
                "If the answer cannot be found in the context, say 'I cannot answer based on the given context.'"
            )

        # Build context from passages
        context_parts = []
        for i, passage in enumerate(passages, 1):
            text = passage.get('text', passage.get('content', ''))
            context_parts.append(f"[{i}] {text}")

        context = "\n\n".join(context_parts)

        # Build full prompt
        prompt = f"""{instructions}

Context:
{context}

Question: {query}

Answer:"""

        return prompt

#     def build_multi_hop_prompt(
#         self,
#         query: str,
#         passages: List[Dict[str, Any]],
#         facets: List[Dict[str, Any]]
#     ) -> str:
#         """Build a prompt for multi-hop reasoning."""
#         # Build context
#         context_parts = []
#         for i, passage in enumerate(passages, 1):
#             text = passage.get('text', passage.get('content', ''))
#             source = passage.get('source', f'Document {i}')
#             context_parts.append(f"[{source}] {text}")

#         context = "\n\n".join(context_parts)

#         # Build reasoning requirements
#         requirements = []
#         for facet in facets:
#             facet_type = facet.get('type', 'UNKNOWN')
#             template = facet.get('template', {})
#             requirements.append(f"- {facet_type}: {template}")

#         requirements_text = "\n".join(requirements) if requirements else "None specified"

#         # Build full prompt
#         prompt = f"""You are answering a complex question that requires multi-hop reasoning.

# Context Documents:
# {context}

# Reasoning Requirements:
# {requirements_text}

# Question: {query}

# Please provide a step-by-step answer that addresses all reasoning requirements:

# Answer:"""

#         return prompt

    def build_multi_hop_prompt(
        self,
        question: str,
        passages: List[Dict[str, Any]],
        facets: List[Any],
        dataset: str = "multi-hop QA",
    ) -> str:
        """
        Build a multi-hop QA prompt with an explicit 'Final answer:' line
        so that downstream extraction is robust.

        Args:
            question: The question to answer
            passages: List of passage dictionaries with 'text' key
            facets: List of facet requirements
            dataset: Dataset name for prompt context (e.g., "HotpotQA", "MuSiQue", "2WikiMultiHop")
        """
        context_block = self.build_rag_prompt(question, passages)

        if facets:
            requirements = "\n".join(f"- {f}" for f in facets)
            requirements_block = f"Reasoning requirements:\n{requirements}\n\n"
        else:
            requirements_block = ""

        if str(dataset).lower() == "fever":
            # KILT FEVER only has SUPPORTS / REFUTES labels (no NOT ENOUGH INFO).
            # Forcing a binary choice prevents the chat-templated model from
            # hedging with NEI on most claims. You MUST pick one.
            prompt = (
                f"{context_block}\n"
                f"{requirements_block}"
                "You are verifying a factual claim. You MUST decide whether\n"
                "the context SUPPORTS or REFUTES the claim. Do not say the\n"
                "context is insufficient — make your best judgment.\n"
                "1. Use the information in the context above.\n"
                "2. Briefly reason in one or two short steps.\n"
                "3. On the LAST line, output exactly in the form:\n"
                "   Final answer: SUPPORTS\n"
                "   OR\n"
                "   Final answer: REFUTES\n"
                "4. Do NOT output any label other than SUPPORTS or REFUTES.\n"
                "5. Do NOT add anything after the 'Final answer:' line.\n\n"
                f"Claim: {question}\n"
                "Now reason briefly and then give the final answer (SUPPORTS or REFUTES).\n"
            )
        else:
            prompt = (
                f"{context_block}\n"
                f"{requirements_block}"
                f"You are answering a multi-hop question from a {dataset} dataset.\n"
                "1. Use ONLY the information in the context above.\n"
                "2. First, think briefly and, if helpful, reason in a few short steps.\n"
                "3. Then, on the LAST line of your response, output the answer in the form:\n"
                "   Final answer: <short answer>\n"
                "   - If the question is yes/no, use exactly 'yes' or 'no' as the short answer.\n"
                "   - Otherwise, use a short phrase or name, not a full sentence.\n"
                "4. Do NOT add anything after the 'Final answer:' line.\n"
                "If you truly cannot answer based on the context, use:\n"
                "   Final answer: I cannot answer based on the given context.\n\n"
                f"Question: {question}\n"
                "Now reason and then give the final answer.\n"
            )
        return prompt


    def extract_answer(self, generated_text: str) -> str:
        """
        Extract the answer from generated text.

        Prioritizes extracting from "Final answer:" format (used by build_multi_hop_prompt),
        then falls back to removing common prefixes.
        """
        answer = generated_text.strip()

        # First, try to extract from "Final answer:" format (case-insensitive)
        # This is the format requested by build_multi_hop_prompt
        import re

        # Look for "Final answer:" pattern (case-insensitive)
        match = re.search(r'(?:^|\n)\s*final\s+answer\s*:\s*(.+?)(?:\n|$)', answer, re.IGNORECASE | re.MULTILINE)
        if match:
            # Extract the answer portion after "Final answer:"
            answer = match.group(1).strip()
            # If there are multiple "Final answer:" lines, we already got the first one
            # Remove any trailing "Final answer:" artifacts
            answer = re.sub(r'\s*final\s+answer\s*:.*$', '', answer, flags=re.IGNORECASE).strip()
        else:
            # Fall back to removing common prefixes
            prefixes_to_remove = [
                "Answer:", "A:", "Response:", "The answer is:",
                "Based on the context,", "According to the documents,"
            ]

            for prefix in prefixes_to_remove:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
                    break

        # Take first sentence if answer is very long
        if len(answer) > 500 and '.' in answer:
            answer = answer.split('.')[0] + '.'

        return answer

    def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **kwargs
    ) -> List[LLMOutput]:
        """Generate responses for multiple prompts in batches."""
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = []

            for prompt in batch:
                output = self.generate(prompt, **kwargs)
                batch_results.append(output)

            results.extend(batch_results)

        return results

    def compute_token_cost(self, text: str) -> int:
        """Compute token cost for a text."""
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'temperature': self.temperature,
            'max_new_tokens': self.max_new_tokens,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }


class VLLMInterface:
    """Interface for vLLM backend (for high-throughput serving)."""

    def __init__(self, model_name: str, **kwargs):
        # P0-2: Store model_name for HF fallback
        self.model_name = model_name

        try:
            from vllm import LLM, SamplingParams

            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
                gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
                trust_remote_code=kwargs.get('trust_remote_code', True)
            )

            self.sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', 0.0),
                max_tokens=kwargs.get('max_new_tokens', 256),
                top_p=kwargs.get('top_p', 1.0)
            )

        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")

    def generate(self, prompt: str, **kwargs) -> LLMOutput:
        """Generate using vLLM."""
        import os
        start_time = time.time()

        # Update sampling params if provided
        temperature = kwargs.get('temperature', self.sampling_params.temperature)
        max_tokens = kwargs.get('max_new_tokens', self.sampling_params.max_tokens)
        top_p = kwargs.get('top_p', getattr(self.sampling_params, 'top_p', 1.0))
        stop = kwargs.get('stop', None)

        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        # P0: Add guards and detailed logging
        debug = os.environ.get("TRIDENT_DEBUG_LLM_CERT", "0") == "1"

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)

        # P0: Check if outputs is empty BEFORE accessing [0]
        if not outputs:
            error_msg = (
                f"vLLM returned empty outputs list!\n"
                f"  Prompt length: {len(prompt)}\n"
                f"  Prompt (first 100): {prompt[:100]}\n"
                f"  Sampling params: temp={temperature}, max_tokens={max_tokens}"
            )
            if debug:
                print(f"[VLLM ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        output = outputs[0]

        # P0: Check if output.outputs is empty BEFORE accessing [0]
        if not hasattr(output, 'outputs') or not output.outputs:
            error_msg = (
                f"vLLM output has no outputs!\n"
                f"  Output type: {type(output)}\n"
                f"  Has outputs attr: {hasattr(output, 'outputs')}\n"
                f"  Prompt length: {len(prompt)}"
            )
            if debug:
                print(f"[VLLM ERROR] {error_msg}")
            raise RuntimeError(error_msg)

        generated_text = output.outputs[0].text

        # P0: Check if generated text is None or empty
        if generated_text is None:
            error_msg = (
                f"vLLM generated None text!\n"
                f"  Output: {output}\n"
                f"  Prompt length: {len(prompt)}"
            )
            if debug:
                print(f"[VLLM ERROR] {error_msg}")
            # Return empty string instead of None to avoid crashes
            generated_text = ""

        tokens_used = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
        latency_ms = (time.time() - start_time) * 1000

        if debug and not generated_text:
            print(f"[VLLM WARNING] Generated empty text (length=0)")
            print(f"  Tokens used: {tokens_used}")
            print(f"  Latency: {latency_ms:.1f}ms")

        return LLMOutput(
            text=generated_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMOutput]:
        """Batch generation with vLLM."""
        start_time = time.time()

        from vllm import SamplingParams
        temperature = kwargs.get('temperature', self.sampling_params.temperature)
        max_tokens = kwargs.get('max_new_tokens', self.sampling_params.max_tokens)
        top_p = kwargs.get('top_p', getattr(self.sampling_params, 'top_p', 1.0))
        stop = kwargs.get('stop', None)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            text = output.outputs[0].text
            tokens = len(output.outputs[0].token_ids)
            results.append(LLMOutput(
                text=text,
                tokens_used=tokens,
                latency_ms=(time.time() - start_time) * 1000 / len(prompts)
            ))

        return results
