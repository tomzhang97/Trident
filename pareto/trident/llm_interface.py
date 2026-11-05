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
    GenerationConfig,
    set_seed
)


@dataclass
class LLMOutput:
    """Output from LLM generation."""
    text: str
    tokens_used: int
    latency_ms: float
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

        # Generation config
        self.generation_config = GenerationConfig(
            temperature=temperature if temperature > 0 else 1e-7,
            do_sample=temperature > 0,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logprobs: bool = False
    ) -> LLMOutput:
        """Generate text from prompt."""
        start_time = time.time()

        # Apply chat template for instruct/chat models so Llama-3-8B-Instruct
        # responds as an assistant instead of auto-completing the raw prompt.
        # Detect via (a) explicit name tags or (b) tokenizer carrying a
        # chat_template attribute (covers Qwen, Mistral, etc.).
        mn = (self.model_name or "").lower()
        name_chat = any(tag in mn for tag in ("instruct", "-it", "chat", "qwen"))
        tok_chat = bool(getattr(self.tokenizer, "chat_template", None))
        is_chat = name_chat or tok_chat
        if is_chat and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Qwen3 wraps answers in <think>...</think> by default;
                # disable to get a direct answer the existing extractor can read.
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

        # Update generation config if needed
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens
        if temperature is not None:
            gen_config.temperature = temperature if temperature > 0 else 1e-7
            gen_config.do_sample = temperature > 0

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                output_scores=return_logprobs,
                return_dict_in_generate=True
            )

        # Decode output
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Calculate tokens used
        tokens_used = len(generated_ids) + input_length

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
        """
        context_block = self.build_rag_prompt(question, passages)

        if facets:
            requirements = "\n".join(f"- {f}" for f in facets)
            requirements_block = f"Reasoning requirements:\n{requirements}\n\n"
        else:
            requirements_block = ""

        if str(dataset).lower() == "fever":
            # KILT FEVER only has SUPPORTS / REFUTES labels (no NEI).
            # Force a binary choice — the chat-templated model otherwise
            # hedges with NEI for most claims.
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
        start_time = time.time()

        # Update sampling params if provided
        sampling_params = self.sampling_params
        if 'temperature' in kwargs:
            sampling_params.temperature = kwargs['temperature']
        if 'max_new_tokens' in kwargs:
            sampling_params.max_tokens = kwargs['max_new_tokens']

        # Generate
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0]

        generated_text = output.outputs[0].text
        tokens_used = len(output.outputs[0].token_ids)
        latency_ms = (time.time() - start_time) * 1000

        return LLMOutput(
            text=generated_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms
        )

    def batch_generate(self, prompts: List[str], **kwargs) -> List[LLMOutput]:
        """Batch generation with vLLM."""
        start_time = time.time()

        outputs = self.llm.generate(prompts, self.sampling_params)

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
