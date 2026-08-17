from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from v22.generate_task import CHECKS
from v22.parsing import substantively_completed_case_ids


class StopWhenAllCasesSubstantive(StoppingCriteria):
    """Stop only after every check and FINAL have nonempty values.

    End-of-text is accepted, so a newline after the last answer is not required.
    An empty final field such as ``OWNER=`` is never substantive.
    """

    def __init__(self, tokenizer, prompt_length: int, expected_ids: set[str]):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.expected_ids = expected_ids

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        seen = substantively_completed_case_ids(text, self.expected_ids, CHECKS)
        return bool(self.expected_ids) and self.expected_ids.issubset(seen)


class HFGenerator:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        lower_id = model_id.lower()
        self.use_chat_template = (
            "instruct" in lower_id or lower_id.endswith("-it")
        ) and bool(getattr(self.tokenizer, "chat_template", None))
        use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
        if use_cuda:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int, expected_ids: set[str] | None = None) -> tuple[str, dict]:
        rendered_prompt = prompt
        if self.use_chat_template:
            rendered_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        stopping = None
        if expected_ids:
            stopping = StoppingCriteriaList([
                StopWhenAllCasesSubstantive(self.tokenizer, prompt_length, expected_ids)
            ])
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=stopping,
            )
        generated = output[0, prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        count = int(generated.shape[0])
        return text, {
            "model": self.model_id,
            "used_chat_template": self.use_chat_template,
            "prompt_tokens": int(prompt_length),
            "generated_tokens": count,
            "max_new_tokens": max_new_tokens,
            "reached_generation_limit": count >= max_new_tokens,
            "elapsed_seconds": time.time() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--expected-cases", type=int)
    args = parser.parse_args()
    expected = {f"C-{i:03d}" for i in range(1, args.expected_cases + 1)} if args.expected_cases else None
    runner = HFGenerator(args.model, args.device)
    response, metadata = runner.generate(
        Path(args.prompt).read_text(encoding="utf-8"), args.max_new_tokens, expected
    )
    Path(args.output).write_text(response.rstrip() + "\n", encoding="utf-8")
    Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
