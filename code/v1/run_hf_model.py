from __future__ import annotations

from benchmark_parsing import (
    expected_ids_from_prompt,
    allowed_labels_from_prompt,
    valid_answer_ids,
)
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


from benchmark_parsing import expected_ids_from_prompt, allowed_labels_from_prompt, valid_answer_ids


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        dtype = torch.float32 if device == "cpu" else (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(device)
    model.eval()
    return tokenizer, model


def prepare_inputs(tokenizer, prompt: str, model_id: str):
    if "Instruct" in model_id:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt")


class AllValidClassificationsSeen(StoppingCriteria):
    def __init__(self, tokenizer, input_length: int, expected_ids: set[str], allowed_labels: set[str]):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.expected_ids = expected_ids
        self.allowed_labels = allowed_labels

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.input_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        seen = valid_answer_ids(text, self.allowed_labels)
        return bool(self.expected_ids) and self.expected_ids.issubset(seen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument(
        "--stop-after-classifications",
        action="store_true",
        help="Capability calibration only: stop once every requested T-ID has an exact allowed label.",
    )
    args = parser.parse_args()

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    tokenizer, model = load_model(args.model, args.device)
    inputs = prepare_inputs(tokenizer, prompt, args.model)
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[-1])

    expected_ids = expected_ids_from_prompt(prompt)
    allowed_labels = allowed_labels_from_prompt(prompt)
    stopping = None
    if args.stop_after_classifications:
        stopping = StoppingCriteriaList([
            AllValidClassificationsSeen(tokenizer, input_tokens, expected_ids, allowed_labels)
        ])

    started = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    elapsed = time.time() - started

    generated = outputs[0][input_tokens:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    output_tokens = int(generated.shape[-1])
    Path(args.output).write_text(response.rstrip() + "\n", encoding="utf-8")

    seen_ids = valid_answer_ids(response, allowed_labels)
    all_classifications_seen = bool(expected_ids) and expected_ids.issubset(seen_ids)
    hit_limit = output_tokens >= args.max_new_tokens and not all_classifications_seen

    metadata = {
        "model": args.model,
        "device_requested": args.device,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "max_new_tokens": args.max_new_tokens,
        "hit_generation_limit": hit_limit,
        "valid_classification_count": len(seen_ids & expected_ids),
        "expected_classification_count": len(expected_ids),
        "all_classifications_seen": all_classifications_seen,
        "stop_after_classifications": args.stop_after_classifications,
        "elapsed_seconds": elapsed,
        "do_sample": False,
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(args.output).with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(response)
    print("\n--- RUN METADATA ---")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
