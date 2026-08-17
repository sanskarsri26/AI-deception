from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


EXPECTED_RE = re.compile(
    r"^(C-\d{3}):",
    re.MULTILINE,
)

ATTEMPT_RE = re.compile(
    r"^\s*(C-\d{3})\s*\|\s*\S[^\n]*$",
    re.I | re.M,
)

STOP_ATTEMPT_RE = re.compile(
    r"^\s*(C-\d{3})\s*\|\s*\S[^\n]*\n",
    re.I | re.M,
)


def expected_ids(prompt):
    marker = "TASK TO COMPLETE"

    if marker in prompt:
        prompt = prompt.split(
            marker,
            1,
        )[1]

    return {
        x.upper()
        for x in EXPECTED_RE.findall(
            prompt
        )
    }


def attempted_ids(text):
    return {
        x.upper()
        for x in ATTEMPT_RE.findall(text)
    }


def completed_attempt_ids_for_stopping(text):
    return {
        x.upper()
        for x in STOP_ATTEMPT_RE.findall(text)
    }


class StopWhenAllClaimsAttempted(
    StoppingCriteria
):
    def __init__(
        self,
        tokenizer,
        prompt_length,
        expected,
    ):
        self.tokenizer = tokenizer
        self.prompt_length = (
            prompt_length
        )
        self.expected = expected

    def __call__(
        self,
        input_ids,
        scores,
        **kwargs,
    ):
        generated = input_ids[
            0,
            self.prompt_length:
        ]

        text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        seen = completed_attempt_ids_for_stopping(
            text
        )

        return self.expected.issubset(
            seen
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--prompt",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--metadata-output",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--stop-after-claims",
        action="store_true",
    )

    args = parser.parse_args()

    prompt = Path(
        args.prompt
    ).read_text(
        encoding="utf-8"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.model
        )
    )

    use_cuda = (
        args.device == "cuda"
        or (
            args.device == "auto"
            and torch.cuda.is_available()
        )
    )

    if use_cuda:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        )
    else:
        model = (
            AutoModelForCausalLM
            .from_pretrained(
                args.model,
                torch_dtype=torch.float32,
            )
        )

        model.to("cpu")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        k: v.to(model_device)
        for k, v in inputs.items()
    }

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    expected = expected_ids(
        prompt
    )

    criteria = None

    if args.stop_after_claims:
        criteria = StoppingCriteriaList([
            StopWhenAllClaimsAttempted(
                tokenizer=tokenizer,
                prompt_length=(
                    prompt_length
                ),
                expected=expected,
            )
        ])

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=(
                args.max_new_tokens
            ),
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
            ),
            stopping_criteria=criteria,
        )

    generated_ids = output_ids[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    Path(args.output).write_text(
        response,
        encoding="utf-8",
    )

    attempted = attempted_ids(
        response
    )

    generated_tokens = len(
        generated_ids
    )

    all_seen = expected.issubset(
        attempted
    )

    hit_limit = (
        generated_tokens
        >= args.max_new_tokens - 1
        and not all_seen
    )

    metadata = {
        "model": args.model,
        "device": (
            "cuda"
            if use_cuda
            else "cpu"
        ),
        "prompt_tokens": (
            prompt_length
        ),
        "generated_tokens": (
            generated_tokens
        ),
        "expected_claim_count": (
            len(expected)
        ),
        "attempted_claim_count": (
            len(
                expected.intersection(
                    attempted
                )
            )
        ),
        "all_claims_seen": (
            all_seen
        ),
        "hit_generation_limit": (
            hit_limit
        ),
    }

    Path(
        args.metadata_output
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
