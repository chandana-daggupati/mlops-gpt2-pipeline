import json
import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    logger.info(f"Loading model from: {model_dir}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded on device: {device}")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        return json.loads(request_body)

    raise ValueError(
        f"Unsupported Content-Type: '{request_content_type}'. "
        "Send requests with Content-Type: application/json"
    )


def predict_fn(data, model_artifacts):
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]

    prompt = data.get("prompt", "")
    max_new_tokens = int(data.get("max_new_tokens", 100))
    temperature = float(data.get("temperature", 0.8))
    top_p = float(data.get("top_p", 0.9))

    if not prompt:
        return {"error": "No 'prompt' field in request body.",
                "generated_text": ""}

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens_generated": len(output_ids[0]) - len(inputs["input_ids"][0]),
    }


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
