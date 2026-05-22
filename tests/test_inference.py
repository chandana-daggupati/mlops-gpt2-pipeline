import json
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestInputFn:
    def test_parses_valid_json(self):
        from inference import input_fn

        payload = json.dumps(
            {"prompt": "To be or not to be", "max_new_tokens": 50})
        result = input_fn(payload, "application/json")

        assert result["prompt"] == "To be or not to be"
        assert result["max_new_tokens"] == 50

    def test_raises_on_unsupported_content_type(self):
        from inference import input_fn

        with pytest.raises(ValueError, match="Unsupported Content-Type"):
            input_fn("plain text here", "text/plain")


class TestOutputFn:
    def test_returns_json_string_and_content_type(self):
        from inference import output_fn

        prediction = {
            "generated_text": "To be or not to be, that is the question."}
        body, content_type = output_fn(prediction, "application/json")

        assert content_type == "application/json"
        parsed = json.loads(body)
        assert parsed["generated_text"] == prediction["generated_text"]

    def test_output_is_valid_json(self):
        from inference import output_fn

        prediction = {
            "prompt": "hi",
            "generated_text": "hi there!",
            "tokens_generated": 5}
        body, _ = output_fn(prediction, "application/json")

        parsed = json.loads(body)
        assert "generated_text" in parsed


@pytest.fixture(scope="module")
def model_artifacts():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    return {"model": model, "tokenizer": tokenizer, "device": "cpu"}


class TestPredictFn:
    def test_generates_text_from_prompt(self, model_artifacts):
        from inference import predict_fn

        data = {
            "prompt": "The history of artificial intelligence",
            "max_new_tokens": 30}
        result = predict_fn(data, model_artifacts)

        assert "generated_text" in result
        assert len(result["generated_text"]) > len(data["prompt"])

    def test_empty_prompt_returns_error(self, model_artifacts):
        from inference import predict_fn

        data = {"prompt": ""}
        result = predict_fn(data, model_artifacts)

        assert "error" in result
