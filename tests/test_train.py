import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["train.py"])
        from train import parse_args

        args = parse_args()

        assert args.epochs == 3
        assert args.learning_rate == 5e-5
        assert args.max_length == 128
        assert args.seed == 42

    def test_override_epochs(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["train.py", "--epochs", "1"])
        from train import parse_args

        args = parse_args()
        assert args.epochs == 1

class TestTokenizeFunction:

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import GPT2Tokenizer

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        return tok

    def test_truncates_long_text(self, tokenizer):
        from train import tokenize_function

        long_text = "hello world " * 200
        result = tokenize_function({"text": [long_text]}, tokenizer, max_length=32)

        assert len(result["input_ids"][0]) == 32

    def test_pads_short_text(self, tokenizer):
        from train import tokenize_function

        result = tokenize_function({"text": ["Hi"]}, tokenizer, max_length=64)

        assert len(result["input_ids"][0]) == 64

    def test_returns_attention_mask(self, tokenizer):
        from train import tokenize_function

        result = tokenize_function({"text": ["Hello there"]}, tokenizer, max_length=16)

        assert "attention_mask" in result

