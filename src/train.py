import argparse
import logging
import os

from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get(
            "SM_MODEL_DIR",
            "./model_output"))
    parser.add_argument(
        "--output_data_dir",
        type=str,
        default=os.environ.get(
            "SM_OUTPUT_DATA_DIR",
            "./output"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # 1. Load tokenizer and model
    logger.info("Loading GPT-2 tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Model parameters: {model.num_parameters():,}")

    # 2. Load dataset
    logger.info("Loading wikitext-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Filter out blank lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 10)

    logger.info(f"Train examples: {len(dataset['train']):,}")
    logger.info(f"Validation examples: {len(dataset['validation']):,}")

    # 3. Tokenize
    logger.info(f"Tokenizing (max_length={args.max_length})...")
    tokenized = dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

    # 4. Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=200,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=os.path.join(args.output_data_dir, "logs"),
        logging_steps=100,
        report_to="none",
    )

    # 6. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    # 7. Save
    logger.info(f"Saving model to {args.model_dir}")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
