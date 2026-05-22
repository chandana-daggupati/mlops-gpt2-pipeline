import argparse
import logging
import os

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--role", required=True)
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--job_name", required=True)
    parser.add_argument("--instance_type", default="ml.m5.xlarge")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    return parser.parse_args()

def main():
    args = parse_args()

    session = sagemaker.Session()
    region = boto3.session.Session().region_name
    logger.info(f"AWS region: {region}")
    logger.info(f"Training job name: {args.job_name}")

    huggingface_estimator = HuggingFace(
        entry_point="train.py",
        source_dir="./src",
        instance_type=args.instance_type,
        instance_count=1,
        role=args.role,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        output_path=f"s3://{args.s3_bucket}/models/",
        base_job_name=args.job_name,
        hyperparameters={
            "epochs": args.epochs,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
        },
    )

    logger.info("Submitting training job to SageMaker...")
    huggingface_estimator.fit(wait=True)

    model_artifact = huggingface_estimator.model_data
    logger.info(f"Training complete! Model saved to: {model_artifact}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"model_artifact={model_artifact}\n")


if __name__ == "__main__":
    main()