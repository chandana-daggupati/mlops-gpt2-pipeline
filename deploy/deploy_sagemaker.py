import argparse
import logging
import os

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--role", required=True)
    parser.add_argument("--model_artifact", required=True)
    parser.add_argument("--endpoint_name", default="mlops-gpt2-endpoint")
    parser.add_argument("--instance_type", default="ml.m5.large")
    parser.add_argument("--run_number", default="0")

    return parser.parse_args()

def endpoint_exists(sm_client, endpoint_name):
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm_client.exceptions.ClientError:
        return False
    
def main():
    args = parse_args()

    sm_client = boto3.client("sagemaker")
    session = sagemaker.Session()
    region = boto3.session.Session().region_name
    logger.info(f"Model artifact: {args.model_artifact}")
    logger.info(f"Endpoint name: {args.endpoint_name}")

    model_name = f"mlops-gpt2-model-{args.run_number}"

    huggingface_model = HuggingFaceModel(
        model_data=args.model_artifact,
        role=args.role,
        transformers_version="4.37",
        pytorch_version="2.1",
        py_version="py310",
        name=model_name,
        entry_point="inference.py",
        source_dir="./src",
        sagemaker_session=session,
    )

    if endpoint_exists(sm_client, args.endpoint_name):
        logger.info(f"Endpoint exists — updating...")
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
            update_endpoint=True,
        )
    else:
        logger.info(f"Creating new endpoint: {args.endpoint_name}")
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type=args.instance_type,
            endpoint_name=args.endpoint_name,
        )

    logger.info(f"Endpoint '{args.endpoint_name}' is live!")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"endpoint_name={args.endpoint_name}\n")


if __name__ == "__main__":
    main()