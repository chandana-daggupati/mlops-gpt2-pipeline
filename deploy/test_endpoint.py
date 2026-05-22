import argparse
import json
import logging
import sys
import time

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    {
        "prompt": "The history of artificial intelligence began",
        "max_new_tokens": 80,
        "temperature": 0.8,
    },
    {
        "prompt": "In the year 2045, humans and machines",
        "max_new_tokens": 80,
        "temperature": 0.9,
    },
    {
        "prompt": "Scientists recently discovered that",
        "max_new_tokens": 60,
        "temperature": 0.7,
    },
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_name", default="mlops-gpt2-endpoint")
    parser.add_argument("--region", default=None)
    return parser.parse_args()


def invoke_endpoint(runtime_client, endpoint_name, payload):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

def main():
    args = parse_args()

    runtime = boto3.client("sagemaker-runtime", region_name=args.region)

    logger.info(f"Testing endpoint: {args.endpoint_name}")
    logger.info("-" * 60)

    all_passed = True

    for i, test_case in enumerate(TEST_PROMPTS, start=1):
        prompt = test_case["prompt"]
        logger.info(f"\nTest {i}/{len(TEST_PROMPTS)}")
        logger.info(f"Prompt: '{prompt}'")

        start = time.time()
        try:
            result = invoke_endpoint(runtime, args.endpoint_name, test_case)
            elapsed = time.time() - start

            generated = result.get("generated_text", "")
            tokens = result.get("tokens_generated", "?")

            logger.info(f"Generated ({tokens} new tokens, {elapsed:.1f}s):")
            logger.info(f"'{generated}'")

            assert "generated_text" in result
            assert len(generated) > len(prompt)
            logger.info("PASS")

        except Exception as e:
            logger.error(f"FAIL: {e}")
            all_passed = False

    logger.info("-" * 60)
    if all_passed:
        logger.info("All smoke tests PASSED! Endpoint is healthy.")
        sys.exit(0)
    else:
        logger.error("One or more smoke tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()