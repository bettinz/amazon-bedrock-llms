import json
import logging
import difflib
from typing import Dict, List, Any, Optional
import botocore.exceptions
import argparse
import re

from pydantic import BaseModel, field_validator
from copy import deepcopy


class PricePerUnit(BaseModel):
    currency: str
    value: float


class PriceDimensions(BaseModel):
    unit: str
    description: str
    pricePerUnit: PricePerUnit

    @field_validator("pricePerUnit", mode="before")
    def parse_price(cls, v):
        currency, value = next(iter(v.items()))
        return {"currency": currency, "value": float(value)}


class OnDemand(BaseModel):
    priceDimensions: PriceDimensions


class Product(BaseModel):
    inferenceType: str
    usagetype: str
    model: Optional[str] = None
    provider: Optional[str] = None
    OnDemand: OnDemand


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
DEFAULT_REGION = "us-east-1"

UNIT_MAPPING = {
    "1K tokens": 1000,
    "1M tokens": 1000000,
}

_GENERIC_TOKENS = {
    "amazon",
    "anthropic",
    "mistral",
    "meta",
    "cohere",
    "ai21",
    "stability",
    "ai",
    "model",
    "foundation",
    "inference",
    "profile",
    "global",
    "us",
    "eu",
    "apac",
    "v",
    "text",
}

_FAMILY_TOKENS = {
    "claude",
    "nova",
    "titan",
    "mistral",
    "mixtral",
    "llama",
    "command",
    "jurassic",
    "jamba",
    "embed",
}


def _normalize_identifier(value: str) -> str:
    lowered = value.lower().strip()
    lowered = lowered.split("/")[-1]
    lowered = lowered.replace("arn:aws:bedrock:", " ")
    lowered = lowered.replace("foundation-model", " ")
    lowered = lowered.replace("inference-profile", " ")
    lowered = lowered.replace("anthropic.claude", "anthropic claude")
    lowered = lowered.replace("amazon.nova", "amazon nova")
    lowered = lowered.replace("amazon.titan", "amazon titan")
    lowered = lowered.replace("mistral.", "mistral ")
    lowered = lowered.replace("meta.", "meta ")
    lowered = lowered.replace("cohere.", "cohere ")
    lowered = lowered.replace("ai21.", "ai21 ")
    lowered = lowered.replace(":", " ")
    lowered = lowered.replace("-", " ")
    lowered = lowered.replace("_", " ")
    lowered = lowered.replace(".", " ")
    lowered = lowered.replace("(", " ")
    lowered = lowered.replace(")", " ")
    lowered = lowered.replace(",", " ")
    lowered = re.sub(r"\b(v\d+)\b", " ", lowered)
    lowered = re.sub(r"\b0\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _tokenize_identifier(value: str) -> list[str]:
    normalized = _normalize_identifier(value)
    tokens: set[str] = set()
    for token in normalized.split():
        if not token:
            continue
        tokens.add(token)
        if token.isdigit() and len(token) == 4:
            tokens.add(token[:2])
            tokens.add(token[2:])
    return sorted(tokens)


def _extract_provider(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in {"amazon", "anthropic", "mistral", "meta", "cohere", "ai21", "stability"}:
            return token
    return None


def _score_model_name_candidate(model_id: str, candidate: str) -> tuple[float, str]:
    source_tokens = _tokenize_identifier(model_id)
    candidate_tokens = _tokenize_identifier(candidate)
    source_set = set(source_tokens)
    candidate_set = set(candidate_tokens)

    if not source_set or not candidate_set:
        return 0.0, "empty"

    source_provider = _extract_provider(source_tokens)
    candidate_provider = _extract_provider(candidate_tokens)
    score = 0.0

    if source_provider and candidate_provider:
        if source_provider == candidate_provider:
            score += 6.0
        else:
            score -= 2.0

    family_overlap = (source_set & candidate_set) & _FAMILY_TOKENS
    score += 5.0 * len(family_overlap)

    meaningful_source = source_set - _GENERIC_TOKENS
    meaningful_candidate = candidate_set - _GENERIC_TOKENS
    overlap = meaningful_source & meaningful_candidate
    score += 2.0 * len(overlap)

    normalized_source = _normalize_identifier(model_id)
    normalized_candidate = _normalize_identifier(candidate)
    score += difflib.SequenceMatcher(None, normalized_source, normalized_candidate).ratio() * 5.0

    if normalized_candidate in normalized_source or normalized_source in normalized_candidate:
        score += 4.0

    explanation = f"provider={source_provider}/{candidate_provider} overlap={sorted(overlap)} family={sorted(family_overlap)}"
    return score, explanation


def _resolve_model_name_locally(model_names: List[str], model_id: str) -> str:
    best_name = ""
    best_score = float("-inf")
    best_reason = ""

    for candidate in model_names:
        score, reason = _score_model_name_candidate(model_id, candidate)
        if score > best_score:
            best_score = score
            best_name = candidate
            best_reason = reason

    if not best_name:
        logger.error(f"Could not resolve pricing model name for {model_id}: empty candidate list")
        return "Error"

    logger.info(
        f"Resolved pricing model name locally for {model_id}: {best_name} (score={best_score:.2f}; {best_reason})"
    )
    return best_name if best_score > 1.5 else "Error"


def get_flattened_product(original_data):
    product = original_data.get("product", {})
    attributes = product.get("attributes", {})
    on_demand_terms = original_data.get("terms", {}).get("OnDemand", {})

    first_on_demand = next(iter(on_demand_terms.values()), {})
    price_dimensions = next(
        iter(first_on_demand.get("priceDimensions", {}).values()), {}
    )

    model = attributes.get("model") or attributes.get("titanModel")
    provider = attributes.get("provider")

    amazon_models = {
        "Nova Canvas",
        "Nova Pro",
        "Nova Lite",
        "Nova Micro",
        "Nova Ultra",
        "Nova Reel",
    }

    if attributes.get("titanModel") or (any([name in model for name in amazon_models])):
        provider = "Amazon"

    flattened_data = {
        "inferenceType": attributes.get("inferenceType"),
        "usagetype": attributes.get("usagetype"),
        "model": model,
        "provider": provider,
        "OnDemand": {
            "priceDimensions": price_dimensions,
        },
    }

    return Product(**flattened_data)


def fetch_aws_pricing(client) -> List[Product]:
    """
    Fetch AWS Bedrock pricing for a specific provider, handling pagination.

    Parameters:
        client (boto3.client): AWS Pricing client.
        provider_name (str): The provider name to filter (e.g., "anthropic", "mistral").

    Returns:
        dict: A dictionary containing the filtered PriceList.
    """
    all_price_list = []
    next_token = None

    try:
        while True:
            request_params = {
                "ServiceCode": "AmazonBedrock",
                "Filters": [
                    {
                        "Type": "TERM_MATCH",
                        "Field": "regionCode",
                        "Value": client.meta.region_name,
                    },
                    {
                        "Type": "TERM_MATCH",
                        "Field": "feature",
                        "Value": "On-demand Inference",
                    },
                ],
                "MaxResults": 100,
            }

            if next_token:
                request_params["NextToken"] = next_token

            response = client.get_products(**request_params)

            price_list = response.get("PriceList", [])
            all_price_list.extend(
                get_flattened_product(json.loads(price)) for price in price_list
            )

            next_token = response.get("NextToken")

            if not next_token:
                break

        return all_price_list

    except Exception as e:
        logger.error(f"Error fetching AWS service pricing: {e}")
        return []


def filter_pricing_by_model(
    pricing_data: List[Product], model_name: str
) -> List[Product]:
    filtered_products = [
        product for product in pricing_data if product.model.startswith(model_name)
    ]
    output_products = []

    for product in filtered_products:
        output_products.append(product)

        if (
            product.provider == "Anthropic"
            and "input-tokens" in product.usagetype.lower()
        ):
            output_product = deepcopy(product)
            output_product.usagetype = product.usagetype.replace(
                "input-tokens", "output-tokens"
            )
            output_product.inferenceType = product.inferenceType.replace(
                "Input", "Output"
            )

            original_price = product.OnDemand.priceDimensions.pricePerUnit.value

            if product.model.startswith("Claude 3"):
                new_price = original_price * 0.5
            elif product.model.startswith("Claude 2") or product.model.startswith(
                "Claude Instant"
            ):
                new_price = original_price * 3
            else:
                new_price = original_price

            output_product.OnDemand.priceDimensions.pricePerUnit.value = new_price

            output_product.OnDemand.priceDimensions.description = (
                product.OnDemand.priceDimensions.description.replace(
                    f"{original_price}", f"{new_price}"
                ).replace("input-tokens", "output-tokens")
            )

            output_products.append(output_product)

    return output_products


def get_model_names(pricing_data: List[Product]) -> List[str]:
    model_names = set()

    for price in pricing_data:
        try:
            model_name = price.model
            if model_name:
                model_names.add(model_name)
        except KeyError:
            continue

    return sorted(model_names)


def parse_pricing_with_model(model_names: List[str], model_id: str, client: Any) -> str:
    try:
        return _resolve_model_name_locally(model_names, model_id)

    except botocore.exceptions.ClientError as e:
        logger.error(f"Error fetching model name for {model_id}: {e}")
        return "Error"

    except Exception as e:
        logger.error(f"Unexpected error in parse_pricing_with_model: {e}")
        return "Error"


def extract_model_pricing(
    filtered_pricing: List[Product], model_id: str
) -> Dict[str, Any]:
    """
    Extracts pricing details for the given model from the filtered pricing data.

    Args:
        filtered_pricing (List[Product]): Filtered list of pricing data for the model.
        model_id (str): The AWS Bedrock model ID.

    Returns:
        Dict[str, Any]: Pricing details for input, output, and cache read input tokens.
    """
    pricing_details = {"input": {}, "output": {}, "cache_read_input": {}}

    found_types = set()

    for product in filtered_pricing:
        inference_type = product.inferenceType.lower()
        found_types.add(inference_type)

        price_dimension = product.OnDemand.priceDimensions
        price_per_unit = price_dimension.pricePerUnit.value
        unit = price_dimension.unit
        currency = price_dimension.pricePerUnit.currency

        if "input tokens" in inference_type and "cache read" not in inference_type:
            pricing_details["input"] = {
                "price": price_per_unit,
                "unit": UNIT_MAPPING.get(unit, unit),
                "currency": currency,
            }
        elif "output tokens" in inference_type:
            pricing_details["output"] = {
                "price": price_per_unit,
                "unit": UNIT_MAPPING.get(unit, unit),
                "currency": currency,
            }
        elif (
            "cache read input tokens" in inference_type
            or "prompt cache read input tokens" in inference_type
        ):
            pricing_details["cache_read_input"] = {
                "price": price_per_unit,
                "unit": UNIT_MAPPING.get(unit, unit),
                "currency": currency,
            }

    expected_types = {"input tokens", "output tokens", "cache read input tokens"}
    missing_types = expected_types - found_types

    if missing_types:
        logger.warning(f"Missing inference types for model {model_id}: {missing_types}")

    return {model_id: pricing_details}


def main(model_id: str, region: str):
    """
    Main function to retrieve and parse AWS Bedrock pricing information.

    Args:
        model (str): The model ID to use.
        region (str): The AWS region to use.
    """
    import boto3

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        pricing = boto3.client("pricing", region_name=region)

        pricing_data = fetch_aws_pricing(pricing)

        model_names = get_model_names(pricing_data)

        model_name = parse_pricing_with_model(model_names, model_id, bedrock)

        if model_name == "Error":
            logger.error("Failed to determine model name. Exiting.")
            return

        filtered_pricing = filter_pricing_by_model(pricing_data, model_name)

        model_pricing = {
            model_id: {"input": None, "output": None, "cache_read_input": None}
        }
        if filtered_pricing:
            model_pricing = extract_model_pricing(filtered_pricing, model_id)

        print(json.dumps(model_pricing, indent=4))
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve and parse AWS Bedrock pricing information."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="The model ID to use"
    )
    parser.add_argument(
        "--region", type=str, default=DEFAULT_REGION, help="The AWS region to use"
    )
    args = parser.parse_args()

    main(args.model, args.region)
