import os
import json
from enum import Enum
from typing import List, Optional, Type

from pydantic import Field, ConfigDict
from langchain_core.messages import AIMessage
from langchain_aws import ChatBedrock
import logging

from cat.mad_hatter.decorators import tool, hook
from cat.mad_hatter.mad_hatter import MadHatter
from cat.factory.llm import LLMSettings
from cat.plugins.aws_integration import Boto3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PLUGIN_NAME = "amazon_nova_llm"

# Default Nova model IDs (with region prefix)
# Use us. for US regions, eu. for EU regions
# Examples: us.amazon.nova-pro-v1:0, eu.amazon.nova-lite-v1:0
NOVA_PRO_MODEL_ID = "us.amazon.nova-pro-v1:0"
NOVA_LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"
DEFAULT_MODEL_ID = NOVA_PRO_MODEL_ID

CACHED_COST_FILE = os.path.join(
    MadHatter().plugins.get(PLUGIN_NAME)._path, "cached_model_costs.json"
)

bedrock_runtime_client = Boto3().get_client("bedrock-runtime")


class BudgetMode(str, Enum):
    DISABLED = "Disabled"
    MONITOR = "Monitor"
    NOTIFY = "Notify"
    TRACE = "Trace"
    BLOCK = "Block"


# Nova Pro pricing (per 1K tokens) - US East region
NOVA_PRO_INPUT_PRICE = 0.0008  # $0.0008 per 1K input tokens
NOVA_PRO_OUTPUT_PRICE = 0.0032  # $0.0032 per 1K output tokens

# Nova Lite pricing (per 1K tokens) - US East region
NOVA_LITE_INPUT_PRICE = 0.00006  # $0.00006 per 1K input tokens
NOVA_LITE_OUTPUT_PRICE = 0.00024  # $0.00024 per 1K output tokens

NOVA_TOKEN_UNIT = 1000


def get_default_pricing(model_id: str):
    """Returns default pricing based on model ID."""
    if "lite" in model_id.lower():
        return NOVA_LITE_INPUT_PRICE, NOVA_LITE_OUTPUT_PRICE
    return NOVA_PRO_INPUT_PRICE, NOVA_PRO_OUTPUT_PRICE


class NovaLLM(ChatBedrock):
    """Custom ChatBedrock class for Amazon Nova models (Pro and Lite)."""

    def __init__(self, **kwargs):
        model_id = kwargs.get("model_id", DEFAULT_MODEL_ID)
        default_input_price, default_output_price = get_default_pricing(model_id)

        input_kwargs = {
            "model_id": model_id,
            "provider": "amazon",
            "streaming": True,
            "model_kwargs": json.loads(kwargs.get("model_kwargs", "{}")),
            "client": Boto3().get_client("bedrock-runtime"),
        }

        input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}
        super(NovaLLM, self).__init__(**input_kwargs)

        if kwargs.get("budget_mode", "Disabled") != "Disabled":
            budget_limit = kwargs.get("budget_limit", 0.0)
            input_price = kwargs.get("input_token_price", default_input_price)
            output_price = kwargs.get("output_token_price", default_output_price)

            def parse_float(value, default=0.0):
                if isinstance(value, (int, float)):
                    return float(value)
                return float(value) if str(value).replace(".", "", 1).isdigit() else default

            budget_limit = parse_float(budget_limit)
            input_price = parse_float(input_price)
            output_price = parse_float(output_price)

            setattr(
                NovaLLM,
                "_budget_config",
                {
                    "budget_limit": budget_limit,
                    "input_token_price": input_price / NOVA_TOKEN_UNIT,
                    "output_token_price": output_price / NOVA_TOKEN_UNIT,
                    "budget_mode": kwargs.get("budget_mode", "Disabled"),
                },
            )

    def get_current_model_cost(self):
        """Retrieves the total model cost from the cache file."""
        if os.path.exists(CACHED_COST_FILE):
            try:
                with open(CACHED_COST_FILE, "r") as file:
                    pricing_cache = json.load(file) or {}
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load pricing cache. Error: {e}")
                pricing_cache = {}
        else:
            pricing_cache = {}

        return float(pricing_cache.get("current_cost", 0.0))

    def compute_invocation_cost(self, input_tokens, output_tokens, total_tokens):
        """Computes cost for the current request and updates the total model cost."""
        budget_config = getattr(self, "_budget_config", {})
        input_price = budget_config.get("input_token_price", 0.0)
        output_price = budget_config.get("output_token_price", 0.0)

        input_cost = input_price * input_tokens
        output_cost = output_price * output_tokens
        current_request_cost = round(input_cost + output_cost, 6)

        model_total_cost = self.get_current_model_cost() + current_request_cost

        pricing_cache = {"current_cost": model_total_cost}
        try:
            with open(CACHED_COST_FILE, "w") as file:
                json.dump(pricing_cache, file, indent=4)
        except IOError as e:
            logger.error(f"Error saving pricing cache: {e}")

        return model_total_cost, current_request_cost

    def invoke(self, *args, **kwargs):
        budget_config = getattr(self, "_budget_config", {})
        budget_mode = str(budget_config.get("budget_mode", BudgetMode.DISABLED)).capitalize()
        budget_limit = float(budget_config.get("budget_limit", 0.0))

        model_total_cost = self.get_current_model_cost()

        alert_message = ""
        if budget_limit > 0 and model_total_cost > budget_limit:
            alert_message = f"⚠️ **Budget Limit Exceeded!** Budget Limit: ${budget_limit:.6f}"
            logger.warning(alert_message)

        if budget_mode == BudgetMode.BLOCK.value and model_total_cost > budget_limit:
            return AIMessage(
                content="⛔ **Invocation Blocked Due to Budget Constraints.**\n"
                "Your request cannot be processed because the total cost has exceeded the budget limit.\n"
                "💰 **Cost Breakdown:**\n"
                f"   - 🎯 **Budget Limit:** `${budget_limit:.6f}`\n"
                f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
            )

        response = super().invoke(*args, **kwargs)

        try:
            usage_metadata = response.usage_metadata
            model_total_cost, current_request_cost = self.compute_invocation_cost(**usage_metadata)

            response.usage_metadata["current_request_cost"] = round(current_request_cost, 6)
            response.usage_metadata["model_total_cost"] = round(model_total_cost, 6)

            if budget_mode == BudgetMode.MONITOR.value:
                logger.info(f"Invocation Cost: ${current_request_cost:.6f}")
                logger.info(f"Total Cost (All Calls): ${model_total_cost:.6f}")
                if alert_message:
                    logger.warning(alert_message)

            if budget_mode == BudgetMode.NOTIFY.value and alert_message:
                response.content += f"\n\n🚨 **{alert_message}** 🚨\n"

            if budget_mode == BudgetMode.TRACE.value:
                response.content += "\n\n"
                if alert_message:
                    response.content += f"🚨 **{alert_message}** 🚨\n"
                response.content += (
                    f"💰 **Cost Breakdown:**\n"
                    f"   - 📝 Request Cost: `${current_request_cost:.6f}`\n"
                    f"   - 📊 Total Cost: `${model_total_cost:.6f}`"
                )

            response.usage_metadata["budget_mode"] = budget_mode

        except Exception as e:
            logger.error(f"Error processing cost computation: {e}")

        return response


class NovaLLMConfig(LLMSettings):
    """Configuration for Amazon Nova LLMs (Pro and Lite)."""

    model_id: str = Field(
        default=DEFAULT_MODEL_ID,
        description=(
            "The Amazon Nova model ID with region prefix. "
            "Use 'us.' for US regions or 'eu.' for EU regions.\n"
            "Supported models:\n"
            f"- Nova Pro (US): {NOVA_PRO_MODEL_ID}\n"
            f"- Nova Lite (US): {NOVA_LITE_MODEL_ID}\n"
            "- Nova Pro (EU): eu.amazon.nova-pro-v1:0\n"
            "- Nova Lite (EU): eu.amazon.nova-lite-v1:0"
        ),
    )
    model_kwargs: Optional[str] = Field(
        default="{}",
        description="Additional keyword arguments for the model in JSON string format.",
    )
    input_token_price: float = Field(
        default=NOVA_PRO_INPUT_PRICE,
        description=f"The price per {NOVA_TOKEN_UNIT} input tokens (in USD). Default is Nova Pro pricing.",
    )
    output_token_price: float = Field(
        default=NOVA_PRO_OUTPUT_PRICE,
        description=f"The price per {NOVA_TOKEN_UNIT} output tokens (in USD). Default is Nova Pro pricing.",
    )
    budget_mode: BudgetMode = Field(
        default=BudgetMode.DISABLED,
        description=(
            "The budget mode for the model, which controls cost monitoring and enforcement. "
            "Options:\n"
            "Disabled: No budget tracking or restrictions.\n"
            "Monitor: Logs the cost of each invocation without any notifications or enforcement.\n"
            "Notify: Sends a warning notification when the budget limit is exceeded.\n"
            "Trace: Appends cost breakdown details to the model's response.\n"
            "Block: Prevents further invocations once the budget limit is exceeded."
        ),
    )
    budget_limit: Optional[float] = Field(
        default=0.0,
        description="The maximum budget for the model in USD.",
    )
    _pyclass: Type = NovaLLM

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Amazon Nova",
            "description": "Amazon Nova LLMs (Pro and Lite) - Powerful foundation models from AWS Bedrock",
            "link": "https://aws.amazon.com/bedrock/",
        },
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="allow",
    )


@hook
def agent_prompt_prefix(prefix, cat):
    prefix = """Please do not include any cost breakdowns, request costs, or total cost information in responses. 
        Focus only on the main conversation topic and user requests."""
    return prefix


@tool(
    "Reset Cumulative Model Cost",
    return_direct=False,
    examples=[
        "Reset the stored model cost data.",
        "Clear all accumulated cost records for the model.",
        "Delete the current cost cache and start fresh.",
    ],
)
def reset_cached_model_costs(data, cat):
    """Reset the cumulative model cost data.

    This function clears the cached record of the total cost accumulated across all model invocations.
    It ensures that future cost tracking starts from zero.
    """
    try:
        with open(CACHED_COST_FILE, "w") as f:
            json.dump({}, f)
        return "✅ Cumulative model cost has been reset."
    except Exception as e:
        return f"❌ Error resetting cumulative model cost: {str(e)}"


@tool(
    "Get Current Model Cost",
    return_direct=False,
    examples=[
        "What is the total cost of my model usage?",
        "Show me the current cumulative cost for the model.",
        "How much have I spent on LLM calls so far?",
    ],
)
def get_current_model_cost(data, cat):
    """Retrieve the current cumulative model cost.

    Reads the cached model cost data and returns the total cost accumulated across all model invocations.
    """
    try:
        if not os.path.exists(CACHED_COST_FILE):
            return "⚠️ No cost data found. The cache might be empty."

        with open(CACHED_COST_FILE, "r") as f:
            cost_data = json.load(f)

        total_cost = cost_data.get("current_cost", 0.0)

        return (
            f"💰 **Total Accumulated Cost:** `${total_cost:.6f}`\n"
            "🔹 This includes all previous model invocations.\n"
            "⚠️ *The cost of the current request is not included and will be added after execution.*"
        )
    except Exception as e:
        return f"❌ Error retrieving model cost: {str(e)}"


@tool(
    "Get Current Model Pricing",
    return_direct=False,
    examples=[
        "How much does Nova Pro charge per token?",
        "What is the pricing for Nova Lite?",
        "Show the token price for Nova models.",
    ],
)
def get_current_model_pricing(data, cat):
    """Retrieve the pricing information for Amazon Nova models.

    Returns the cost per token for Nova Pro and Nova Lite models.
    """
    return (
        f"💲 **Amazon Nova Pricing**\n\n"
        f"**Nova Pro:**\n"
        f"🔹 **Input Cost:** ${NOVA_PRO_INPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n"
        f"🔹 **Output Cost:** ${NOVA_PRO_OUTPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n\n"
        f"**Nova Lite:**\n"
        f"🔹 **Input Cost:** ${NOVA_LITE_INPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens\n"
        f"🔹 **Output Cost:** ${NOVA_LITE_OUTPUT_PRICE:.6f} per {NOVA_TOKEN_UNIT} tokens"
    )


@tool(
    "Get Current Model",
    return_direct=False,
    examples=[
        "Which LLM model am I using?",
        "What is my current AI model?",
        "Show the model name I am working with.",
    ],
)
def get_current_model(data, cat):
    """Retrieve the currently selected AI model.

    Returns the name and ID of the model in use.
    """
    return (
        f"🤖 **Current Model Information**\n"
        f"🔹 **Model:** Amazon Nova\n"
        f"🔹 **Available Models (use region prefix us. or eu.):**\n"
        f"   - Nova Pro (US): `{NOVA_PRO_MODEL_ID}`\n"
        f"   - Nova Lite (US): `{NOVA_LITE_MODEL_ID}`\n"
        f"   - Nova Pro (EU): `eu.amazon.nova-pro-v1:0`\n"
        f"   - Nova Lite (EU): `eu.amazon.nova-lite-v1:0`\n"
        "Amazon Nova models are powerful foundation models from AWS Bedrock."
    )


@hook
def factory_allowed_llms(allowed, cat) -> List:
    return allowed + [NovaLLMConfig]

