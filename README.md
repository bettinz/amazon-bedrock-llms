# Amazon Bedrock LLMs

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

This plugin integrates Amazon Bedrock LLMs (Language Models) into the Cheshire Cat AI framework, allowing users to leverage a variety of powerful language models provided by Amazon Web Services (AWS).

## Key Features

- Dynamic model selection: Automatically fetches and configures available Bedrock models.
- Guardrail support: Implements AWS Bedrock guardrails for enhanced security and compliance.
- Streaming support: Enables streaming responses for supported models.
- Flexible configuration: Allows customization of model parameters and settings.
- Cost Monitoring: Integrates cost analysis and monitoring for running AWS Bedrock models, including detailed pricing breakdowns by model usage (e.g., input/output tokens, cache read tokens).

## How It Works

1. The plugin uses the AWS Boto3 client to interact with Amazon Bedrock services.
2. It dynamically fetches available models and guardrails from the Bedrock API.
3. Custom Bedrock LLM classes are created for each available model.
4. The plugin integrates with the Cheshire Cat AI framework, allowing the use of these models in various AI tasks.
5. Cost Monitoring: The plugin now retrieves and monitors the cost of using various AWS Bedrock models by querying AWS pricing APIs. It provides breakdowns of model usage costs, such as token-based pricing, and helps users track their expenses.

## Configuration

The plugin provides a dynamic settings model that allows users to:

- Enable/disable specific Bedrock models
- Configure model-specific parameters (e.g., temperature, max tokens)
- Set guardrails for enhanced security and compliance
- Customize model behavior through additional keyword arguments
- Enable Cost Monitoring: Configure the cost analysis feature to track usage-based pricing for different models and usage types (e.g., input-output tokens).

## Usage

1. Ensure you have the necessary AWS credentials and permissions to access Amazon Bedrock services.
2. Install the plugin in your Cheshire Cat AI environment.
3. Configure the desired Bedrock models and settings through the Cheshire Cat AI interface.
4. The plugin will automatically integrate the selected Bedrock models into your AI pipeline.
5. Monitor Costs: Enable cost monitoring to keep track of the costs associated with the models.

## Important Cheshire Cat installation note

Cheshire Cat imports plugin Python modules from the plugin folder path. Because of that, the plugin folder name must be import-safe.

- If you install from a ZIP package, use a package whose root folder is named `amazon_bedrock_llms`.
- If you copy the plugin manually into `cat/plugins`, rename the folder to `amazon_bedrock_llms` before starting Cheshire Cat.

This repository includes a packaging helper that builds a compatible ZIP archive:

```python
python build_plugin_package.py
```

It creates:

- `dist/amazon_bedrock_llms.zip`

Using that archive avoids plugin discovery issues caused by hyphenated folder names.

For detailed configuration options and advanced usage, please refer to the plugin settings in the Cheshire Cat AI interface.

## Note

This plugin requires an active AWS account with access to Amazon Bedrock services. Make sure you understand the pricing and usage terms of Amazon Bedrock before using this plugin in production environments.

