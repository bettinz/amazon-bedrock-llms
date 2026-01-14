# Amazon Nova LLM

[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=383938&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![Awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=Awesome+plugin&color=000000&style=for-the-badge&logo=cheshire_cat_ai)](https://)
[![awesome plugin](https://custom-icon-badges.demolab.com/static/v1?label=&message=awesome+plugin&color=F4F4F5&style=for-the-badge&logo=cheshire_cat_black)](https://)

This plugin integrates Amazon Nova LLMs (Pro and Lite) into the Cheshire Cat AI framework, allowing users to leverage AWS's powerful Nova foundation models.

## Key Features

- **Amazon Nova Support**: Direct integration with Amazon Nova Pro and Nova Lite models from AWS Bedrock.
- **Configurable Model**: Choose between Nova Pro and Nova Lite via settings.
- **Streaming Support**: Enables streaming responses for real-time interaction.
- **Cost Monitoring**: Track and manage costs with built-in budget controls and pricing information.
- **Flexible Configuration**: Customize model parameters and budget settings.

## Supported Models

| Model | Model ID | Description |
|-------|----------|-------------|
| Nova Pro | `amazon.nova-pro-v1:0` | High-performance model for complex tasks |
| Nova Lite | `amazon.nova-lite-v1:0` | Cost-effective model for lighter workloads |

## How It Works

1. The plugin uses the AWS Boto3 client to interact with Amazon Bedrock services.
2. Configure your preferred Nova model (Pro or Lite) in the settings.
3. The plugin integrates with the Cheshire Cat AI framework, allowing the use of Nova models in various AI tasks.
4. Cost monitoring tracks token usage and provides detailed pricing breakdowns.

## Configuration

The plugin provides settings that allow users to:

- **Model ID**: Choose between Nova Pro (`amazon.nova-pro-v1:0`) or Nova Lite (`amazon.nova-lite-v1:0`)
- **Model kwargs**: Configure model-specific parameters via JSON format
- **Input/Output Token Price**: Customize pricing for accurate cost tracking
- **Budget Mode**: Choose from:
  - **Disabled**: No budget tracking
  - **Monitor**: Log costs without restrictions
  - **Notify**: Warn when budget is exceeded
  - **Trace**: Show cost breakdown in responses
  - **Block**: Prevent calls when budget is exceeded
- **Budget Limit**: Set maximum budget in USD

## Pricing

### Nova Pro (US East region)
- **Input tokens**: $0.0008 per 1,000 tokens
- **Output tokens**: $0.0032 per 1,000 tokens

### Nova Lite (US East region)
- **Input tokens**: $0.00006 per 1,000 tokens
- **Output tokens**: $0.00024 per 1,000 tokens

## Usage

1. Ensure you have the necessary AWS credentials and permissions to access Amazon Bedrock services.
2. Install the plugin in your Cheshire Cat AI environment.
3. Configure the Nova model settings through the Cheshire Cat AI interface.
4. Select your preferred model ID (Nova Pro or Nova Lite).
5. The plugin will automatically integrate Nova into your AI pipeline.
6. Optionally enable cost monitoring to track usage expenses.

## Available Tools

- **Get Current Model**: Shows the current model information
- **Get Current Model Pricing**: Displays Nova Pro and Lite pricing details
- **Get Current Model Cost**: Shows accumulated usage costs
- **Reset Cumulative Model Cost**: Clears the cost tracking data

## Requirements

- Active AWS account with access to Amazon Bedrock services
- AWS credentials configured with appropriate permissions
- Access to Amazon Nova models in your AWS region

## Note

This plugin requires an active AWS account with access to Amazon Bedrock services. Make sure you understand the pricing and usage terms of Amazon Bedrock before using this plugin in production environments.

## Author

Created by [bettinz](https://github.com/bettinz)

