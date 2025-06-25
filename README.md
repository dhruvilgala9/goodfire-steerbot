# SteerBot: Goodfire ⚡ CodeAgent 🤖

Interactive research assistant that lets you steer LLM behavior using Goodfire's feature-level controls through natural language conversation.

## 🚀 Features

- **Natural Language Steering**: Converse in plain English to modify LLM behavior ("be funnier", "reduce medical advice")
- **Live Variant Management**: Keep a live `Variant` object across conversation turns for iterative refinement
- **Real-time Preview**: Generate and inspect responses after every steering edit
- **Feature Discovery**: Search for latent features and manually tune their weights
- **Evaluation Framework**: Test your steered model on custom datasets with LLM-based scoring
- **Visual Diff Plots**: See before/after feature weight changes automatically

## 🛠️ Installation

```bash
pip install goodfire-sdk smolagents openai pandas datasets matplotlib
```

## 🔑 Setup

Set your API keys as environment variables:

```bash
export GOODFIRE_API_KEY="sk-goodfire-..."
export OPENAI_API_KEY="sk-..."
```

## 🎯 Quick Start

### CLI Mode
```bash
python steerbot.py
```

## 🧰 Available Tools

### Core Steering
- **`autosteer(spec)`**: Convert natural language requests into feature edits
- **`set_edits(edits)`**: Apply steering edits to the live variant
- **`search_features(query, k=5)`**: Find latent features by semantic search

### Testing & Evaluation
- **`chat_once(prompt)`**: Generate a single response from current variant
- **`evaluate_dataset(prompts, criteria)`**: Test variant on dataset with LLM scoring

### Memory Management
- **`store_to_memory(key, value)`**: Save results for later use
- **`read_from_memory(key)`**: Retrieve stored data

## 📖 Usage Examples

### Example 1: Style Modification
```
User: "Make the model more formal"
Bot: [Uses autosteer → set_edits → chat_once to test]
```

### Example 2: Feature Discovery
```
User: "Search for humor-related features"
Bot: [Searches → stores results → awaits further instructions]
```

### Example 3: Manual Tuning
```
User: "Set the 3rd humor feature to weight 0.5"
Bot: [Reads from memory → applies specific edit]
```

### Example 4: Dataset Evaluation
```
User: "Test on these prompts with criteria 'reply is helpful'"
Bot: [Evaluates → shows mean score and samples]
```

## 🔬 Research Workflow

1. **Explore**: Use `search_features()` to discover relevant latent features
2. **Steer**: Apply `autosteer()` for natural language requests or `set_edits()` for precise control
3. **Test**: Use `chat_once()` for quick checks or `evaluate_dataset()` for systematic evaluation
4. **Iterate**: Refine behavior based on results and continue steering

## 🏗️ Architecture

- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Agent Framework**: SmolAgents with CodeAgent
- **LLM Backend**: LiteLLM (defaults to GPT-4o)
- **Memory**: Global key-value store for cross-turn persistence

## 📊 Evaluation

The evaluation system uses an LLM judge to score responses against human-defined criteria:

```python
# Example evaluation
results = evaluate_dataset(
    prompts=["Hello", "How are you?"],
    criteria="The reply is friendly and helpful"
)
# Returns: {"responses": [...], "mean_score": 0.85}
```

## 🙏 Acknowledgments

Built with:
- [Goodfire](https://goodfire.ai/) - Feature-level LLM control
- [SmolAgents](https://github.com/smol-ai/smolagents) - Agent framework
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM abstraction layer