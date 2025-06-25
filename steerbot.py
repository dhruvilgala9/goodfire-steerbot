#!/usr/bin/env python3
"""
Steer‑Bot: Goodfire ⚡ CodeAgent 🤖
====================================
Interactive research assistant that lets you:
• converse in NL to *steer* a Goodfire `Variant` (AutoSteer / search / manual `set` / conditional rules)
• keep a live `Variant` object across turns so you can iteratively refine behaviour
• preview generations (`chat_once`) and inspect / visualise feature‑weight diffs after every edit
• evaluate the current Variant on an arbitrary dataset and compare to the *base* model

Run it locally:
    $ python steerbot.py        # CLI loop
    or, inside Jupyter: `from steerbot import agent; agent.run("…")`

Install:
    pip install goodfire-sdk smolagents openai pandas datasets matplotlib

Secrets (env vars):
    GOODFIRE_API_KEY – Goodfire key
    OPENAI_API_KEY   – OpenAI / LiteLLM key (we default to GPT‑4o)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset as hf_load_dataset

from goodfire import Client, Variant, Feature, FeatureEdits
from smolagents import CodeAgent, LiteLLMModel, tool

# ---------------------------------------------------------------------------
# 🔑 Keys & objects
# ---------------------------------------------------------------------------
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOODFIRE_API_KEY:
    raise RuntimeError("Set GOODFIRE_API_KEY e.g. `export GOODFIRE_API_KEY=sk-goodfire-…`.")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY e.g. `export OPENAI_API_KEY=sk-…`.")

GF: Client = Client(api_key=GOODFIRE_API_KEY)

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
variant: Variant = Variant(BASE_MODEL)          # ✏️ steered copy
base_variant: Variant = Variant(BASE_MODEL)     # 🧑‍🔬 untouched baseline

# ---------------------------------------------------------------------------
# 🛠️ Tools (must have proper docstrings for smolagents)
# ---------------------------------------------------------------------------

@tool
def autosteer(spec: str) -> FeatureEdits:
    """Convert a natural-language behaviour request into Goodfire *FeatureEdits*. Only use this tool when the user gives a plain-English style or behaviour request ("be funnier", "reduce medical advice"). Don't use this tool if the user gives a specific feature or weight to set.

    Args:
        spec (str): Natural‑language description such as "sound more formal" or
            "inject pirate slang".

    Returns:
        FeatureEdits: The steering vector proposed by Goodfire's AutoSteer.

    Always store_to_memory the results of any tool call in the global memory using store_to_memory() and you can read_from_memory them later using read_from_memory().
    """
    return GF.features.AutoSteer(specification=spec, model=variant)


@tool
def set_edits(edits: Any) -> str:
    """Apply steering edits to the live Variant. You can use this tool to set the weight of a specific feature or to set the weight of multiple features at once.

    Args:
        edits (Any): Can be a `FeatureEdits` or a `dict[Feature|str, float] where the key is a feature uuid or a feature object and the value is the weight to set for that feature.

    Returns:
        str: Confirmation including the total number of edited features.
    """
    variant.set(edits)
    return f"✅ Applied edits — features now steered."


@tool
def search_features(query: str, k: int = 5) -> List[Feature]:
    """Semantic search for latent features.

    Args:
        query (str): Concept to search (e.g. "humor", "medical advice").
        k (int, optional): Number of results. Defaults to 5.

    Returns:
        List[Feature]: Top‑k matching features.

    Always call store_to_memory() after calling this tool to store_to_memory the results for later use.
    """
    return GF.features.search(query, model=variant, top_k=k)


@tool
def chat_once(prompt: str) -> str:
    """Generate a single reply from the *current* steered Variant.

    Args:
        prompt (str): User prompt.

    Returns:
        str: Assistant response.
    """
    resp = GF.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=variant,
    )
    return resp.choices[0].message["content"]


def _hf_or_local(path: str) -> pd.DataFrame:
    """Load a local CSV/JSONL/Parquet *or* a HF dataset tag into a DataFrame."""
    if os.path.isfile(path):
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".jsonl"):
            return pd.read_json(path, lines=True)
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported file type: {path}")

    ds = hf_load_dataset(path)
    split = next(iter(ds.keys())) if isinstance(ds, dict) else "train"
    return ds[split].to_pandas()


def _judge_reply(original: str, reply: str, criteria: str) -> float:
    """
    Ask LLM whether *reply* meets *criteria* w.r.t. *original*.

    Returns 1.0 for "yes / passes", 0.0 for "no / fails".
    The prompt is deterministic (temperature 0).
    """
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    judge_prompt = (
        "You are an impartial evaluator.\n"
        "Reply ONLY with '1' if the reply SATISFIES the criterion, "
        "or '0' if it does NOT.\n\n"
        f"CRITERION: {criteria}\n\n"
        f"ORIGINAL INPUT:\n{original}\n\n"
        f"MODEL REPLY:\n{reply}\n"
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return float(response.choices[0].message.content.strip().startswith("1"))


@tool
def evaluate_dataset(list_of_prompts: List[str], criteria: str = None) -> Dict[str, Any]:
    """
    Run the current Variant on *df* and score each reply with LLM using a human-supplied criteria.

    Args:
        list_of_prompts (List[str]): List of prompts to evaluate.
        criteria (str): Natural-language rubric for evaluation (e.g. "The reply is calm / non-angry.").

    Returns:
        dict: {"responses": List[str], "mean_score": float}

    Always store the results of any tool call in the global memory using store_to_memory() and you can read_from_memory them later using read_from_memory().
    You can also read_from_memory the entire memory using read_from_memory() without a key.
    """
    if not criteria or not isinstance(criteria, str) or not criteria.strip():
        print("[error] No evaluation criteria provided. Please specify a natural-language criteria for the LLM judge (e.g. 'The reply is calm / non-angry.').")
        return {"error": "No evaluation criteria provided. Please specify a criteria for the LLM judge."}
    
    for prompt in list_of_prompts:
        # 1│ generate
        replies = [chat_once(str(p)) for p in list_of_prompts]

    # 2│ judge
    scores = [
        _judge_reply(orig, rep, criteria) for orig, rep in zip(list_of_prompts, replies)
    ]
    mean_score = float(np.mean(scores))

    # 3│ preview
    print(f"\n📊  {len(list_of_prompts)} prompts — mean score (criterion: {criteria!r}): {mean_score:.2f}\n")
    for i in range(min(3, len(list_of_prompts))):
        print("Prompt :", list_of_prompts[i][:100].replace("\n", " "))
        print("Reply  :", replies[i][:100].replace("\n", " "))
        print("Score  :", scores[i])
        print("—"*40)

    return {"responses": replies, "mean_score": mean_score}

# Simple key–value store_to_memory the agent can access every turn
_GLOBAL_MEM: dict[str, Any] = {}

@tool
def store_to_memory(key: str, value: Any) -> str:
    """Save *value* under *key* for later turns. ALWAYS store any tool call results in the global memory using store_to_memory() and you can read_from_memory them later using read_from_memory().
    
    Args:
        key (str): The key to save the value under.
        value (Any): The value to save.

    Returns:
        str: Confirmation message.
    """
    _GLOBAL_MEM[key] = value
    return f"Saved under key '{key}'."

@tool
def read_from_memory(key: str = None) -> Any:
    """Retrieve an object previously stored with store_to_memory().
    
    Args:
        key (str): The key to retrieve the value from. If no key is provided, you will get the entire memory.

    Returns:
        Any: The value stored under the key or the entire memory if no key is provided.
    """
    return _GLOBAL_MEM[key] if key else _GLOBAL_MEM

# ---------------------------------------------------------------------------
# 🤖 Build CodeAgent
# ---------------------------------------------------------------------------

TOOLS = [
    autosteer,
    set_edits,
    search_features,
    chat_once,
    evaluate_dataset,
    store_to_memory,
    read_from_memory,
]

SYSTEM_PROMPT = """
You are **Steer-Bot**, a lab companion that teaches researchers how to steer
LLMs with Goodfire's *feature-level* controls.  
Every reply you generate is executed as Python, so follow the tool signatures
exactly and **store anything you will need later to the global memory** with `store_to_memory()`.

──────────────────────────  YOUR TOOLBOX  ──────────────────────────
1. store_to_memory(key: str, value: Any) → str
   • Save *value* under *key* for later turns.

2. read_from_memory(key: str) → Any
   • Retrieve an object previously stored to the global memory. If no key is provided, you will get the entire memory.

3. autosteer(spec: str) → FeatureEdits
   • Use when the user gives a plain-English style or behaviour request
     ("be funnier", "reduce medical advice").
   • Returns a *proposal* only — you still need to call set_edits to apply.

4. set_edits(edits: FeatureEdits | dict[Feature|str, float]) → str
   • Applies edits to the live Variant **and auto-plots a before/after diff**.
   • Accepts:
       – the FeatureEdits from autosteer
       – a dict like {feature_uuid: weight}

5. search_features(query: str, k: int = 5) → List[Feature]
   • Use to discover latent features when the user mentions a concept
     ("humor", "pirate slang").
   • If search returns a good match, skip autosteer and go straight to
     set_edits with a hand-tuned weight.

6. chat_once(prompt: str) → str
   • Generates ONE reply from the current Variant.
   • Ideal for quick spot-checks after steering.

7. evaluate_dataset(list_of_prompts: List[str], criteria: str) → dict
   • Runs the prompts in the list through chat_once to get the model responses.
   • **Uses an LLM judge**: Each reply is scored by GPT-4o against the
     *human-supplied* criteria (e.g. "The reply is calm / non-angry.").
   • If no criteria is provided, you MUST ask the user to supply one before proceeding.

───────────────────────  MEMORY-FIRST WORKFLOW  ────────────────────────

★ Rule of thumb  
  • **Search / Autosteer** → store_to_memory result  
  • **Edit / Evaluate**    → read_from_memory what you need

──────────────────────────  CHAINING EXAMPLES  ──────────────────────
Example A – "Make it more sarcastic and test it"
    edits = autosteer("be more sarcastic")
    set_edits(edits)
    reply = chat_once("Why is the sky blue?")   ← STOP & hand control back.
    store_to_memory("reply", reply)
    STOP (show reply & await user's test or further instructions).

Example B – "Evaluate on tweet-eval emotion"
    results = evaluate_dataset(list_of_prompts, criteria="The reply is calm / non-angry.")
    store_to_memory("results", results)
    STOP (show mean score & sample outputs).

Example C – "Search for features on medical advice"
    medical_features = search_features("medical advice", 5)
    store_to_memory("medical_features", medical_features) # ALWAYS STORE INTERMEDIATE RESULTS TO THE GLOBAL MEMORY
    STOP (await user's test or further instructions).

Example D - "Set the 3rd feature from medical_features to 0.5"
    feature = read_from_memory("medical_features")[2] # from prior search_features call
    set_edits({feature: 0.5})
    STOP (await user's test or further instructions).

────────────────────────  GENERAL GUIDELINES  ───────────────────────
• **Think step-by-step** but only expose the minimal reasoning needed.
• **After each tool call**:
    – If the user's request is satisfied → say "Done." and WAIT.
    – Else pick the next tool logically required.
    - Store the results of any tool call in the global memory using store_to_memory() and you can read from memory later using read_from_memory().
• Never write raw imports; always use the tools above.
• Be concise (≤ 40 tokens) unless the user explicitly asks for detail.
"""

agent = CodeAgent(
    description=SYSTEM_PROMPT,
    tools=TOOLS,
    model=LiteLLMModel(model_id="o4-mini", api_key=OPENAI_API_KEY),
    additional_authorized_imports=["matplotlib", "matplotlib.pyplot", "pandas"],
)

# ---------------------------------------------------------------------------
# 🖥️  Optional CLI entrypoint
# ---------------------------------------------------------------------------

def _cli() -> None:
    """Launch a simple REPL loop (exit with Ctrl‑C or 'quit'). Supports up-arrow history."""
    import readline
    import atexit
    import os
    HISTFILE = os.path.expanduser("~/.steerbot_history")
    try:
        readline.read_history_file(HISTFILE)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, HISTFILE)

    print("Steer‑Bot CLI — type 'quit' to exit.\n")
    while True:
        try:
            user = input("🧑‍💻 > ").strip()
            if user.lower() in {"quit", "exit"}:
                break
            print("🤖 :", agent.run(user))
        except KeyboardInterrupt:
            break
        except Exception as exc:  # pylint: disable=broad-except
            print("[error]", exc)


if __name__ == "__main__":
    _cli()
