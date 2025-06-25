#!/usr/bin/env python3
"""
Steer‑Bot: Goodfire ⚡ CodeAgent 🤖
====================================
Interactive research assistant that lets you:
• converse in NL to *steer* a Goodfire `Variant` (AutoSteer / search / manual `set` / conditional rules)
• keep a live `Variant` object across turns so you can iteratively refine behaviour
• preview generations (`test_once`) and inspect / visualise feature‑weight diffs after every edit
• evaluate the current Variant on an arbitrary dataset and compare to the *base* model

Run it locally:
    $ python steerbot.py        # CLI loop
    or, inside Jupyter: `from steerbot import agent; agent.run("…")`

Install:
    pip install goodfire-sdk smolagents openai pandas datasets matplotlib

Secrets (env vars):
    GOODFIRE_API_KEY – Goodfire key
    OPENAI_API_KEY   – OpenAI / LiteLLM key (we default to o4-mini)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import json

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
    """
    return GF.features.search(query, model=variant, top_k=k)


@tool
def test_once(prompt: str) -> str:
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
    """
    if not criteria or not isinstance(criteria, str) or not criteria.strip():
        print("[error] No evaluation criteria provided. Please specify a natural-language criteria for the LLM judge (e.g. 'The reply is calm / non-angry.').")
        return {"error": "No evaluation criteria provided. Please specify a criteria for the LLM judge."}
    
    for prompt in list_of_prompts:
        # 1│ generate
        replies = [test_once(str(p)) for p in list_of_prompts]

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

@tool
def set_when(condition: Any, edits: Any) -> str:
    """Apply feature edits when a condition is met.

    Args:
        condition (Any): A ConditionalGroup or comparison (e.g. feature > 0.7).
        edits (Any): FeatureEdits or dict[Feature|str, float] to apply when the condition is met.

    Returns:
        str: Confirmation message.
    """
    variant.set_when(condition, edits)
    return "✅ Conditional intervention set."

@tool
def abort_when(condition: Any) -> str:
    """Abort generation when a condition is met by raising an InferenceAbortedException.

    Args:
        condition (Any): A ConditionalGroup or comparison (e.g. feature > 0.7).

    Returns:
        str: Confirmation message.
    """
    variant.abort_when(condition)
    return "✅ Abort condition set."

@tool
def export_variant(name: str = "variant.json") -> str:
    """Export the current Variant as a JSON string.

    Args:
        name (str): The name of the variant file to export (inside variants/).

    Returns:
        str: Confirmation message.
    """
    os.makedirs("variants", exist_ok=True)
    path = os.path.join("variants", name)
    with open(path, "w") as f:
        f.write(json.dumps(variant.json()))
    return f"✅ Variant exported to {path}."

@tool
def load_variant_from_json(variant_json: str) -> str:
    """Load a Variant from a JSON string and set it as the current Variant.

    Args:
        variant_json (str): JSON string representing a Variant.

    Returns:
        str: Confirmation message.
    """
    global variant
    from goodfire import Variant as GFVariant
    variant = GFVariant.from_json(variant_json)
    return "✅ Variant loaded from JSON."


# ---------------------------------------------------------------------------
# 🤖 Build CodeAgent
# ---------------------------------------------------------------------------

TOOLS = [
    autosteer,
    set_edits,
    search_features,
    test_once,
    evaluate_dataset,
    set_when,
    abort_when,
    export_variant,
    load_variant_from_json,

]

SYSTEM_PROMPT = """
You are **Steer-Bot**, a lab companion that teaches researchers how to steer
LLMs with Goodfire's *feature-level* controls.  
Every reply you generate is executed as Python, so follow the tool signatures
exactly.

──────────────────────────  YOUR TOOLBOX  ──────────────────────────
1. autosteer(spec: str) → FeatureEdits
   • Use when the user gives a plain-English style or behaviour request
     ("be funnier", "reduce medical advice").
   • Returns a *proposal* only — you still need to call set_edits to apply.

2. set_edits(edits: FeatureEdits | dict[Feature|str, float]) → str
   • Applies edits to the live Variant **and auto-plots a before/after diff**.
   • Accepts:
       – the FeatureEdits from autosteer
       – a dict like {feature_uuid: weight}

3. search_features(query: str, k: int = 5) → List[Feature]
   • Use to discover latent features when the user mentions a concept
     ("humor", "pirate slang").
   • If search returns a good match, skip autosteer and go straight to
     set_edits with a hand-tuned weight.

4. test_once(prompt: str) → str
   • Generates ONE reply from the current model Variant.
   • Ideal for quick spot-checks after steering.

5. evaluate_dataset(list_of_prompts: List[str], criteria: str) → dict
   • Runs the prompts in the list through test_once to get the model responses.
   • **Uses an LLM judge**: Each reply is scored by GPT-4o against the
     *human-supplied* criteria (e.g. "The reply is calm / non-angry.").
   • If no criteria is provided, you MUST ask the user to supply one before proceeding.

6. set_when(condition: Any, edits: FeatureEdits | dict[Feature|str, float]) → str
   • Applies feature edits only when a condition is met (conditional steering).
   • Example: set pirate features when whale features are detected.

7. abort_when(condition: Any) → str
   • Aborts generation when a condition is met (e.g., if certain content is detected).
   • Example: abort if whale features are too strong.

8. export_variant() → str
   • Exports the current Variant as a JSON string.
   • Example: Save the current steering state for later use or sharing.

9. load_variant_from_json(variant_json: str) → str
   • Loads a Variant from a JSON string and sets it as the current Variant.
   • Example: Restore a previously saved Variant.

──────────────────────────  CHAINING EXAMPLES  ──────────────────────
Example A – "Make it more sarcastic and test it"
    edits = autosteer("be more sarcastic")
    set_edits(edits)
    reply = test_once("Why is the sky blue?")   ← STOP & hand control back.
    STOP (show reply & await user's test or further instructions).

Example B – "Evaluate on tweet-eval emotion"
    results = evaluate_dataset(list_of_prompts, criteria="The reply is calm / non-angry.")
    STOP (show mean score & sample outputs).

Example C – "Search for features on medical advice"
    medical_features = search_features("medical advice", 5)
    STOP (await user's test or further instructions).

Example D - "Set the 3rd feature from medical_features to 0.5"
    feature = medical_features[2] # from prior search_features call
    set_edits({feature: 0.5})
    STOP (await user's test or further instructions).

Example E – "Talk like a pirate when whales are mentioned"
    whale_feature = search_features("whales", 1)[0]
    pirate_feature = search_features("talk like a pirate", 1)[0]
    set_when(whale_feature > 0.75, {pirate_feature: 0.4})
    reply = test_once("Tell me about whales.")
    STOP (show reply & await user's test or further instructions).

Example F – "Abort if whale content is detected"
    whale_feature = search_features("whales", 1)[0]
    abort_when(whale_feature > 0.75)
    try:
        reply = test_once("Tell me about whales.")
    except Exception as exc:
        print("Generation aborted due to whale content")
    STOP (show result & await user's test or further instructions).
    
Example G – "Export and import a variant"
    # Export current variant to JSON
    variant_json = export_variant()
    # ...save or share variant_json...
    # Load variant from JSON
    load_variant_from_json(variant_json)
    STOP (confirm variant loaded & await further instructions).

Example H – "Try 'What is the weather in Tokyo?'"
    reply = test_once("What is the weather in Tokyo?")
    STOP (show reply & await user's test or further instructions).

────────────────────────  GENERAL GUIDELINES  ───────────────────────
• **Think step-by-step** but only expose the minimal reasoning needed.
• **After each tool call**:
    – If the user's request is satisfied → say "Done." and WAIT.
    – Else pick the next tool logically required.
• Never write raw imports; always use the tools above.
• Be concise (≤ 40 tokens) unless the user explicitly asks for detail.
"""

agent = CodeAgent(
    description=SYSTEM_PROMPT,
    tools=TOOLS,
    model=LiteLLMModel(model_id="o4-mini", api_key=OPENAI_API_KEY),
    additional_authorized_imports=["matplotlib", "matplotlib.pyplot", "pandas", "repr", "json", "dir", "open"],
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
            print("🤖 :", agent.run(user, reset=False))
        except KeyboardInterrupt:
            break
        except Exception as exc:  # pylint: disable=broad-except
            print("[error]", exc)


if __name__ == "__main__":
    _cli()
