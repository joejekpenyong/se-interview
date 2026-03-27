import os
import argparse
from dotenv import load_dotenv
from datetime import datetime, timedelta
from phoenix.client import Client
from phoenix.evals import OpenAIModel, llm_classify

load_dotenv()

# ── Parse arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--rerun",
    action="store_true",
    help="Rerun evaluations on all spans, overwriting existing annotations"
)
args = parser.parse_args()

# Remove conflicting env vars for this process only
os.environ.pop("PHOENIX_CLIENT_HEADERS", None)
os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)

# Connect to Phoenix Cloud
client = Client(
    base_url="https://app.phoenix.arize.com/s/se-interview",
    headers={"Authorization": f"Bearer {os.getenv('PHOENIX_API_KEY')}"}
)

# ── Fetch spans from Phoenix ──────────────────────────────────────────────────
print("Fetching spans from Phoenix...")
spans_df = client.spans.get_spans_dataframe(
    project_identifier="travel-assistant",
    limit=200,
    start_time=datetime.now() - timedelta(days=7)
)
print(f"Found {len(spans_df)} spans")

# Inspect available data
print("\nAvailable columns:")
print(spans_df.columns.tolist())
print("\nSpan kinds available:")
print(spans_df["span_kind"].value_counts())

# Filter to LLM spans only
llm_spans = spans_df[spans_df["span_kind"] == "LLM"].copy()

# ── Skip already evaluated spans unless --rerun is passed ────────────────────
if not args.rerun:
    print("\nChecking for existing annotations (pass --rerun to overwrite)...")
    try:
        existing = client.spans.get_span_annotations(project_name="travel-assistant")
        already_evaluated = set(existing["span_id"].tolist())
        before = len(llm_spans)
        llm_spans = llm_spans[~llm_spans["context.span_id"].isin(already_evaluated)]
        skipped = before - len(llm_spans)
        if skipped:
            print(f"Skipping {skipped} already evaluated spans")
    except Exception as e:
        print(f"Could not fetch existing annotations, evaluating all spans: {e}")
else:
    print("\n--rerun flag set, overwriting all existing annotations")

if llm_spans.empty:
    print("No new spans to evaluate. Run with --rerun to overwrite existing annotations.")
    exit()

# Rename columns to match template variables
llm_spans = llm_spans.rename(columns={
    "attributes.input.value": "input",
    "attributes.output.value": "output"
})

# ── Split spans by type ───────────────────────────────────────────────────────
# Tool-calling spans: LLM decided to call tools (finish_reason = tool_calls)
# Final answer spans: LLM produced a final response (finish_reason = stop)
tool_calling_spans = llm_spans[
    llm_spans["output"].str.contains('"finish_reason": "tool_calls"', na=False)
].copy()

final_answer_spans = llm_spans[
    llm_spans["output"].str.contains('"finish_reason": "stop"', na=False)
].copy()

print(f"\nFound {len(tool_calling_spans)} tool-calling spans (for tool selection eval)")
print(f"Found {len(final_answer_spans)} final answer spans (for frustration and completeness evals)")

# ── Set up evaluator model ────────────────────────────────────────────────────
eval_model = OpenAIModel(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ── Eval 1: User Frustration (final answer spans) ────────────────────────────
print("\nRunning user frustration evaluation...")

FRUSTRATION_TEMPLATE = """
You are evaluating whether a user message shows signs of frustration.

Signs of frustration include:
- Repeated questions or complaints
- Expressions of impatience or anger
- ALL CAPS or excessive punctuation
- Phrases like "why can't you", "this is useless", "just tell me"
- Complaints about previous responses

[BEGIN DATA]
User message: {input}
Agent response: {output}
[END DATA]

Is the user frustrated?
Respond with either "frustrated" or "not_frustrated" and nothing else.
"""

frustration_results = llm_classify(
    data=final_answer_spans,
    template=FRUSTRATION_TEMPLATE,
    model=eval_model,
    rails=["frustrated", "not_frustrated"],
    provide_explanation=True
)
print("Frustration evaluation complete")
print(frustration_results["label"].value_counts())

# ── Eval 2: Tool Selection Correctness (tool-calling spans) ──────────────────
print("\nRunning tool selection correctness evaluation...")

TOOL_SELECTION_TEMPLATE = """
You are evaluating whether an AI travel assistant selected the appropriate
tools for a user's request.

Available tools:
- get_weather: for weather questions about a specific city
- get_exchange_rate: for currency conversion questions
- duckduckgo_search: for general travel information, attractions, hotels, flights

[BEGIN DATA]
User message: {input}
Agent output: {output}
[END DATA]

Did the agent select the correct tools for this request?
Consider:
- Did it use get_weather when the user asked about weather?
- Did it use get_exchange_rate when the user asked about currency?
- Did it use duckduckgo_search for general travel questions?
- Did it avoid calling unnecessary tools?

Respond with either "correct" or "incorrect" and nothing else.
"""

tool_selection_results = llm_classify(
    data=tool_calling_spans,
    template=TOOL_SELECTION_TEMPLATE,
    model=eval_model,
    rails=["correct", "incorrect"],
    provide_explanation=True
)
print("Tool selection evaluation complete")
print(tool_selection_results["label"].value_counts())

# ── Eval 3: Answer Completeness (final answer spans) ─────────────────────────
print("\nRunning answer completeness evaluation...")

COMPLETENESS_TEMPLATE = """
You are evaluating whether an AI travel assistant fully addressed
everything the user asked for.

[BEGIN DATA]
User message: {input}
Agent response: {output}
[END DATA]

Did the agent fully address all parts of the user's request?
Consider:
- Were all questions answered?
- Was relevant information provided for each part of the request?
- Were any parts of the request ignored or only partially addressed?

Respond with either "complete" or "incomplete" and nothing else.
"""

completeness_results = llm_classify(
    data=final_answer_spans,
    template=COMPLETENESS_TEMPLATE,
    model=eval_model,
    rails=["complete", "incomplete"],
    provide_explanation=True
)
print("Completeness evaluation complete")
print(completeness_results["label"].value_counts())

# ── Push results back to Phoenix ──────────────────────────────────────────────
print("\nPushing evaluation results to Phoenix...")

def prepare_eval_df(results_df, spans_df):
    """Prepare evaluation dataframe with span_id as index."""
    eval_df = results_df[["label", "explanation"]].copy()
    eval_df.index = spans_df["context.span_id"].values
    eval_df.index.name = "span_id"
    return eval_df

frustration_df = prepare_eval_df(frustration_results, final_answer_spans)
tool_selection_df = prepare_eval_df(tool_selection_results, tool_calling_spans)
completeness_df = prepare_eval_df(completeness_results, final_answer_spans)

client.spans.log_span_annotations_dataframe(
    dataframe=frustration_df,
    annotation_name="user_frustration",
    annotator_kind="LLM"
)

client.spans.log_span_annotations_dataframe(
    dataframe=tool_selection_df,
    annotation_name="tool_selection_correctness",
    annotator_kind="LLM"
)

client.spans.log_span_annotations_dataframe(
    dataframe=completeness_df,
    annotation_name="answer_completeness",
    annotator_kind="LLM"
)

print("\nAll evaluations complete and pushed to Phoenix!")
print("Check your Phoenix Cloud dashboard to see results.")