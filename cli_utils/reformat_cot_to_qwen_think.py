"""Reformat Augmentoolkit-style chain-of-thought into Qwen-style <think> blocks.

Augmentoolkit emits assistant answers shaped like::

    Thought Process:
    ...reasoning...

    Answer:
    ...final answer...

Qwen models reason natively with ``<think>...</think>`` followed by the answer.
Training a native-reasoning model (Qwen) on the Augmentoolkit format teaches a
*competing* CoT convention, so this script rewrites the SFT data into Qwen's
format before fine-tuning. It:

* Converts EVERY assistant turn (conversations are multi-turn) from
  ``Thought Process:/Answer:`` into ``<think>...</think>\n\n<answer>``.
* Rewrites the system prompt that instructs the old format so it agrees with
  the message format (the match is whitespace/quote tolerant on purpose --
  copy-paste pipelines love to turn straight quotes into curly ones).
* Keeps "Sources Cited" in the visible answer (outside the think block).
* Supports a per-conversation ``--keep-cot-ratio`` so some conversations keep
  visible reasoning and the rest become direct Q->A, with the system prompt
  adjusted to match each case.
* Drops records whose reasoning has no answer (malformed/truncated generations)
  rather than emitting think-only training examples.

Point it at the SFT ShareGPT JSONL (e.g.
``outputs/.../sft_run/combined_factual_data/combined_all_0.jsonl``) -- NOT the
pretraining text, which has no CoT.
"""

import argparse
import json
import random
import re
from typing import Optional, Tuple


THOUGHT_PROCESS_LABEL = "Thought Process:"
ANSWER_LABEL = "Answer:"

# Match the Augmentoolkit system-prompt instruction in a way that survives
# copy-paste corruption and minor edits: any run of whitespace (\s+), straight
# OR curly double quotes (["“”]), and the historic "reponse" typo as
# well as the corrected "response". Built from \u escapes so the *pattern*
# itself stays pure ASCII and can't be mangled by quote-smartening.
_OLD_INSTRUCTION_RE = re.compile(
    r'Preface\s+your\s+thoughts\s+with\s+["“”]Thought\s+Process:["“”]\s+'
    r'and\s+your\s+(?:reponse|response)\s+with\s+["“”]Answer:["“”]\.\s+'
    r'Write\s+the\s+filenames\s+of\s+any\s+sources\s+you\s+recalled\s+from\s+memory\s+'
    r'in\s+a\s+list\s+titled\s+["“”]Sources\s+Cited["“”]\s+'
    r'at\s+the\s+bottom\s+of\s+your\s+response\.',
    re.DOTALL,
)

_THINK_INSTRUCTION = (
    "Use <think>...</think> for any internal reasoning you need, then provide "
    "your final answer after the </think> tag. At the bottom of your response, "
    'you may include up to four (4) source filenames in a list titled '
    '"Sources Cited".'
)

_NO_THINK_INSTRUCTION = (
    "Think through the problem internally, then respond directly with your "
    "final answer and do not show your reasoning. At the bottom of your "
    'response, you may include up to four (4) source filenames in a list '
    'titled "Sources Cited".'
)


def split_thought_and_answer(text: str) -> Tuple[str, str]:
    """Split an assistant message into ``(reasoning, answer)``.

    Splits on the first ``Thought Process:`` and the first following
    ``Answer:``. If no ``Thought Process:`` marker is present, returns
    ``("", text)``. If the marker is present but there is no ``Answer:``,
    returns ``(reasoning, "")`` so the caller can drop the malformed record.
    """
    tp_idx = text.find(THOUGHT_PROCESS_LABEL)
    if tp_idx == -1:
        return "", text

    after_tp = text[tp_idx + len(THOUGHT_PROCESS_LABEL):].lstrip()
    ans_idx = after_tp.find(ANSWER_LABEL)
    if ans_idx == -1:
        return after_tp.strip(), ""

    reasoning = after_tp[:ans_idx].rstrip()
    answer = after_tp[ans_idx + len(ANSWER_LABEL):].lstrip()
    return reasoning, answer


def rewrite_system_prompt(value: str, use_think_blocks: bool) -> Tuple[str, bool]:
    """Rewrite the old-format instruction in a system prompt.

    Returns ``(new_value, changed)``. When ``use_think_blocks`` is True the
    instruction is replaced with Qwen ``<think>`` guidance; otherwise with a
    "answer directly" instruction. If the instruction is not found, the value
    is returned unchanged with ``changed=False``.
    """
    if not _OLD_INSTRUCTION_RE.search(value):
        return value, False
    replacement = _THINK_INSTRUCTION if use_think_blocks else _NO_THINK_INSTRUCTION
    # Use a function replacement so the text is inserted literally (no regex
    # backreference interpretation).
    return _OLD_INSTRUCTION_RE.sub(lambda _m: replacement, value), True


def process_line(
    obj: dict,
    keep_cot_ratio: float,
    rng: random.Random,
    stats: dict,
) -> Optional[dict]:
    """Transform one record in place. Returns the object, or None to drop it."""
    conv = obj.get("conversations")
    if not isinstance(conv, list):
        return obj

    system_idx = (
        0
        if conv and isinstance(conv[0], dict) and conv[0].get("from") == "system"
        else None
    )

    # Collect every assistant turn that uses the Thought Process/Answer format.
    assistant_splits = []  # (idx, reasoning, answer)
    for idx, msg in enumerate(conv):
        if not isinstance(msg, dict) or msg.get("from") not in {"gpt", "assistant"}:
            continue
        reasoning, answer = split_thought_and_answer(msg.get("value", ""))
        if not reasoning:
            continue  # plain answer, leave as-is
        if not answer:
            stats["dropped_malformed"] += 1
            return None  # reasoning with no answer -> drop the whole record
        assistant_splits.append((idx, reasoning, answer))

    # No CoT turns: only align the system prompt wording (assume think-style).
    if not assistant_splits:
        if system_idx is not None:
            new_val, changed = rewrite_system_prompt(
                conv[system_idx].get("value", ""), use_think_blocks=True
            )
            conv[system_idx]["value"] = new_val
            stats["sysprompt_rewrites"] += int(changed)
        return obj

    # One decision per conversation so the system prompt and answers agree.
    keep_cot = rng.random() < keep_cot_ratio
    stats["kept_cot"] += int(keep_cot)
    stats["stripped_cot"] += int(not keep_cot)

    if system_idx is not None:
        new_val, changed = rewrite_system_prompt(
            conv[system_idx].get("value", ""), use_think_blocks=keep_cot
        )
        conv[system_idx]["value"] = new_val
        stats["sysprompt_rewrites"] += int(changed)

    for idx, reasoning, answer in assistant_splits:
        if keep_cot:
            conv[idx]["value"] = f"<think>\n{reasoning}\n</think>\n\n{answer}"
        else:
            conv[idx]["value"] = answer or conv[idx].get("value", "")

    stats["transformed_turns"] += len(assistant_splits)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reformat Augmentoolkit Thought Process:/Answer: CoT into "
            "Qwen-style <think> blocks in SFT JSONL files."
        )
    )
    parser.add_argument("input", help="Path to input JSONL file")
    parser.add_argument("output", help="Path to output JSONL file")
    parser.add_argument(
        "--keep-cot-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of conversations (with detected CoT) to keep as "
            "<think>...</think> + answer (default: 1.0). The rest have "
            "reasoning stripped, keeping only the answer."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for CoT keep/drop (default: 0)"
    )
    args = parser.parse_args()

    if not (0.0 <= args.keep_cot_ratio <= 1.0):
        raise SystemExit("--keep-cot-ratio must be between 0.0 and 1.0")

    rng = random.Random(args.seed)
    stats = {
        "read": 0,
        "written": 0,
        "dropped_malformed": 0,
        "dropped_unparseable": 0,
        "kept_cot": 0,
        "stripped_cot": 0,
        "transformed_turns": 0,
        "sysprompt_rewrites": 0,
    }

    with open(args.input, "r", encoding="utf-8") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats["dropped_unparseable"] += 1
                continue
            new_obj = process_line(obj, args.keep_cot_ratio, rng, stats)
            if new_obj is None:
                continue
            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            stats["written"] += 1

    print("Reformat complete:")
    for key in (
        "read",
        "written",
        "kept_cot",
        "stripped_cot",
        "transformed_turns",
        "sysprompt_rewrites",
        "dropped_malformed",
        "dropped_unparseable",
    ):
        print(f"  {key:20s}: {stats[key]}")


if __name__ == "__main__":
    main()
