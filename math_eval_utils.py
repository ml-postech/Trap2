"""Answer extraction, normalization, and evaluation utilities for math tasks."""

import re
import torch
from tqdm.auto import tqdm


def extract_final_answer(text, dataset_name):
    """Extract the final answer from model-generated text.

    Returns the extracted answer as a string, or "" if extraction fails.
    """
    if dataset_name in ("gsm8k", "asdiv"):
        return _extract_gsm8k_answer(text)
    else:
        return _extract_last_number(text)


def _extract_boxed_answer(text):
    """Extract answer from \\boxed{...} in MATH dataset solutions."""
    # Find the last \boxed{...} occurrence
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        # Fallback: try to get the last number
        return _extract_last_number(text)

    # Get content inside the last \boxed{}
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1].strip()
    return _extract_last_number(text)


def _extract_gsm8k_answer(text):
    """Extract answer after #### in GSM8K-style responses."""
    # Look for #### pattern
    match = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if match:
        return match.group(1).strip()
    # Fallback: last number
    return _extract_last_number(text)


def _extract_last_number(text):
    """Extract the last number from text as a fallback."""
    # Match integers, decimals, negatives, and fractions
    matches = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?', text)
    if matches:
        return matches[-1].strip()
    return ""


def normalize_math_answer(answer_str):
    """Normalize a math answer string for comparison.

    Handles commas, whitespace, trailing periods, dollar signs,
    percent signs, LaTeX fractions (\frac, \tfrac, \dfrac),
    simple fractions (a/b), and numeric equivalence.
    """
    if not answer_str:
        return ""

    s = answer_str.strip()

    # Remove surrounding LaTeX math delimiters
    s = s.strip('$')
    s = s.replace('\\$', '')
    s = s.strip()

    # Remove \text{...} wrappers
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    # Remove \mathrm{...}, \mathbf{...}, etc.
    s = re.sub(r'\\math\w+\{([^}]*)\}', r'\1', s)
    # Remove \left, \right
    s = s.replace('\\left', '').replace('\\right', '')

    # Remove trailing period
    if s.endswith('.'):
        s = s[:-1]

    # Remove commas in numbers (e.g., "1,000" -> "1000")
    s = s.replace(',', '')

    # Remove spaces
    s = s.replace(' ', '')

    # Remove dollar sign, percent sign
    s = s.replace('$', '').replace('%', '').replace('\\%', '')

    # Normalize LaTeX fractions: \frac{a}{b}, \tfrac{a}{b}, \dfrac{a}{b}
    def _replace_latex_frac(match):
        num_str, den_str = match.group(1), match.group(2)
        try:
            num, den = float(num_str), float(den_str)
            if den != 0:
                return str(num / den)
        except ValueError:
            pass
        return match.group(0)

    s = re.sub(r'\\[dt]?frac\{([^}]+)\}\{([^}]+)\}', _replace_latex_frac, s)

    # Try to evaluate simple fraction a/b
    simple_frac = re.match(r'^(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)$', s)
    if simple_frac:
        try:
            num, den = float(simple_frac.group(1)), float(simple_frac.group(2))
            if den != 0:
                s = str(num / den)
        except ValueError:
            pass

    # Remove pi symbol (treat as string for symbolic answers)
    # but keep it for string comparison

    # Try to convert to float for numeric comparison
    try:
        val = float(s)
        # Round to avoid floating point artifacts (e.g., 0.33333333 vs 0.3333333)
        val = round(val, 8)
        # Normalize to remove trailing zeros: 18.0 -> 18, 3.50 -> 3.5
        if val == int(val) and abs(val) < 1e15:
            s = str(int(val))
        else:
            s = str(val)
    except ValueError:
        # Keep as-is for non-numeric answers (e.g., symbolic)
        s = s.lower()

    return s


def is_math_correct(pred_text, gold_text, dataset_name):
    """Check if a predicted answer matches the gold answer."""
    pred = normalize_math_answer(extract_final_answer(pred_text, dataset_name))
    gold = normalize_math_answer(extract_final_answer(gold_text, dataset_name))

    if not pred or not gold:
        return False

    # Exact string match
    if pred == gold:
        return True

    # Numeric equivalence as fallback (handles rounding differences)
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        return abs(pred_val - gold_val) < 1e-6
    except ValueError:
        return False


@torch.no_grad()
def evaluate_math_generation(model, tokenizer, eval_loader, max_new_tokens=256,
                              num_print_samples=10):
    """Evaluate a model on math generation tasks using exact match.

    Args:
        model: CausalLM model (possibly with LoRA)
        tokenizer: tokenizer for decoding
        eval_loader: DataLoader using eval_collate_fn
            (batches have input_ids, attention_mask, gold_answers, dataset_name)
        max_new_tokens: max tokens to generate per example
        num_print_samples: number of sample predictions to print for debugging

    Returns:
        dict with 'em' (exact match accuracy) and 'total' (number of examples)
    """
    model.eval()
    correct = 0
    total = 0
    printed = 0

    for batch in tqdm(eval_loader, desc="Evaluating math generation", leave=False):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        gold_answers = batch["gold_answers"]
        dataset_name = batch["dataset_name"]

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only the generated part (after the prompt)
        for i in range(len(gold_answers)):
            # Use attention_mask to find true prompt length (excludes left-padding)
            prompt_len = attention_mask[i].sum().item()
            # outputs[i] includes the full sequence (pad + prompt + generated)
            generated_ids = outputs[i][input_ids.shape[1]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            match = is_math_correct(pred_text, gold_answers[i], dataset_name)
            if match:
                correct += 1
            total += 1

            if printed < num_print_samples:
                pred_ans = normalize_math_answer(extract_final_answer(pred_text, dataset_name))
                gold_ans = normalize_math_answer(extract_final_answer(gold_answers[i], dataset_name))
                status = "OK" if match else "WRONG"
                print(f"\n[SAMPLE {printed+1}] {status}")
                print(f"  gold_answer: {gold_answers[i][:200]}")
                print(f"  gold_extracted: {gold_ans}")
                print(f"  pred_text: {pred_text[:300]}")
                print(f"  pred_extracted: {pred_ans}")
                printed += 1

    em = correct / total if total > 0 else 0.0
    return {"em": em, "correct": correct, "total": total}


@torch.no_grad()
def evaluate_math_loss(model, eval_loader):
    """Evaluate average CLM loss on math eval set (fast, no generation).

    Uses the train collator format (with labels) for loss computation.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # Count non-masked tokens for proper averaging
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return avg_loss
