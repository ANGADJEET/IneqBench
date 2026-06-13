"""
51_ineqmath_full_benchmark.py — Complete IneqMath Benchmark (Presentation 2)
=============================================================================
Robust version: avoids DSPy JSON parsing for fragile outputs and normalizes
bad keys like {">=": ">="} into {"relation": ">="}.
"""

import json
import re
import dspy
from nim_config import get_dspy_lm2

# Use NIM via DSPy LM client, but do NOT rely on DSPy structured parsing.
lm = get_dspy_lm2(temperature=0.0)
judge_lm = get_dspy_lm2(temperature=0.1)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def call_text(model, prompt: str) -> str:
    resp = model(prompt=prompt)

    # 🔥 HANDLE LIST RESPONSES FROM LITELLM / NIM
    if isinstance(resp, list):
        return " ".join(
            r.get("text", "") if isinstance(r, dict) else str(r)
            for r in resp
        )

    if isinstance(resp, dict):
        return resp.get("text", str(resp))

    return str(resp)


def extract_json(text: str):
    """Extract the first JSON object from messy model output."""
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # First {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Repair common malformed relation outputs like {"reasoning":"...","<=":"<="}
        candidate = re.sub(
            r'"(>=|<=|>|<|=)"\s*:\s*"\1"',
            r'"relation": "\1"',
            candidate,
        )
        candidate = re.sub(
            r'"(>=|<=|>|<|=)"\s*:',
            r'"relation":',
            candidate,
        )
        candidate = candidate.replace("':", '"').replace('":', '"')

        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


def normalize_relation(parsed: dict):
    """Turn weird keys like '>=', '<=', '>' into relation."""
    if not isinstance(parsed, dict):
        return parsed

    if "relation" in parsed:
        return parsed

    for k in list(parsed.keys()):
        if k in {">=", "<=", ">", "<", "="}:
            parsed["relation"] = parsed[k] if parsed[k] in {">=", "<=", ">", "<", "="} else k
            break

    return parsed


def ask_json(model, prompt: str, fallback: dict):
    """Ask for JSON and recover from minor formatting errors."""
    text = call_text(model, prompt)
    parsed = extract_json(text)

    if parsed is None:
        return fallback, text

    parsed = normalize_relation(parsed)
    return parsed, text


def yes_no(text: str) -> bool:
    t = text.lower()
    if "true" in t and "false" not in t:
        return True
    if "false" in t and "true" not in t:
        return False
    # fallback heuristic
    return "yes" in t and "no" not in t


# ═══════════════════════════════════════════════════════════════════
# 1. Subtasks
# ═══════════════════════════════════════════════════════════════════

def solve_bound_estimation(inequality: str):
    prompt = f"""
Return ONLY valid JSON.

Task: Find the optimal constant C in the inequality.

Inequality:
{inequality}

Required JSON schema:
{{
  "reasoning": "step-by-step derivation",
  "optimal_constant": "the optimal constant as a string",
  "proof": "rigorous proof"
}}

Rules:
- output JSON only
- do not add markdown
- do not add extra keys
"""
    fallback = {
        "reasoning": "Parsing failed",
        "optimal_constant": "ERROR",
        "proof": "Parsing failed",
    }
    parsed, raw = ask_json(lm, prompt, fallback)

    if "optimal_constant" not in parsed:
        parsed["optimal_constant"] = "ERROR"
    if "reasoning" not in parsed:
        parsed["reasoning"] = "Parsing failed"
    if "proof" not in parsed:
        parsed["proof"] = "Parsing failed"

    return parsed, raw


def solve_relation_prediction(expression_left: str, expression_right: str, conditions: str):
    prompt = f"""
Return ONLY valid JSON.

Task: Predict the correct relation between two expressions.

Left:
{expression_left}

Right:
{expression_right}

Conditions:
{conditions}

Required JSON schema:
{{
  "reasoning": "step-by-step reasoning",
  "relation": "one of >, >=, =, <=, <"
}}

Rules:
- output JSON only
- if the model wants to emit the symbol as a key, do NOT do that
- put the symbol in the value of the key "relation"
- do not add markdown
"""
    fallback = {
        "reasoning": "Parsing failed",
        "relation": "ERROR",
    }
    parsed, raw = ask_json(lm, prompt, fallback)

    if "relation" not in parsed:
        # last-resort regex from the raw text
        m = re.search(r"(>=|<=|>|<|=)", raw)
        parsed["relation"] = m.group(1) if m else "ERROR"

    if "reasoning" not in parsed:
        parsed["reasoning"] = "Parsing failed"

    return parsed, raw


# ═══════════════════════════════════════════════════════════════════
# 2. Proof generation
# ═══════════════════════════════════════════════════════════════════

def theorem_guided_proof(inequality: str, guiding_theorems=None):
    if guiding_theorems is None:
        guiding_theorems = "AM-GM, Cauchy-Schwarz, Jensen's, Schur's, Power Mean"

    prompt = f"""
You are a mathematician.

Prove the inequality rigorously and concisely.

Inequality:
{inequality}

Helpful theorems:
{guiding_theorems}

Return only the proof text. No JSON. No markdown headings.
"""
    return call_text(lm, prompt)


def self_critique_proof(inequality: str, max_iters: int = 2):
    proof = theorem_guided_proof(inequality)

    for _ in range(max_iters):
        critique_prompt = f"""
You are a strict mathematical critic.

Question:
{inequality}

Proof:
{proof}

Return only JSON:
{{
  "critique": "specific feedback",
  "needs_revision": "True or False"
}}
"""
        crit, _ = ask_json(
            judge_lm,
            critique_prompt,
            {"critique": "Parsing failed", "needs_revision": "False"},
        )

        if str(crit.get("needs_revision", "False")).lower().find("false") != -1:
            break

        revise_prompt = f"""
You are an expert mathematician.

Question:
{inequality}

Current proof:
{proof}

Critique:
{crit.get("critique", "")}

Rewrite the proof incorporating the critique. Return only the revised proof text.
"""
        proof = call_text(lm, revise_prompt)

    return proof


# ═══════════════════════════════════════════════════════════════════
# 3. Judges
# ═══════════════════════════════════════════════════════════════════

ERROR_TAXONOMY = {
    "final_answer": {"count": 0, "examples": []},
    "toy_case": {"count": 0, "examples": []},
    "logical_gap": {"count": 0, "examples": []},
    "numerical_approx": {"count": 0, "examples": []},
    "computation": {"count": 0, "examples": []},
}


def run_5_judge_pipeline(target: str, proof: str, verbose: bool = True) -> dict:
    results = {}
    all_pass = True

    # Judge 1: Final Answer
    p1 = f"""
You are Judge 1.

Target:
{target}

Proof:
{proof}

Return only JSON:
{{"is_correct":"True or False"}}
"""
    j1, _ = ask_json(judge_lm, p1, {"is_correct": "False"})
    results["final_answer"] = yes_no(str(j1.get("is_correct", "False")))
    if not results["final_answer"]:
        ERROR_TAXONOMY["final_answer"]["count"] += 1
        all_pass = False
    if verbose:
        print(f"  {'✅' if results['final_answer'] else '❌'} Final Answer Judge")

    # Judge 2: Toy Case
    p2 = f"""
You are Judge 2.

Proof:
{proof}

Check whether the proof survives simple toy values like a=1, b=2, c=3 where relevant.
Return only JSON:
{{"is_correct":"True or False"}}
"""
    j2, _ = ask_json(judge_lm, p2, {"is_correct": "False"})
    results["toy_case"] = yes_no(str(j2.get("is_correct", "False")))
    if not results["toy_case"]:
        ERROR_TAXONOMY["toy_case"]["count"] += 1
        all_pass = False
    if verbose:
        print(f"  {'✅' if results['toy_case'] else '❌'} Toy Case Judge")

    # Judge 3: Logical Gap
    p3 = f"""
You are Judge 3.

Proof:
{proof}

Check for missing transitions or unjustified logical leaps.
Return only JSON:
{{
  "is_correct":"True or False",
  "gap_description":"..."
}}
"""
    j3, _ = ask_json(
        judge_lm,
        p3,
        {"is_correct": "False", "gap_description": "Parsing failed"},
    )
    results["logical_gap"] = yes_no(str(j3.get("is_correct", "False")))
    results["gap_details"] = j3.get("gap_description", "")
    if not results["logical_gap"]:
        ERROR_TAXONOMY["logical_gap"]["count"] += 1
        all_pass = False
    if verbose:
        print(f"  {'✅' if results['logical_gap'] else '❌'} Logical Gap Judge: {results['gap_details'][:80]}")

    # Judge 4: Numerical Approximation
    p4 = f"""
You are Judge 4.

Proof:
{proof}

Check for invalid decimal approximations or non-rigorous numerical substitutions.
Return only JSON:
{{
  "is_correct":"True or False",
  "approximation_issues":"..."
}}
"""
    j4, _ = ask_json(
        judge_lm,
        p4,
        {"is_correct": "False", "approximation_issues": "Parsing failed"},
    )
    results["numerical_approx"] = yes_no(str(j4.get("is_correct", "False")))
    results["approximation_issues"] = j4.get("approximation_issues", "")
    if not results["numerical_approx"]:
        ERROR_TAXONOMY["numerical_approx"]["count"] += 1
        all_pass = False
    if verbose:
        print(f"  {'✅' if results['numerical_approx'] else '❌'} Numerical Approximation Judge")

    # Judge 5: Computation
    p5 = f"""
You are Judge 5.

Proof:
{proof}

Check symbolic/numeric computations.
Return only JSON:
{{
  "is_correct":"True or False",
  "verification_code":"..."
}}
"""
    j5, _ = ask_json(
        judge_lm,
        p5,
        {"is_correct": "False", "verification_code": "Parsing failed"},
    )
    results["computation"] = yes_no(str(j5.get("is_correct", "False")))
    results["verification_code"] = j5.get("verification_code", "")
    if not results["computation"]:
        ERROR_TAXONOMY["computation"]["count"] += 1
        all_pass = False
    if verbose:
        print(f"  {'✅' if results['computation'] else '❌'} Computation Judge")

    results["all_pass"] = all_pass
    results["answer_accuracy"] = results["final_answer"]
    results["overall_soundness"] = all_pass
    return results


# ═══════════════════════════════════════════════════════════════════
# 4. Benchmark
# ═══════════════════════════════════════════════════════════════════

BENCHMARK_PROBLEMS = [
    {
        "type": "bound_estimation",
        "inequality": "For positive reals a,b: a^2 + b^2 >= C*a*b. Find optimal C.",
        "expected_constant": "2",
    },
    {
        "type": "relation_prediction",
        "left": "(a+b+c)/3",
        "right": "(abc)^(1/3)",
        "conditions": "positive reals a,b,c",
        "expected_relation": ">=",
    },
    {
        "type": "proof",
        "inequality": "AM-GM: For positive reals a,b: (a+b)/2 >= sqrt(ab)",
        "target": "AM-GM Inequality",
    },
    {
        "type": "proof",
        "inequality": "Cauchy-Schwarz: (a1^2+a2^2)(b1^2+b2^2) >= (a1*b1+a2*b2)^2",
        "target": "Cauchy-Schwarz Inequality",
    },
]


def run_benchmark():
    results_summary = {"total": 0, "answer_correct": 0, "fully_sound": 0}

    for i, problem in enumerate(BENCHMARK_PROBLEMS):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}: {problem.get('inequality', problem.get('left', ''))}")
        print(f"{'='*60}")

        if problem["type"] == "bound_estimation":
            result, raw = solve_bound_estimation(problem["inequality"])
            print(f"\nReasoning:\n{result.get('reasoning', '')}")
            print(f"\nOptimal C = {result.get('optimal_constant', 'ERROR')}")
            print(f"Expected  = {problem['expected_constant']}")
            print(f"\nProof:\n{result.get('proof', '')}")
            results_summary["total"] += 1

        elif problem["type"] == "relation_prediction":
            result, raw = solve_relation_prediction(
                problem["left"],
                problem["right"],
                problem["conditions"],
            )
            print(f"\nRelation: {result.get('relation', 'ERROR')}")
            print(f"Expected: {problem['expected_relation']}")
            results_summary["total"] += 1

        elif problem["type"] == "proof":
            proof = theorem_guided_proof(problem["inequality"])
            print(f"\nProof:\n{proof[:300]}...\n")
            judges = run_5_judge_pipeline(problem["target"], proof)
            results_summary["total"] += 1
            if judges["answer_accuracy"]:
                results_summary["answer_correct"] += 1
            if judges["overall_soundness"]:
                results_summary["fully_sound"] += 1

    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Total problems:     {results_summary['total']}")
    print(f"  Answer Accuracy:    {results_summary['answer_correct']}")
    print(f"  Overall Soundness:  {results_summary['fully_sound']}")

    print(f"\n  Error Taxonomy:")
    for error_type, data in ERROR_TAXONOMY.items():
        if data["count"] > 0:
            print(f"    {error_type}: {data['count']} failures")

    return results_summary


if __name__ == "__main__":
    print("=" * 60)
    print("COMPLETE IneqMath BENCHMARK (Presentation 2)")
    print("=" * 60)
    run_benchmark()