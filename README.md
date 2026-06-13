# IneqBench: LLM Mathematical Reasoning Benchmark

IneqBench is a lightweight, DSPy-based evaluation framework designed to benchmark Large Language Models on their mathematical inequality reasoning capabilities. 

Rather than simply asking models for an answer and string-matching the final result, IneqBench evaluates the *rigor* of mathematical proofs through an automated, multi-agent critique and judgment pipeline.

## Features

- **Automated Proof Generation & Self-Critique:** Uses DSPy to generate initial proofs guided by well-known theorems (AM-GM, Cauchy-Schwarz, Jensen's, Schur's), followed by a multi-iteration self-critique loop to revise and refine logical arguments.
- **Robust 5-Judge Verification Pipeline:** Evaluates proofs based on five distinct dimensions:
  1. **Final Answer:** Is the ultimate conclusion correct?
  2. **Toy Case Check:** Does the proof hold up when tested with simple numerical substitutions (e.g., $a=1, b=2$)?
  3. **Logical Gap Detection:** Are there missing transitions or unjustified leaps in logic?
  4. **Numerical Approximation:** Does the proof rely on invalid decimal approximations instead of rigorous symbolic logic?
  5. **Computational Soundness:** Are the symbolic/numeric derivations structurally correct?
- **Robust JSON Extraction:** Built-in safeguards to parse messy or malformed JSON outputs from LLMs, automatically normalizing broken relation keys (`>=`, `<=`, etc.).
- **Error Taxonomy:** Automatically categorizes failure modes into a taxonomy for detailed model diagnostics.

## Evaluation Subtasks

The benchmark is broken down into three primary subtasks:
1. **Bound Estimation:** Finding the optimal constant $C$ in a given inequality format.
2. **Relation Prediction:** Predicting the correct relational operator (`>`, `>=`, `=`, `<=`, `<`) between two expressions given a set of conditions.
3. **Proof Verification:** Generating a rigorous proof and running it through the 5-Judge Pipeline.

## Usage

**Prerequisites:**
- Python 3.8+
- `dspy-ai`
- Any valid LM provider integrated via your `nim_config.py` (e.g., litellm).

**Running the benchmark:**
Simply execute the script to run the default set of seed problems:
```bash
python 51_inequality_benchmark.py
```

## How It Works

1. **Proof Generation:** The `lm` generates a proof.
2. **Self-Critique:** The `judge_lm` provides specific feedback and determines if revision is needed. The `lm` rewrites the proof based on this feedback.
3. **Judging:** The final proof is passed to 5 independent judge prompts.
4. **Scoring:** The pipeline tracks "Answer Accuracy" and "Overall Soundness" (meaning it passed all 5 judges).

## Extending IneqBench

You can easily extend the framework to arbitrary problem sets by modifying the `BENCHMARK_PROBLEMS` list at the bottom of the script. Support for integrating dynamic prompt optimization via DSPy's teleprompters (like MIPRO/GEPA) can be layered natively over the core evaluation functions.
