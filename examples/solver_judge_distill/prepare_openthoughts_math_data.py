"""Prepare OpenThoughts math dataset for OPSD training.

OpenThoughts-114k-math provides full chain-of-thought solutions, which OPSD
requires for effective teacher conditioning (vs. Polaris which only has final answers).
"""

from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.math_utils.utils import extract_answer


def prepare_openthoughts_math_data():
    """Load OpenThoughts-114k-math and AIME test sets for OPSD training."""
    # OpenThoughts-114k-math: problem + full solution (chain-of-thought)
    train_dataset = load_dataset("open-r1/OpenThoughts-114k-math", split="train")

    # AIME for validation
    test_dataset1 = load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_dataset2 = load_dataset("MathArena/aime_2025", split="train")

    def preprocess_openthoughts(example, idx):
        """Convert OpenThoughts format to solver-judge expected format."""
        problem = example["problem"]
        solution = example.get("solution", "")
        # Extract final answer from solution for reward evaluation
        ground_truth = extract_answer(solution) if solution else None
        if ground_truth is None:
            ground_truth = ""
        return {
            "idx": idx,
            "question": problem,
            "ground_truth": ground_truth,
            "solution": solution or "",  # Full chain-of-thought for OPSD teacher conditioning
            "data_source": "math",
        }

    def preprocess_aime(example, idx):
        """Convert AIME format (no full solution, just answer)."""
        return {
            "idx": idx,
            "question": example["problem"],
            "ground_truth": str(example["answer"]),
            "data_source": "math",
        }

    train_dataset = train_dataset.map(
        preprocess_openthoughts, with_indices=True, remove_columns=train_dataset.column_names
    )
    # Filter out examples without valid solution (needed for OPSD teacher conditioning)
    train_dataset = train_dataset.filter(lambda x: bool(x.get("solution", "").strip()))
    test_dataset1 = test_dataset1.map(
        preprocess_aime, with_indices=True, remove_columns=test_dataset1.column_names
    )
    test_dataset2 = test_dataset2.map(
        preprocess_aime, with_indices=True, remove_columns=test_dataset2.column_names
    )
    test_dataset = concatenate_datasets([test_dataset1, test_dataset2])

    # Register as solver_judge_math (same name as Polaris) so OPSD training can load it
    train_dataset = DatasetRegistry.register_dataset(
        "solver_judge_math", train_dataset, "train"
    )
    test_dataset = DatasetRegistry.register_dataset(
        "solver_judge_math", test_dataset, "test"
    )

    print(f"OpenThoughts train dataset size: {len(train_dataset)}")
    print(f"AIME test dataset size: {len(test_dataset)}")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_openthoughts_math_data()
    print("Train dataset path:", train_dataset.get_data_path())
    print("Test dataset path:", test_dataset.get_data_path())

    print("\nSample train example (with solution for OPSD):")
    sample = train_dataset[0]
    print(f"  question (first 200 chars): {sample['question'][:200]}...")
    print(f"  ground_truth: {sample['ground_truth']}")
    print(f"  solution (first 300 chars): {sample.get('solution', '')[:300]}...")
