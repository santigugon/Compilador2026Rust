from __future__ import annotations

import unittest

from pipeline.runner_core import (
    build_tritonbench_item_summary,
    tritonbench_op_filename,
)


class TritonBenchScoringTest(unittest.TestCase):
    def test_infers_filename_from_wrapper_entry_information(self) -> None:
        item = {
            "instruction": (
                "Functional Description: ... "
                "Wrapper Entry Information: torch.linalg.cholesky(A, *, upper=False)"
            )
        }

        self.assertEqual(tritonbench_op_filename(item), "cholesky.py")

    def test_scores_only_the_matching_operation(self) -> None:
        eval_summary = {
            "per_op": {
                "add.py": {
                    "call_acc_passed": True,
                    "exe_acc_passed": True,
                }
            }
        }
        item = {
            "instruction": (
                "Functional Description: ... "
                "Wrapper Entry Information: cholesky(A, *, upper=False)"
            )
        }

        summary, op_filename = build_tritonbench_item_summary(
            item=item,
            has_code=True,
            output_profile="tritonbench_python",
            eval_summary=eval_summary,
            perf_ok=True,
        )

        self.assertEqual(op_filename, "cholesky.py")
        self.assertFalse(summary["compiled"])
        self.assertFalse(summary["ran_successfully"])
        self.assertFalse(summary["correctness_passed"])
        self.assertFalse(summary["overall_passed"])
        self.assertEqual(summary["failure_stage"], "compile")


if __name__ == "__main__":
    unittest.main()
