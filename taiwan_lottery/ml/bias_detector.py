"""Bias detection module for lottery draw analysis.

Tests whether the lottery draws deviate from uniform randomness
using chi-square tests, runs tests, and positional bias analysis.
"""

from collections import Counter

import numpy as np
from scipy import stats as sp_stats


class BiasDetector:
    """Detect statistical biases in lottery draw history."""

    def __init__(self, max_num: int, pick_count: int):
        self.max_num = max_num
        self.pick_count = pick_count

    def chi_square_test(
        self,
        history: list[list[int]],
        segments: list[int | None] | None = None,
    ) -> dict:
        """Chi-square goodness-of-fit test for each number's frequency.

        Tests whether each number appears at a rate consistent with
        uniform distribution across different history segments.

        Args:
            history: Full draw history.
            segments: List of segment sizes to test. None means full history.

        Returns:
            Dict with per-segment results including per-number p-values.
        """
        if segments is None:
            segments = [100, 500, 1000, None]

        results = {}
        for seg in segments:
            seg_key = str(seg) if seg else "all"
            data = history[-seg:] if seg and len(history) >= seg else history
            total_draws = len(data)

            # Count occurrences of each number
            counter = Counter()
            for nums in data:
                counter.update(nums)

            observed = np.array(
                [counter.get(n, 0) for n in range(1, self.max_num + 1)],
                dtype=np.float64,
            )
            # Expected frequency under uniform distribution
            expected_per_num = total_draws * self.pick_count / self.max_num
            expected = np.full(self.max_num, expected_per_num, dtype=np.float64)

            # Overall chi-square test
            chi2, p_overall = sp_stats.chisquare(observed, f_exp=expected)

            # Per-number contribution and individual z-test p-values
            per_number = []
            for num in range(1, self.max_num + 1):
                obs = observed[num - 1]
                exp = expected[num - 1]
                # z-test for individual number
                z = (obs - exp) / np.sqrt(exp) if exp > 0 else 0
                p_num = 2 * (1 - sp_stats.norm.cdf(abs(z)))
                chi2_contrib = (obs - exp) ** 2 / exp if exp > 0 else 0
                per_number.append({
                    "number": num,
                    "observed": int(obs),
                    "expected": round(float(exp), 2),
                    "chi2_contribution": round(float(chi2_contrib), 4),
                    "z_score": round(float(z), 4),
                    "p_value": round(float(p_num), 6),
                })

            results[seg_key] = {
                "total_draws": total_draws,
                "chi2_statistic": round(float(chi2), 4),
                "p_value": round(float(p_overall), 6),
                "is_significant": bool(p_overall < 0.05),
                "per_number": per_number,
            }

        return results

    def runs_test(self, history: list[list[int]]) -> dict:
        """Wald-Wolfowitz runs test for each number's appearance sequence.

        Tests whether the sequence of appearances (1) and non-appearances (0)
        is random for each number.
        """
        results = []
        for num in range(1, self.max_num + 1):
            # Build binary sequence: 1 = appeared, 0 = not appeared
            seq = [1 if num in draw else 0 for draw in history]
            n1 = sum(seq)
            n0 = len(seq) - n1

            if n1 == 0 or n0 == 0:
                results.append({
                    "number": num,
                    "runs": 0,
                    "expected_runs": 0,
                    "z_score": 0,
                    "p_value": 1.0,
                })
                continue

            # Count runs
            runs = 1
            for i in range(1, len(seq)):
                if seq[i] != seq[i - 1]:
                    runs += 1

            # Expected runs and variance under randomness
            n = n0 + n1
            expected_runs = 1 + 2 * n0 * n1 / n
            var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))

            if var_runs > 0:
                z = (runs - expected_runs) / np.sqrt(var_runs)
                p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
            else:
                z = 0
                p = 1.0

            results.append({
                "number": num,
                "runs": runs,
                "expected_runs": round(float(expected_runs), 2),
                "z_score": round(float(z), 4),
                "p_value": round(float(p), 6),
            })

        return {
            "per_number": results,
            "significant_count": sum(1 for r in results if r["p_value"] < 0.05),
        }

    def positional_bias(self, history: list[list[int]]) -> dict:
        """Test if numbers appear uniformly across sorted positions.

        For each position in the sorted draw (1st, 2nd, ..., pick_count-th),
        tests whether all numbers are equally likely to appear there.
        """
        total = len(history)
        if total == 0:
            return {"positions": [], "significant_count": 0}

        positions = []
        for pos in range(self.pick_count):
            counter = Counter()
            for draw in history:
                sorted_draw = sorted(draw)
                if pos < len(sorted_draw):
                    counter[sorted_draw[pos]] += 1

            # For each position, test uniformity among numbers that actually appear
            observed_nums = sorted(counter.keys())
            if len(observed_nums) < 2:
                continue

            observed = np.array([counter[n] for n in observed_nums], dtype=np.float64)
            expected = np.full(len(observed_nums), total / len(observed_nums))

            chi2, p = sp_stats.chisquare(observed, f_exp=expected)

            # Top biased numbers for this position
            deviations = []
            for i, num in enumerate(observed_nums):
                dev = (observed[i] - expected[i]) / np.sqrt(expected[i])
                deviations.append({"number": num, "z_score": round(float(dev), 4)})
            deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)

            positions.append({
                "position": pos + 1,
                "chi2": round(float(chi2), 4),
                "p_value": round(float(p), 6),
                "is_significant": bool(p < 0.05),
                "top_biased": deviations[:5],
            })

        return {
            "positions": positions,
            "significant_count": sum(1 for p in positions if p["is_significant"]),
        }

    def full_report(self, history: list[list[int]]) -> dict:
        """Generate comprehensive bias detection report."""
        chi2 = self.chi_square_test(history)
        runs = self.runs_test(history)
        positional = self.positional_bias(history)

        # Collect significantly biased numbers (p < 0.05 in full-history chi-square)
        sig_biases = []
        all_key = "all"
        if all_key in chi2:
            for item in chi2[all_key]["per_number"]:
                if item["p_value"] < 0.05:
                    sig_biases.append({
                        "number": item["number"],
                        "observed": item["observed"],
                        "expected": item["expected"],
                        "z_score": item["z_score"],
                        "p_value": item["p_value"],
                        "direction": "over" if item["z_score"] > 0 else "under",
                    })

        overall_p = chi2.get(all_key, {}).get("p_value", 1.0)

        return {
            "chi_square_results": chi2,
            "significant_biases": sig_biases,
            "overall_uniformity_p": overall_p,
            "runs_test_results": runs,
            "positional_bias": positional,
            "temporal_bias": None,  # reserved for bingo temporal analysis
        }

    def get_bias_boost(self, history: list[list[int]], threshold: float = 0.01) -> np.ndarray:
        """Compute bias-based weight boost for FrequencyModel integration.

        Numbers with significant bias (p < threshold) get a multiplicative boost.
        Over-represented numbers get a slight boost; under-represented get larger boost
        (mean reversion expectation).

        Returns shape: (max_num,) — multiplicative factors (default 1.0).
        """
        boost = np.ones(self.max_num, dtype=np.float64)

        chi2 = self.chi_square_test(history, segments=[None])
        all_key = "all"
        if all_key not in chi2:
            return boost

        for item in chi2[all_key]["per_number"]:
            if item["p_value"] < threshold:
                num = item["number"]
                z = item["z_score"]
                if z > 0:
                    # Over-represented: slight boost (the bias might continue)
                    boost[num - 1] = 1.0 + min(abs(z) * 0.02, 0.1)
                else:
                    # Under-represented: larger boost (mean reversion)
                    boost[num - 1] = 1.0 + min(abs(z) * 0.03, 0.15)

        return boost
