import numpy as np
import pytest

from src.explain import (
    feature_family,
    permutation_baseline,
    random_reason_baseline,
    top_reasons,
    typology_hit_rate,
)


def test_top_reasons_are_the_largest_positive_contributions():
    contrib = np.array([[0.5, -2.0, 0.1, 0.9], [-0.1, -0.2, -0.3, 0.0]])
    names = ["a", "b", "c", "d"]
    assert top_reasons(contrib, names, k=2) == [["d", "a"], []]


def test_features_belong_to_families():
    assert feature_family("src_out_uniq_24h") == "fan_out"
    assert feature_family("dst_in_cnt_1h") == "fan_in"
    assert feature_family("cycle3_before") == "loop"
    assert feature_family("paths2_before") == "parallel_paths"
    assert feature_family("mins_since_src_in") == "pass_through"
    assert feature_family("fmt") == "other"


def test_hit_when_any_top_reason_matches_the_typology():
    reasons = [["src_out_uniq_24h", "fmt"], ["fmt", "hour"], ["dst_in_cnt_24h"]]
    typologies = ["FAN-OUT", "FAN-IN", "FAN-IN"]
    assert typology_hit_rate(reasons, typologies) == pytest.approx(2 / 3)


def test_permutation_baseline_is_lower_when_reasons_track_the_true_pattern():
    reasons = [["src_out_uniq_24h"]] * 50 + [["dst_in_uniq_24h"]] * 50
    typologies = ["FAN-OUT"] * 50 + ["FAN-IN"] * 50
    observed = typology_hit_rate(reasons, typologies)
    mean, p = permutation_baseline(reasons, typologies, n_perm=200, seed=0)
    assert observed == 1.0
    assert 0.35 < mean < 0.65
    assert p < 0.01


def test_random_reason_baseline_counts_the_chance_of_drawing_a_matching_feature():
    features = ["src_out_cnt_24h", "src_out_uniq_24h", "hour", "fmt"]  # 2 of 4 are fan-out
    # P(at least one fan-out feature in 1 random draw) = 2/4
    assert random_reason_baseline(["FAN-OUT"], features, k=1) == pytest.approx(0.5)
    # Two draws without replacement: 1 - C(2,2)/C(4,2) = 1 - 1/6
    assert random_reason_baseline(["FAN-OUT"], features, k=2) == pytest.approx(5 / 6)
