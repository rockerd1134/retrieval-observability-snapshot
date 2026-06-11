from retrieval_arena.scoring import score_pair, tokenize


def test_tokenize_handles_case_punctuation_and_underscores():
    assert tokenize("Hello, CACHE_size!") == ["hello", "cache_size"]


def test_perfect_score():
    score = score_pair("Install with pip", "Install with pip")
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f1"] == 1.0
    assert score["lexical_overlap"] == 1.0
    assert score["match"] is True


def test_empty_prediction_scores_zero():
    score = score_pair("", "Install with pip")
    assert score["precision"] == 0.0
    assert score["recall"] == 0.0
    assert score["f1"] == 0.0
    assert score["match"] is False


def test_repeated_tokens_use_multiset_overlap():
    score = score_pair("pip pip install", "pip install")
    assert score["precision"] == 2 / 3
    assert score["recall"] == 1.0


def test_match_threshold_behavior():
    assert score_pair("alpha beta", "alpha gamma", match_threshold=0.5)["match"] is True
    assert score_pair("alpha beta", "alpha gamma", match_threshold=0.6)["match"] is False


def test_empty_reference_and_prediction_overlap_is_one_but_no_match_by_default():
    score = score_pair("", "")
    assert score["lexical_overlap"] == 1.0
    assert score["f1"] == 0.0