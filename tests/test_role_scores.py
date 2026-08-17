from src.models.role_scores import ROLE_SCORE_NAMES, compute_role_scores, estimate_stat
from tests.fakes import blissey_like, garchomp_like, jolteon_like


def test_stat_formula_matches_gen3():
    # Blissey HP at 85, 31 IV, 85 EV: floor((2*255+31+21)*85/100)+85+10
    hp = estimate_stat(255, 85, is_hp=True)
    assert 540 < hp < 590


def test_blissey_is_stall_support_not_sweeper():
    scores = dict(zip(ROLE_SCORE_NAMES, compute_role_scores(blissey_like())))
    assert scores["stall"] > 0.2
    assert scores["support"] > 0.2
    assert scores["bulky"] > scores["sweeper"]
    assert scores["sweeper"] == 0.0


def test_garchomp_setup_is_sweeper():
    scores = dict(zip(ROLE_SCORE_NAMES, compute_role_scores(garchomp_like())))
    assert scores["sweeper"] > 0.05
    assert scores["stall"] == 0.0
    assert scores["wallbreaker"] == 0.0


def test_jolteon_is_pivot_and_fast_attacker():
    scores = dict(zip(ROLE_SCORE_NAMES, compute_role_scores(jolteon_like())))
    assert scores["pivot"] > 0.0
    assert scores["fast_attacker"] > scores["sweeper"]


def test_opponent_unrevealed_moves_scale_to_zero():
    mon = blissey_like()
    mon.moves = {}
    scores = compute_role_scores(mon, is_opponent=True)
    assert float(scores.sum()) == 0.0


def test_partial_reveal_scales_scores():
    full = compute_role_scores(blissey_like(), is_opponent=False)
    quarter = compute_role_scores(blissey_like(), is_opponent=True, revealed_move_frac=0.25)
    assert float(quarter.sum()) < float(full.sum())
    assert float(quarter.max()) <= float(full.max()) + 1e-6
