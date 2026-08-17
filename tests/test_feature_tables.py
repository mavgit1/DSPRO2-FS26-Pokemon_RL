from poke_env.battle.move import Move

from src.models.feature_tables import infer_tags_from_move, item_class_vector, tags_for_move


def test_recovery_and_setup_and_pivot_tags():
    assert "recovery" in infer_tags_from_move(Move("softboiled", gen=8))
    assert "setup" in infer_tags_from_move(Move("swordsdance", gen=8))
    assert "pivot" in infer_tags_from_move(Move("uturn", gen=8))
    assert "pivot" in infer_tags_from_move(Move("voltswitch", gen=8))
    assert "protect" in infer_tags_from_move(Move("protect", gen=8))
    assert "priority" in infer_tags_from_move(Move("aquajet", gen=8))
    assert "hazard" in infer_tags_from_move(Move("stealthrock", gen=8))
    assert "hazard_control" in tags_for_move(Move("rapidspin", gen=8))
    assert "contact" in infer_tags_from_move(Move("closecombat", gen=8))


def test_item_class_vector():
    leftovers = item_class_vector("Leftovers")
    assert leftovers[1] == 1.0
    assert sum(leftovers) == 1.0
    choice = item_class_vector("choiceband")
    assert choice[0] == 1.0
    unknown = item_class_vector("unknown_item")
    assert sum(unknown) == 0.0
    empty = item_class_vector(None)
    assert sum(empty) == 0.0
