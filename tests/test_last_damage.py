from src.envs.battle_env import PokemonBattleEnv
from tests.fakes import FakeBattle, garchomp_like, jolteon_like


def test_last_damage_cache_records_hp_drop():
    env = PokemonBattleEnv.__new__(PokemonBattleEnv)
    env._hp_snapshot = {}
    env._last_damage = {}
    our = garchomp_like()
    opp = jolteon_like()
    battle = FakeBattle(our, opp)
    env._update_last_damage(battle, "b1")
    assert env._last_damage == {}
    our.current_hp_fraction = 0.7
    opp.current_hp_fraction = 0.5
    env._update_last_damage(battle, "b1")
    dealt, taken = env._last_damage["b1"]
    assert abs(dealt - 0.5) < 1e-6
    assert abs(taken - 0.3) < 1e-6
