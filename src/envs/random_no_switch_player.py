"""poke-env Player: uniform random legal actions except voluntary switching."""

from __future__ import annotations

import random

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.player.player import Player


class RandomNoSwitchPlayer(Player):
    """Random singles policy that never switches when a move is available.

    When ``force_switch`` is true or there are no usable moves, switches are
    still chosen uniformly at random among legal switch orders.
    """

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, DoubleBattle):
            return Player.choose_random_move(battle)
        if not isinstance(battle, Battle):
            return Player.choose_random_move(battle)

        orders = battle.valid_orders
        if not orders:
            return Player.choose_default_move()

        move_orders = [
            o
            for o in orders
            if isinstance(o, SingleBattleOrder) and isinstance(o.order, Move)
        ]
        if move_orders:
            return move_orders[int(random.random() * len(move_orders))]

        switch_orders = [
            o
            for o in orders
            if isinstance(o, SingleBattleOrder) and isinstance(o.order, Pokemon)
        ]
        if switch_orders:
            return switch_orders[int(random.random() * len(switch_orders))]

        return orders[int(random.random() * len(orders))]
