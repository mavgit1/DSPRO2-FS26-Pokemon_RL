"""
Trainer Teams
============
Opponent teams for training at different difficulty levels.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TrainerTeam:
    """Represents a trainer's team."""
    name: str
    difficulty: str  # easy, medium, hard, elite
    format: str      # gen8ou, gen8randombattle, etc.
    team_export: str  # Showdown export format
    description: str = ""


# =============================================================================
# EASY TEAMS (Beginner AI, simple strategies)
# =============================================================================

EASY_TEAMS: List[TrainerTeam] = [
    TrainerTeam(
        name="Electric Squad",
        difficulty="easy",
        format="gen8ou",
        description="Simple electric-type team with basic moves",
        team_export="""
Pikachu (M) @ Light Ball
Ability: Static
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Thunderbolt
- Quick Attack
- Iron Tail
- Volt Switch

Raichu (M) @ Focus Sash
Ability: Lightning Rod
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Psychic
- Nasty Plot
- Focus Blast

Electabuzz (M) @ Eviolite
Ability: Vital Spirit
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Psychic
- Focus Blast
- Ice Punch

Jolteon (M) @ Choice Specs
Ability: Quick Feet
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Shadow Ball
- Hidden Power [Ice]
- Volt Switch

Magneton @ Eviolite
Ability: Sturdy
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Thunderbolt
- Flash Cannon
- Hidden Power [Fire]
- Volt Switch

Zapdos @ Heavy-Duty Boots
Ability: Static
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Heat Wave
- Hurricane
- Roost
""",
    ),
    
    TrainerTeam(
        name="Starter Squad",
        difficulty="easy",
        format="gen8ou",
        description="The classic Gen 1 starters",
        team_export="""
Charizard (M) @ Life Orb
Ability: Blaze
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Flamethrower
- Air Slash
- Dragon Pulse
- Solar Beam

Blastoise (M) @ White Herb
Ability: Torrent
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Surf
- Ice Beam
- Shell Smash
- Hidden Power [Electric]

Venusaur (M) @ Black Sludge
Ability: Overgrow
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Giga Drain
- Sludge Bomb
- Sleep Powder
- Growth
""",
    ),
    
    TrainerTeam(
        name="Bug Catcher",
        difficulty="easy",
        format="gen8ou",
        description="Bug-type team with basic setup",
        team_export="""
Scizor (M) @ Choice Band
Ability: Technician
EVs: 248 HP / 252 Atk / 8 SpD
Adamant Nature
- Bullet Punch
- U-turn
- Knock Off
- Superpower

Volcarona (M) @ Heavy-Duty Boots
Ability: Flame Body
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Fire Blast
- Bug Buzz
- Quiver Dance
- Giga Drain

Frosmoth (F) @ Heavy-Duty Boots
Ability: Ice Scales
EVs: 252 SpA / 4 SpD / 252 Spe
Modest Nature
- Bug Buzz
- Ice Beam
- Quiver Dance
- Hurricane

Vikavolt (M) @ Choice Specs
Ability: Levitate
EVs: 248 HP / 252 SpA / 8 SpD
Modest Nature
- Thunderbolt
- Bug Buzz
- Hidden Power [Fire]
- Energy Ball
""",
    ),
]


# =============================================================================
# MEDIUM TEAMS (Gym Leader level)
# =============================================================================

MEDIUM_TEAMS: List[TrainerTeam] = [
    TrainerTeam(
        name="Gym Leaders United",
        difficulty="medium",
        format="gen8ou",
        description="A team featuring Pokemon used by famous Gym Leaders",
        team_export="""
Garchomp (M) @ Choice Scarf
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Outrage
- Earthquake
- Stone Edge
- U-turn

Ferrothorn (M) @ Leftovers
Ability: Iron Barbs
EVs: 252 HP / 88 Def / 168 SpD
Relaxed Nature
- Leech Seed
- Stealth Rock
- Gyro Ball
- Power Whip

Togekiss (F) @ Leftovers
Ability: Serene Grace
EVs: 248 HP / 8 SpA / 252 SpD
Calm Nature
- Air Slash
- Thunder Wave
- Nasty Plot
- Roost

Heatran (M) @ Air Balloon
Ability: Flash Fire
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Fire Blast
- Earth Power
- Stealth Rock
- Taunt

Gyarados (M) @ Leftovers
Ability: Intimidate
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Waterfall
- Earthquake
- Ice Fang
- Dragon Dance

Scizor (M) @ Choice Band
Ability: Technician
EVs: 248 HP / 252 Atk / 8 SpD
Adamant Nature
- Bullet Punch
- U-turn
- Knock Off
- Superpower
""",
    ),
    
    TrainerTeam(
        name="Weather Warriors",
        difficulty="medium",
        format="gen8ou",
        description="Weather-based team with Rain and Sun cores",
        team_export="""
Pelipper (M) @ Damp Rock
Ability: Drizzle
EVs: 248 HP / 252 Def / 8 SpA
Bold Nature
- Scald
- Hurricane
- U-turn
- Roost

Ferrothorn (M) @ Leftovers
Ability: Iron Barbs
EVs: 252 HP / 88 Def / 168 SpD
Relaxed Nature
- Leech Seed
- Stealth Rock
- Gyro Ball
- Power Whip

Tornadus-Therian (M) @ Leftovers
Ability: Regenerator
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Hurricane
- Heat Wave
- U-turn
- Taunt

Barraskewda (M) @ Choice Band
Ability: Swift Swim
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Liquidation
- Close Combat
- Psychic Fangs
- Aqua Jet

Toxapex (F) @ Black Sludge
Ability: Regenerator
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Scald
- Recover
- Haze
- Toxic Spikes

Thundurus-Therian (M) @ Choice Specs
Ability: Volt Absorb
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Focus Blast
- Hidden Power [Ice]
- Volt Switch
""",
    ),
]


# =============================================================================
# HARD TEAMS (Competitive OU)
# =============================================================================

HARD_TEAMS: List[TrainerTeam] = [
    TrainerTeam(
        name="OU Standard",
        difficulty="hard",
        format="gen8ou",
        description="Standard competitive OU team",
        team_export="""
Landorus-Therian (M) @ Rocky Helmet
Ability: Intimidate
EVs: 252 HP / 216 Def / 40 SpD
Impish Nature
- Earthquake
- U-turn
- Stealth Rock
- Defog

Tapu Koko @ Choice Specs
Ability: Electric Surge
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Dazzling Gleam
- Hidden Power [Ice]
- Volt Switch

Heatran (M) @ Heavy-Duty Boots
Ability: Flash Fire
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
- Fire Blast
- Earth Power
- Stealth Rock
- Taunt

Dragapult (M) @ Choice Specs
Ability: Clear Body
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- Flamethrower
- U-turn

Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 248 HP / 80 Def / 180 SpD
Impish Nature
- Brave Bird
- Body Press
- Roost
- U-turn

Clefable (F) @ Leftovers
Ability: Magic Guard
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Moonblast
- Thunder Wave
- Wish
- Teleport
""",
    ),
    
    TrainerTeam(
        name="Hyper Offense",
        difficulty="hard",
        format="gen8ou",
        description="Aggressive hyper offense team",
        team_export="""
Dragapult (M) @ Heavy-Duty Boots
Ability: Clear Body
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Dragon Darts
- Phantom Force
- U-turn
- Hex

Garchomp (M) @ Rocky Helmet
Ability: Rough Skin
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Earthquake
- Outrage
- Stealth Rock

Hawlucha (M) @ Electric Seed
Ability: Unburden
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Acrobatics
- High Jump Kick
- Swords Dance
- Roost

Crawdaunt (M) @ Choice Band
Ability: Adaptability
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Crabhammer
- Knock Off
- Aqua Jet
- Close Combat

Weavile (M) @ Heavy-Duty Boots
Ability: Pressure
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Icicle Crash
- Knock Off
- Low Kick
- Ice Shard

Alakazam (M) @ Focus Sash
Ability: Magic Guard
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Psychic
- Shadow Ball
- Focus Blast
- Nasty Plot
""",
    ),
]


# =============================================================================
# ELITE TEAMS (Championship level)
# =============================================================================

ELITE_TEAMS: List[TrainerTeam] = [
    TrainerTeam(
        name="VGC Champion",
        difficulty="elite",
        format="gen8vgc2022",
        description="Championship-level VGC team",
        team_export="""
Calyrex-Ice @ Heavy-Duty Boots
Ability: As One
EVs: 252 Atk / 252 SpA / 4 SpD
Quiet Nature
- Glacial Lance
- Astral Barrage
- High Horsepower
- Leech Seed

Incineroar (M) @ Assault Vest
Ability: Intimidate
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Flare Blitz
- Knock Off
- U-turn
- Fake Out

Rillaboom (M) @ Assault Vest
Ability: Grassy Surge
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Grassy Glide
- Fake Out
- High Horsepower
- U-turn

Tapu Fini @ Leftovers
Ability: Misty Surge
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Moonblast
- Scald
- Calm Mind
- Taunt

Regieleki @ Life Orb
Ability: Transistor
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Thunderbolt
- Electroweb
- Rapid Spin
- Ancient Power

Glastrier @ Weakness Policy
Ability: Chilling Neigh
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Icicle Crash
- Close Combat
- Heavy Slam
- Protect
""",
    ),
]


# =============================================================================
# TEAM ACCESS FUNCTIONS
# =============================================================================

DIFFICULTY_TEAMS: Dict[str, List[TrainerTeam]] = {
    "easy": EASY_TEAMS,
    "medium": MEDIUM_TEAMS,
    "hard": HARD_TEAMS,
    "elite": ELITE_TEAMS,
}


def get_teams(difficulty: str) -> List[str]:
    """Get team exports for a difficulty level."""
    teams = DIFFICULTY_TEAMS.get(difficulty.lower(), [])
    return [t.team_export for t in teams]


def get_random_team(difficulty: Optional[str] = None) -> str:
    """Get a random team export."""
    import random
    
    if difficulty:
        teams = DIFFICULTY_TEAMS.get(difficulty.lower(), [])
    else:
        teams = []
        for t_list in DIFFICULTY_TEAMS.values():
            teams.extend(t_list)
    
    if teams:
        return random.choice(teams).team_export
    return ""


def get_team_names(difficulty: Optional[str] = None) -> List[str]:
    """Get team names for a difficulty level."""
    if difficulty:
        teams = DIFFICULTY_TEAMS.get(difficulty.lower(), [])
    else:
        teams = []
        for t_list in DIFFICULTY_TEAMS.values():
            teams.extend(t_list)
    return [t.name for t in teams]


def list_all_teams() -> str:
    """List all teams for display."""
    lines = []
    for diff, teams in DIFFICULTY_TEAMS.items():
        lines.append(f"\n=== {diff.upper()} ===")
        for team in teams:
            lines.append(f"  - {team.name}: {team.description}")
    return "\n".join(lines)