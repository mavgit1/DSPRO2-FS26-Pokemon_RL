# Validation Team Manifests

Tier 2 uses a fixed paired-team benchmark:

- Generate 20 `gen8randombattle` teams once.
- Store them as 10 fixed pairs.
- For each pair, run both swapped team directions against `RandomPlayer` and
  `SimpleHeuristicsPlayer`.
- This produces 40 battles per checkpoint.

The manifest should store both inspectable structured metadata and executable
Showdown team text:

```json
{
  "metadata": {
    "generation_format": "gen8randombattle",
    "execution_format": "gen8customgame",
    "team_count": 20,
    "pair_count": 10,
    "generated_at": "YYYY-MM-DDTHH:MM:SSZ",
    "generator": "pokemon-showdown random battle generator"
  },
  "teams": [
    {
      "id": "team_01",
      "showdown": "Pikachu @ Light Ball\nAbility: Static\nLevel: 84\n- Thunderbolt\n...",
      "pokemon": [
        {
          "species": "Pikachu",
          "item": "Light Ball",
          "ability": "Static",
          "moves": ["Thunderbolt", "Surf", "Grass Knot", "Volt Switch"],
          "level": 84
        }
      ]
    }
  ],
  "pairs": [
    {
      "id": "pair_01",
      "team_a": "team_01",
      "team_b": "team_02"
    }
  ]
}
```

Expected file path for the first Tier 2 benchmark:

```bash
data/validation/gen8_random_battle_team_pairs.json
```
