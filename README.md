# DSPRO2

## Python Setup (uv)

This project is managed by [uv](https://docs.astral.sh/uv/). It handles Python 3.13 and all dependencies automatically.

### Installation
From the root directory, run:
```bash
uv sync
```
This creates a local .venv and installs the exact versions from uv.lock.

### Running Scripts
Avoid manual venv activation. Use uv run to execute scripts within the correct environment:
```bash
uv run train_battler.py
```
### Dependency Management

To add or remove libraries for the team:
```bash
uv add <package>
uv remove <package>
```
Note: Always commit uv.lock after making changes to dependencies.

## Setting up the environment

### Required additional tools

- Node.js (use nvm for local version management with `nvm install 22.12.0` and `nvm use 22.12.0`)
- npm
- git


### Setting up the server

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

The server should now be running at http://localhost:8000

### Running the server

```bash
cd pokemon-showdown
node pokemon-showdown start --no-security
```

If you want to start multiple instances of the server, you can use the `spin_up_multiple_showdown.sh` script:

```bash
./spin_up_multiple_showdown.sh
```

To stop all instances, you can use the `kill_all_showdown.sh` script:

```bash
./kill_all_showdown.sh
```

### Serer config

If the server is set up as requested above the config folder can be found at 'pokemon-showdown/config/config.js'

Check there for throttling settings and other configurations. If there are problems with the server, check there first.

## Data

### Game trainer data

The game trainer data is stored in the `data/game_trainer_data` directory.

Source is from: 
https://heystacks.com/doc/1042/bdsp-trainer-data

https://docs.google.com/spreadsheets/d/1_uRpnFWroeCY3RaRi4lXeS9uZEDXiRIMBtTQVGrIZ3w/edit?gid=773204596#gid=773204596

## Training

Currently only battle Agent is implmented and started when running the training script.
In the future there needs to be another script for training the teambuilder that uses 
a battle agent with some baseline competency.

Current train_battle.py is using a PPO algorithm and custom embedding and model (nn.Module).
There are possible multiple arguments you can pass.
All of which are subject to further changes and improvements.

### Todo
- Currently no arguments to restart trainig for a certain chekpoint/ model. 
- Create a script to get best config for current hardware
- Parse csv of all trainers from data folder as gauntlet
- Hyperparameter tuning (ONLY AFTER REWARD TUNING)
- reward config tuning

### Configs
The config for the training is in src/config.
Please create your own configs for how much your hardware can handle.
Currently mine is not tested for max performance.

### Teams
Currently small set of AI generated teams. In the src/teams/trainer_teams.py

Maybe just look for already implmented random battle generator. 
In the future for certain curriculum segments.

### Models

Currently implmented in models/battle_transformer.py as a custom version of nn.Modules.

### gym env

Environment used for passing to rllib 
src/envs/battle_env.py