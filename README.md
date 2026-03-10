# DSPRO2

## Setting up the environment

### Required tools

- Python 3.13 (use uv for local virtual environment with `uv venv .venv --python 3.13` and `source .venv/bin/activate`)
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

## Data

### Game trainer data

The game trainer data is stored in the `data/game_trainer_data` directory.

Source is from: https://docs.google.com/spreadsheets/d/1_uRpnFWroeCY3RaRi4lXeS9uZEDXiRIMBtTQVGrIZ3w/edit?gid=773204596#gid=773204596