# DSPRO2: Pokémon RL Battler


## Dev Environment Setup

Follow these steps exactly to set up your local development environment. You will need Node.js (for the Showdown server) and Python (for the RL agent).

### Step 1: Install `nvm` and Node.js
The Pokémon Showdown server runs on Node.js. We use `nvm` (Node Version Manager) to ensure everyone is on the same version.

1. **Install `nvm`** (Mac/Linux):
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.4/install.sh | bash
   ```
   *(Note: Restart your terminal after installing --> close and reopen VSCode).*

2. **Install and use Node v22.12.0**:
   ```bash
   nvm install 22.12.0
   nvm use 22.12.0
   ```

### Step 2: Install `uv` and Python Dependencies
This project uses [uv](https://docs.astral.sh/uv/) to manage Python 3.13 and all dependencies automatically. No manual `venv` activation is required.

1. **Install `uv`** (if you don't have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Install dependencies**:
   From the root of this project, run:
   ```bash
   uv sync
   ```
   *This creates a local `.venv` and installs the exact versions from `uv.lock` --> the file where all version dependencies are stored.*

### Step 3: Environment Variables & MLflow
We use MLflow to track training runs. 

1. Copy the example environment file:
   ```bash
   cp .env-example .env
   ```
2. Open `.env` and fill in the MLflow credentials (ask a team member for the details):
   ```ini
   MLFLOW_TRACKING_URI="https://mlflow-server-url.com"
   MLFLOW_TRACKING_USERNAME="Username"
   MLFLOW_TRACKING_PASSWORD="Password"
   ```

### Step 4: Set Up the Pokémon Showdown Server
The RL agent needs local servers to battle against.

1. Clone the server and install its dependencies:
   ```bash
   git clone https://github.com/smogon/pokemon-showdown.git
   cd pokemon-showdown
   npm install
   cp config/config-example.js config/config.js
   cd ..
   ```
   *(Note: If you encounter server throttling issues during training, check `pokemon-showdown/config/config.js` to adjust rate limits).*

---

## Running the Project

### 1. Start the Showdown Servers
Before training, you need to spin up the local Showdown instances. We have scripts to handle this:

* **Start multiple servers:**
  ```bash
  ./scripts/spin_up_multiple_showdown.sh
  ```
* **Stop all servers:**
  ```bash
  ./scripts/kill_all_showdown.sh
  ```

### 2. Start Training
Use `uv run` to execute scripts within the correct environment. The main entry point is `train_battler.py`.

```bash
uv run train_battler.py --preset quick
```
*You can pass different presets (e.g., `standard`, `optimal`, `large`) defined in `src/config/TM_optimal_config.py` depending on your hardware capabilities.*

---

##  Project Structure

* **`train_battler.py`**: The main entry point for kicking off training.
* **`src/`**: Core Python library.
  * **`config/`**: Training, hardware, and reward configurations.
  * **`envs/`**: Custom Gymnasium environments mapping RL to Pokémon Showdown (`battle_env.py`).
  * **`models/`**: Custom neural network architectures (e.g., `battle_transformer.py`).
  * **`teams/`**: AI-generated and static Pokémon teams for training.
  * **`training/`**: Ray RLlib setup, custom callbacks, and curriculum managers.
* **`scripts/`**: Executable bash scripts (server management, etc.).
* **`examples/`**: Sandboxed scripts, notebooks, and reference players (e.g., `MaxDamagePlayer.py`).
* **`data/`**: Datasets (e.g., BDSP Trainer Data CSVs).

---

## Development Guide

### Managing Dependencies
If you need to add or remove Python libraries, use `uv`:
```bash
uv add <package_name>
uv remove <package_name>
```
**Important:** Always commit `uv.lock` and `pyproject.toml` after making dependency changes so the rest of the team stays in sync!

### Hardware Configs
Training configs are located in `src/config/TM_optimal_config.py`. Create or modify presets based on what your specific machine (CPU/GPU) can handle.