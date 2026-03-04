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

