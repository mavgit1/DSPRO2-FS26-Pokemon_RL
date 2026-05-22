./scripts/setup_training.sh 8 #run 12 showdown servers

uv run --active train_battler.py --preset mav --num-servers 8 # start training

SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py --preset pure_league_play --num-servers 8

rm -rf checkpoints/*
rm -rf logs/*



set -a && [ -f .env ] && . ./.env && set +a
PYTHONUNBUFFERED=1 SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py \
  --preset pure_league_play \
  --num-servers 8 \
  --resume-checkpoint /home/sudome/DSPRO2-FS26-Pokemon_RL/checkpoints/final \
  --mlflow-run-id d28ea360ftmb51400b9dc3fef8fbdfb37e

./scripts/setup_training.sh 8
rm -rf checkpoints/*
rm -rf logs/*
rm train.log 
SAVE_LEAGUE_HISTORY=1 nohup uv run --active train_battler.py --preset pure_league_play --num-servers 8 >> train.log 2>&1 &

set -a && [ -f .env ] && . ./.env && set +a
SAVE_LEAGUE_HISTORY=1 uv run --active train_battler.py --preset pure_league_play --num-servers 8


  while true; do
  # 1. Get GPU stats
  GPU=$(nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total --format=csv,noheader)
  
  # 2. Get CPU usage (100 - idle percentage)
  CPU=$(top -bn1 | grep '%Cpu' | awk '{print 100 - $8}')
  
  # 3. Get RAM usage percentage
  RAM=$(free | awk '/Mem/{printf("%.1f", $3/$2 * 100)}')
  
  # 4. Stitch them together and append to the file
  echo "$GPU, $CPU%, $RAM%" >> system_log.csv
  
  # 5. Wait 60 seconds
  sleep 60
  done

for f in train_battler.py src/config/TM_optimal_config.py src/training/historical_self_player.py src/training/self_play_player.py src/action_space.py src/data/trainer_dataset_utils.py src/training/env_bridge.py src/training/callbacks.py src/training/curriculum.py src/training/trainer.py src/envs/battle_env.py ; do
  echo "--- $f ---"
  cat "$f"
  echo ""
done

tree src/ scripts/ data/ examples/ 

git diff HEAD..marv-dev > all_differences.txt
