RATE=${1:-0.5}
shift

echo " Running experiment with compression_rate=${RATE}"

python train.py --compression_rate ${RATE} "$@"

#./run.sh 0.x --device x --seed x