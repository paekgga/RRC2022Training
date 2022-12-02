python main.py \
--name "TD3BC_SM" \
--env "trifinger-cube-push-real-expert-v0" \
--alpha 2.5 \
--std_k 0.2 \
--lamd_s 0.1

python main.py \
--name "TD3BC_SM" \
--env "trifinger-cube-push-real-mixed-v0" \
--alpha 2.5 \
--std_k 0.2 \
--lamd_s 0.1

python main.py \
--name "TD3BC_SM" \
--env "trifinger-cube-lift-real-expert-v0" \
--alpha 2.5 \
--std_k 0.2 \
--lamd_s 0.1

python main.py \
--name "TD3BC_SM" \
--env "trifinger-cube-lift-real-mixed-v0" \
--alpha 2.5 \
--std_k 0.2 \
--lamd_s 0.1