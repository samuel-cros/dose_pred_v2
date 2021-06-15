#python train_model_v2.py -path 'res_v2/baseline_seed1' -o adam -lr 1e-3 -drop 0.0 \
#-e 200 -seed 1 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' &

#python train_model_v2.py -path 'res_v2/baseline_seed2' -o adam -lr 1e-3 -drop 0.0 \
#-e 200 -seed 2 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' &

#BACK_PID1=$!
#wait $BACK_PID1
# 

#python train_model_v2.py -path 'res_v2/baseline_seed3' -o adam -lr 1e-3 -drop 0.0 \
#-e 200 -seed 3 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' &

python train_model_v2.py -path 'res_v2/attention_seed1' -o adam -lr 1e-3 -drop 0.0 \
-e 200 -seed 1 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' -use_attention &

python train_model_v2.py -path 'res_v2/attention_seed2' -o adam -lr 1e-3 -drop 0.0 \
-e 200 -seed 2 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' -use_attention &

BACK_PID2=$!
wait $BACK_PID2
# 

python train_model_v2.py -path 'res_v2/attention_seed3' -o adam -lr 1e-3 -drop 0.0 \
-e 200 -seed 3 -ids 'data_generation/selected_ids.npy' -loss 'mse' -final_activation 'linear' -use_attention &