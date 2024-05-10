nohup python -u  main.py  --dataset shakespeare --algorithm FedAvg --batch_size 32 \
--num_users 5 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > ./acc_loss_record/shakespeare/GRU/FedAvg.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedAvg --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > ./acc_loss_record/shakespeare/GRU/FedAvg.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm Fedlocal --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/Fedlocal.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedProx --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/FedProxGRU.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedSFL --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/shakespeare/GRU/FedSFL.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm Ditto --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/shakespeare/GRU/Ditto.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm Ditto --batch_size 32 \
--num_users 1000 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/Ditto.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 1000 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/GRU/IFCA.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/shakespeare/GRU/IFCA.out 2>&1 &


nohup python -u  main.py  --dataset shakespeare --algorithm FedAMP --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 5 > ./acc_loss_record/shakespeare/GRU/FedAMP.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedU --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/GRU/FedU.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0 --k 30 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau0_k30.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0 --k 98  --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau0_k98.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0.9 --k 30 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau9_k30.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0.9 --k 98 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 3 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau9_k98.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0.91 --k 30 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau91_k30.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0.9 --k 42 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau9_k42.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0 --k 42 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 3 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau0_k42_small.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature_shakespeare --batch_size 32 \
--num_users 100 --tau 0 --k 42 --feature_hidden_size 256 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/GRU/FedCNFGNN_gcnadj_mask_feature_shakespeare_tau0_k42.out 2>&1 &

python main.py --dataset shakespeare --algorithm FedCNFGNN_gcnadj_mask_feature --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0

python main.py  --dataset shakespeare --algorithm FedAvg --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 

python main.py --dataset shakespeare --algorithm FedKLEM --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm PerFedavg --batch_size 32 \
--num_users 112 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare  --algorithm FedProx --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1

python main.py --dataset shakespeare --algorithm Fedlocal --batch_size 32 \
--num_users 20 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1



