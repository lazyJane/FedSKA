nohup python -u  main.py  --dataset shakespeare --algorithm Fedlocal --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/E=1/Fedlocal.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedAvg --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > ./acc_loss_record/shakespeare/E=1/FedAvg.out 2>&1 &
h
nohup python -u  main.py  --dataset shakespeare --algorithm FedProx --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 0 > ./acc_loss_record/shakespeare/E=1/FedProx.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm Ditto --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/shakespeare/E=1/Ditto.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/E=1/IFCA.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedU --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 1 > ./acc_loss_record/shakespeare/E=1/FedU.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedSFL --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 2 > ./acc_loss_record/shakespeare/E=1/FedSFL.out 2>&1 &

nohup python -u  main.py  --dataset shakespeare --algorithm FedIFCA --batch_size 32 \
--num_users 10 --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu 4 > ./acc_loss_record/shakespeare/E=1/IFCA.out 2>&1 &



