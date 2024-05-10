nohup python -u main.py --dataset METR-LA --algorithm Fedlocal --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 1 > ./acc_loss_record/metr-la/E=1/Fedlocal.out 2>&1 &

nohup python -u main.py --dataset METR-LA --algorithm FedAvg --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/metr-la/E=1/FedIFCA.out 2>&1 &

nohup python -u main.py --dataset METR-LA --algorithm FedProx --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/metr-la/E=1/FedProx.out 2>&1 &

nohup python -u main.py --dataset METR-LA --algorithm Ditto --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 1 > ./acc_loss_record/metr-la/E=1/Ditto.out 2>&1 &
 
nohup python -u main.py --dataset METR-LA --algorithm FedIFCA --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 4 > ./acc_loss_record/metr-la/E=1/FedAvg.out 2>&1 &
  
nohup python -u main.py --dataset METR-LA --algorithm FedU --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 2 > ./acc_loss_record/metr-la/E=1/FedU.out 2>&1 &

nohup python -u main.py --dataset METR-LA --algorithm FedSFL --batch_size 64 \
--num_users 207 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 1 > ./acc_loss_record/metr-la/E=1/FedSFL.out 2>&1 &

nohup python -u main.py --dataset METR-LA --algorithm FedSKA --batch_size 64 \
--num_users 207 --tau 0.9 --k 30 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/metr-la/E=1/FedSKA.out 2>&1 &
 
 

 