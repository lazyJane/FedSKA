nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm Fedlocal --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/Fedlocal.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedAvg --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedAvg.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedProx --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 2 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedProx.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm Ditto --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/Ditto.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedIFCA --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedIFCA.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedU --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 4 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedU.out 2>&1 &

nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedSFL --batch_size 32 \
--num_users 100 --learning_rate 0.01 --num_glob_iters 200 --E 1 --times 1 --gpu 0 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedSFL.out 2>&1 &
 
nohup python -u main.py --dataset Cifar10-allocation_shards2-ratio0.2-u100 --algorithm FedSKA \
--batch_size 32 --tau 0.9 --k 98 --feature_hidden_size 64 --beta 0.01 --num_users 100   --learning_rate 0.01 --num_glob_iters 500 --E 1 --times 1 --gpu -1 > ./acc_loss_record/cifar10/ratio0.2/E=1/shards2/FedSKA.out 2>&1 &