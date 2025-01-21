# pretrain a general model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir test --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/1500_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 0 --K 1 --alpha 5 --max-flow-num 1000000

# Finetune for each topology and traffic model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_GEANT_real_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/GEANT_real_unlabel --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_Cogentco_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/Cogentco_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
# CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_Cogentco_hose_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/Cogentco_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000


CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_Kdl_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/Kdl_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
# CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_Kdl_hose_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/Kdl_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_500_0_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/500_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
# CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_500_0_hose_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/500_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000


CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_1000_0_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/1000_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
# CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_1000_0_hose_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/1000_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000


CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_1500_0_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/1500_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
# CUDA_VISIBLE_DEVICES=0 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_1500_0_hose_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/1500_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=1 python3 train.py --objective max_throughput --init-te-solution his --log-dir maxthrpt_his_ASN2k_traffic_burst_USL --model-load-dir general-maxthrpt-his --train-data-dir ./data/train/maxthrpt/ASN2k_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1 --head-num 1 --num-sample-process 5 --K 5 --alpha 0.5 --max-flow-num 300000


# Final test maxthrpt
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_GEANT_real_USL --test-data-dir ./data/test/maxthrpt/GEANT_real  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_GEANT_real_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_Cogentco_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/Cogentco_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_Cogentco_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_Kdl_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/Kdl_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_Kdl_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_500_0_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/500_0_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_500_0_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_1000_0_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/1000_0_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_1000_0_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_1500_0_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/1500_0_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_1500_0_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective max_throughput --init-te-solution his --use-cuda  --head-num 1  --model-load-dir maxthrpt_his_ASN2k_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/ASN2k_traffic_burst  --K 1  --T 1 --num-sample-process 0 --max-flow-num 10000000  --alpha 10 > test_maxthrpt_ASN2k_traffic_burst_USL_a10.log