# pretrain a general model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir general-minweight-his --train-data-dir ./data/train/minweight/general --training-epochs 10 --training-mode SL  --use-cuda --T 1 --max-flow-num 1000000

# Finetune for each topology and traffic model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir minweight_his_GEANT_real_USL --model-load-dir general-minweight-his --train-data-dir ./data/train/minweight/GEANT_real_unlabel --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir minweight_his_Cogentco_traffic_burst_USL --model-load-dir general-minweight-his --train-data-dir ./data/train/minweight/Cogentco_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000


CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir minweight_his_Kdl_traffic_burst_USL --model-load-dir general-minweight-his --train-data-dir ./data/train/minweight/Kdl_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir minweight_his_500_0_traffic_burst_USL --model-load-dir general-minweight-his --train-data-dir ./data/train/minweight/500_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000


CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_weight --init-te-solution his --log-dir minweight_his_1000_0_traffic_burst_USL --model-load-dir general-minweight-his --train-data-dir ./data/train/minweight/1000_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000



# Final test minweight
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_weight --init-te-solution his --use-cuda    --model-load-dir minweight_his_GEANT_real_USL --test-data-dir ./data/test/minweight/GEANT_real   --T 1 --max-flow-num 10000000  --alpha 10 > test_minweight_GEANT_real_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_weight --init-te-solution his --use-cuda    --model-load-dir minweight_his_Cogentco_traffic_burst_USL --test-data-dir ./data/test/minweight/Cogentco_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minweight_Cogentco_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_weight --init-te-solution his --use-cuda    --model-load-dir minweight_his_Kdl_traffic_burst_USL --test-data-dir ./data/test/minweight/Kdl_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minweight_Kdl_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_weight --init-te-solution his --use-cuda    --model-load-dir minweight_his_500_0_traffic_burst_USL --test-data-dir ./data/test/minweight/500_0_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minweight_500_0_traffic_burst_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_weight --init-te-solution his --use-cuda    --model-load-dir minweight_his_1000_0_traffic_burst_USL --test-data-dir ./data/test/minweight/1000_0_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minweight_1000_0_traffic_burst_USL_a10.log