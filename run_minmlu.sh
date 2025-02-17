# pretrain a general model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir general-minmlu-his --train-data-dir ./data/train/minmlu/general --training-epochs 10 --training-mode SL  --use-cuda --T 1 --max-flow-num 1000000

# Finetune for each topology and traffic model
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_GEANT_real_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/GEANT_real_unlabel --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_Cogentco_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/Cogentco_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_Cogentco_hose_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/Cogentco_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_Kdl_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/Kdl_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_Kdl_hose_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/Kdl_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_500_0_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/500_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_500_0_hose_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/500_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_1000_0_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/1000_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_1000_0_hose_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/1000_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_1500_0_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/1500_0_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_1500_0_hose_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/1500_0_hose_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000

CUDA_VISIBLE_DEVICES=0 python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_ASN2k_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/ASN2k_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 0.5 --max-flow-num 1000000

# Final test minmlu
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_GEANT_real_USL --test-data-dir ./data/test/minmlu/GEANT_real   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_GEANT_real_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_Cogentco_traffic_burst_USL --test-data-dir ./data/test/minmlu/Cogentco_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_Cogentco_traffic_burst_USL_a10.log
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_Cogentco_hose_USL --test-data-dir ./data/test/minmlu/Cogentco_hose   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_Cogentco_hose_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_Kdl_traffic_burst_USL --test-data-dir ./data/test/minmlu/Kdl_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_Kdl_traffic_burst_USL_a10.log
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_Kdl_hose_USL --test-data-dir ./data/test/minmlu/Kdl_hose   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_Kdl_hose_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_500_0_traffic_burst_USL --test-data-dir ./data/test/minmlu/500_0_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_500_0_traffic_burst_USL_a10.log
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_500_0_hose_USL --test-data-dir ./data/test/minmlu/500_0_hose   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_500_0_hose_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_1000_0_traffic_burst_USL --test-data-dir ./data/test/minmlu/1000_0_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_1000_0_traffic_burst_USL_a10.log
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_1000_0_hose_USL --test-data-dir ./data/test/minmlu/1000_0_hose   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_1000_0_hose_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_1500_0_traffic_burst_USL --test-data-dir ./data/test/minmlu/1500_0_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_1500_0_traffic_burst_USL_a10.log
CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_1500_0_hose_USL --test-data-dir ./data/test/minmlu/1500_0_hose   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_1500_0_hose_USL_a10.log

CUDA_VISIBLE_DEVICES=0 python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_ASN2k_traffic_burst_USL --test-data-dir ./data/test/maxthrpt/ASN2k_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10 > test_minmlu_ASN2k_traffic_burst_USL_a10.log