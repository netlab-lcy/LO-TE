# LO-TE
Shooting Large-scale Traffic Engineering by Combining Deep Learning and Optimization Approach (CoNEXT'25)

This is a Pytorch implementation of [FERN](https://ieeexplore.ieee.org/abstract/document/10285729) on TON 2023. 

## Generating training and testing by yourself

## Training and Evaluating the deep learning model of FERN

* Train FERN

```
cd NN-model

# Model pretrain over small scale topologies
python3 train.py --mode classify --train-data-dir MCF_BRITE_small_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 500 --log-dir pretrain_small

# pretrain for general classification model
python3 train.py --mode classify --train-data-dir MCF_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 1000 --log-dir pretrain_calssify --model-load-dir pretrain_small
# pretrain for general regression model
python3 train.py --mode normal --train-data-dir MCF_train --valid-data-dir MCF_BRITE_small_train --use-cuda --training-epochs 1000 --log-dir pretrain_normal --model-load-dir pretrain_small

# P2 training for large-scale topologies, build up a dataset for a specific large-scale topology (e.g., MCF_DialtelecomCz_test)
python3 train.py --mode classify --train-data-dir MCF_DialtelecomCz_test --valid-data-dir MCF_DialtelecomCz_test --use-cuda --training-epochs 10 --log-dir P2_DialtelecomCz  --part-failure --model-load-dir pretrain_classify --batch-size 1
```

* Evaluate FERN

```
# test general classification model
python3 eval.py --mode classify --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --log-dir pretrain_classify 

# test general regression model
python3 eval.py --mode normal --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --log-dir pretrain_normal 

# test general classification model for tripple failures
python3 eval.py --mode classify --eval-data-dir MCF_tripple_test --use-cuda  --log-dir pretrain_classify --failure-type tripple
```

* Detect critical failure scenarios with FERN

```
# detect the critical failure scenarios
python3 detect_critical_failure.py  --eval-data-dir MCF_BRITE_small_vary_capa_test1 --use-cuda  --detect-classify-model-dir pretrain_classify --detect-normal-model-dir pretrain_normal
```

We have uploaded the [pretrained model and related training and testing data](https://1drv.ms/f/c/e7bd018766776d46/EunzQphn-_JLmpyzXjQD4EUBM99uleykfrXBHZ82MeggqA?e=EdSxzN) for FERN.

## Apply FERN for robust network design use cases

* Apply the predicted critical failure set for network upgrade

```
cp -r ./NN-model/failure_set/* ./fault-tolerance-network/network_upgrade/failure_set/
cd ./fault-tolerance-network/network_upgrade
python3 main.py
```

* Apply the predicted critical set for fault-tolerant traffic engineering

```
cp -r ./NN-model/failure_set/* ./fault-tolerance-network/R3/failure_set/
cd ./fault-tolerance-network/R3
python3 main.py
```

If you have any questions, please post an issue or send an email to chenyiliu9@gmail.com.

## Citation

```
@article{liu2023fern,
  title={FERN: Leveraging Graph Attention Networks for Failure Evaluation and Robust Network Design},
  author={Liu, Chenyi and Aggarwal, Vaneet and Lan, Tian and Geng, Nan and Yang, Yuan and Xu, Mingwei and Li, Qing},
  journal={IEEE/ACM Transactions on Networking},
  year={2023},
  publisher={IEEE}
}
```