# Shooting Large-scale Traffic Engineering by Combining Deep Learning and Optimization Approach

This is a Pytorch implementation of [LO-TE](https://doi.org/10.1145/3709372) presented on CoNEXT 2025. LO-TE provides a two-step approach to efficiently resolve large-scale traffic engineering problems: obtaining an initial solution and refining it to achieve a near-optimal TE solution.

## Getting started

### Hardware requirements

* **CPU**: 16+ cores (more cores recommended for parallel training with multiple samples).
* **Memory**: 64+ GB RAM (512+ GB recommended for larger topologies).
* **GPU**: 8+ GB memory (reduce `--max-flow-num` to lower GPU memory usage).
* **Operating System**: Linux (tested on Ubuntu 20.04 and 22.04).

### Dependencies

* Run `pip install -r requirements.txt` to install Python dependencies

* Install gurobipy and get a gurobi license [here](https://www.gurobi.com/academia/academic-program-and-licenses/).

### Download traning and testing data

We have uploaded the [training and testing data](https://1drv.ms/f/c/e7bd018766776d46/Ejx2Bqa0V0xPrtSZboyit_cB-05GU92vHvqcYwKU3jRPTw?e=sC04G3) in LO-TE paper, including topology information, traffic demands, candidate paths， and labeled/unlabeled data for training and testing.

We recommend downloading the necessary data and copying it into the ./data directory before running the program.


## Training and Testing LO-TE

We have listed the training and testing commands in `run_minmlu.sh`, `run_maxthrpt.sh`, and `run_minweight.sh`. We show an example of traning and testing LO-TE on Cogentco topology and traffic burst model below: 

* Pretrain a general model with supervised learning 

```
python3 train.py --objective min_mlu --init-te-solution his --log-dir general-minmlu-his --train-data-dir ./data/train/minmlu/general --training-epochs 10 --training-mode SL  --use-cuda --T 1 --max-flow-num 1000000
```

* Unsupervised learning for a specific topology and traffic model

```
python3 train.py --objective min_mlu --init-te-solution his --log-dir minmlu_his_Cogentco_traffic_burst_USL --model-load-dir general-minmlu-his --train-data-dir ./data/train/minmlu/Cogentco_traffic_burst_unlabel100 --training-epochs 50 --training-mode USL  --use-cuda --T 1  --num-sample-process 5 --K 5 --alpha 1 --max-flow-num 1000000
```

* Testing LO-TE 

```
python3 test.py --objective min_mlu --init-te-solution his --use-cuda    --model-load-dir minmlu_his_Cogentco_traffic_burst_USL --test-data-dir ./data/test/minmlu/Cogentco_traffic_burst   --T 1 --max-flow-num 10000000  --alpha 10
```

## Generating training and testing data by yourself

We have also provided the major data generation codes for generating your own training and testing data. Refer to `./data_generation` for more details. 



If you have any questions, please post an issue or send an email to chenyiliu9@gmail.com.

## Citation

```
@inproceedings{liu2025lote,
  title={Shooting Large-scale Traffic Engineering by Combining Deep Learning and Optimization Approach},
  author={Liu, Chenyi and Deng, Haotian and Aggarwal, Vaneet and Yang, Yuan and Xu, Mingwei },
  booktitle=={Proceedings of the ACM CoNEXT 2025 Conference},
  year={2025},
  publisher={ACM}
}
```