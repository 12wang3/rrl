# Rule-based Representation Learner
This is a PyTorch implementation of Rule-based Representation Learner (RRL) as described in NeurIPS 2021 paper:
[Scalable Rule-Based Representation Learning for Interpretable Classification](https://arxiv.org/abs/2109.15103).
<p align="center">
  <img src="appendix/RRL.png" alt="drawing" width="500"/>
</p>
RRL aims to obtain both good scalability and interpretability, and it automatically learns interpretable non-fuzzy rules for data representation and classification. Moreover, RRL can be easily adjusted to obtain a trade-off between classification accuracy and model complexity for different scenarios.
## Requirements

* torch>=1.3.0
* torchvision>=0.4.1
* tensorboard>=2.0.0
* sklearn>=0.22.2.post1
* numpy>=1.17.2
* pandas>=0.24.2
* matplotlib>=3.0.3
* CUDA==10.1

## Run the demo
We need to put the data sets in the `dataset` folder. You can specify one data set in the `dataset` folder and train the model as follows:

```bash
# trained on the tic-tac-toe data set with one GPU.
python3 experiment.py -d tic-tac-toe -bs 32 -s 1@16 -e401 -lrde 200 -lr 0.002 -ki 0 -mp 12481 -i 0 -wd 1e-6 &
```
The demo reads the data set and data set information first, then trains the RRL on the training set. 
During the training, you can check the training loss and the evaluation result on the validation set by:

```bash
tensorboard --logdir=log_folder/ --bind_all
```
<p align="center">
  <img src="appendix/TensorBoard.png" alt="drawing" width="500"/>
</p>

The training log file (`log.txt`) can be found in a folder created in `log_folder`. In this example, the folder path is 
```
log_folder/tic-tac-toe/tic-tac-toe_e401_bs32_lr0.002_lrdr0.75_lrde200_wd1e-06_ki0_rc0_useNOTFalse_saveBestFalse_estimatedGradFalse_L1@16
```
After training, the evaluation result on the test set is shown in the file `test_res.txt`:
```
[INFO] - On Test Set:
        Accuracy of RRL  Model: 1.0
        F1 Score of RRL  Model: 1.0
```

Moreover, the trained RRL model is saved in `model.pth`, and the discrete RRL is printed in `rrl.txt`:

|RID|class_negative(b=-2.1733)|class_positive(b=1.9689)|Support|Rule|
| ---- | ---- | ---- | ---- | ---- |
|(-1, 1)|-5.8271|6.3045|0.0885|3_x & 6_x & 9_x|
|(-1, 2)|-5.4949|5.4566|0.0781|7_x & 8_x & 9_x|
|(-1, 4)|-4.5605|4.7578|0.1146|1_x & 2_x & 3_x|
| ......| ...... | ...... | ...... | ...... |


#### Your own data sets

You can use the demo to train RRL on your own data set by putting the data and data information files in the `dataset` folder. Please read [DataSetDesc](dataset/README.md) for a more specific guideline.

#### Available arguments
List all the available arguments and their default values by:
```bash
$ python3 experiment.py --help
usage: experiment.py [-h] [-d DATA_SET] [-i DEVICE_IDS] [-nr NR] [-e EPOCH]
                     [-bs BATCH_SIZE] [-lr LEARNING_RATE]
                     [-lrdr LR_DECAY_RATE] [-lrde LR_DECAY_EPOCH]
                     [-wd WEIGHT_DECAY] [-ki ITH_KFOLD] [-rc ROUND_COUNT]
                     [-ma MASTER_ADDRESS] [-mp MASTER_PORT] [-li LOG_ITER]
                     [--use_not] [--save_best] [--estimated_grad]
                     [-s STRUCTURE]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_SET, --data_set DATA_SET
                        Set the data set for training. All the data sets in
                        the dataset folder are available. (default: tic-tac-
                        toe)
  -i DEVICE_IDS, --device_ids DEVICE_IDS
                        Set the device (GPU ids). Split by @. E.g., 0@2@3.
                        (default: None)
  -nr NR, --nr NR       ranking within the nodes (default: 0)
  -e EPOCH, --epoch EPOCH
                        Set the total epoch. (default: 41)
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Set the batch size. (default: 64)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Set the initial learning rate. (default: 0.01)
  -lrdr LR_DECAY_RATE, --lr_decay_rate LR_DECAY_RATE
                        Set the learning rate decay rate. (default: 0.75)
  -lrde LR_DECAY_EPOCH, --lr_decay_epoch LR_DECAY_EPOCH
                        Set the learning rate decay epoch. (default: 10)
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        Set the weight decay (L2 penalty). (default: 0.0)
  -ki ITH_KFOLD, --ith_kfold ITH_KFOLD
                        Do the i-th 5-fold validation, 0 <= ki < 5. (default:
                        0)
  -rc ROUND_COUNT, --round_count ROUND_COUNT
                        Count the round of experiments. (default: 0)
  -ma MASTER_ADDRESS, --master_address MASTER_ADDRESS
                        Set the master address. (default: 127.0.0.1)
  -mp MASTER_PORT, --master_port MASTER_PORT
                        Set the master port. (default: 12345)
  -li LOG_ITER, --log_iter LOG_ITER
                        The number of iterations (batches) to log once.
                        (default: 50)
  --use_not             Use the NOT (~) operator in logical rules. It will
                        enhance model capability but make the RRL more
                        complex. (default: False)
  --save_best           Save the model with best performance on the validation
                        set. (default: False)
  --estimated_grad      Use estimated gradient. (default: False)
  -s STRUCTURE, --structure STRUCTURE
                        Set the number of nodes in the binarization layer and
                        logical layers. E.g., 10@64, 10@64@32@16. (default:
                        5@64)
```
## Citation

If our work is helpful to you, please kindly cite our paper as:

```
@article{wang2021scalable,
  title={Scalable Rule-Based Representation Learning for Interpretable Classification},
  author={Wang, Zhuo and Zhang, Wei and Liu, Ning and Wang, Jianyong},
  journal={arXiv preprint arXiv:2109.15103},
  year={2021}
}
```

## License

[MIT license](LICENSE)