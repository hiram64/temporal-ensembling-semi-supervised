## Temporal Ensembling (Keras)

This repository provides keras implementation of the paper "TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING" by S. Laine et al.

The implementation includes Temporal Ensembling and PI-Model using CIFAR-10. Both methods are proposed in the paper
As the paper, the semi-supervised training is done by 4000 supervised data(400 per class) and 46,000 unsupervised data.
10,000 data are left for evaluation.


## Dependencies
keras, tensorflow, scikit-learn

The versions of test environment :  
Keras==2.1.6, tensorflow-gpu==1.7.0, scikit-learn==0.19.1

## How to use

#### 1. Prepare data

Prepare data and labels to use. For instance, CIFAR-10 is composed of 10 classes and each label should express unique class and be integer. These prepared data should be placed in the data directory.

You can download CIFAR-10 data via :  
https://www.kaggle.com/janzenliu/cifar-10-batches-py

Put them in "data" directory and run the following code to compress them into NPZ file.

```
python make_cifar10_npz.py
```

After running this code, you can get cifar10.npz under "data" directory.

#### 2. Train & Evaluation
After data is prepared, run the following script to train and evaluation.

Please look into the script about other settable parameters or run "python main.py --help". Although the most of this implementation follows the description of the paper, there are some differences. Please see the note below.

```
# Temporal Ensembling
python main_temporal_ensembling.py

# PI-model
python main_pi_model.py
```
Evaluation is done at intervals of 5 epochs.
In the test, Temporal ensembling and PI-model achieved about 87.3% accuracy and about 86.7% accuracy respectively at the end of epoch.

### Note
The differences between the paper and this implementation:
- The learning rate is changed to 0.001 instead of 0.003 because of non-convergence issue
- The training epoch is changed to 350 instead of 300 to achieve higher accuracy


## Reference
* Paper  
 Samuli Laine and Timo Aila : TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING, ICLR, 2017.  
 https://arxiv.org/pdf/1610.02242.pdf

* Official implementation in Theano(lasagne) by the authors  
https://github.com/smlaine2/tempens
