## Why do we need a fishing relevance classifier?
Although we search Youtube with "fishing" as the query term we are not guaranteed to get only fishing related videos. We get videos about persons who like to fish, a property near lake ideal for fishing, guys fishing for girls etc.. So a lot of videos that were posted on our site were not really fishing related one's. A simple script which looks for terms like ["fishing", "salmon", "musky" etc..... ] didn't work well. Hence the classifier.

The classifier was written using convolution neural network using tensorflow. We took the code for this [awesome blog](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) and tried to change few hyper-parameters and get good accuracy. The original code is a sentiment classifier which can also be used for fishing relevance classification. 

## Dataset
So this classifier need atleast a decent dataset to train on, like 10000 videos manually tagged. Why just 10000? Thats what I have seen in most of entry level classification tasks. So I manually watched 10000 videos (or looked at the title, description and youtube tags) and marked them 
* 1 if its fishing related video.
* 0 irrelevant video.
* 2 Kind of relevant video.
* 9 if I have no idea what to do.

This dataset is on table `geoparser.youtube_tag_dataset`. It can also be found in below mentioned github project.

## Code and Training
The complete code for this and the processed dataset can be found in this repo. This blog and some related blog do a good job at explaining the code. I set up a google cloud vm instance with Tesla K80 GPU to run this classifier. This can be done in cpu machine too but its take really long time. Plus just signing up on google cloud gives you 400$ free credit to use. So why not?

Current accuracy of the model is around 91%-92%. If we have more data this can be improved. 




It is slightly simplified implementation of Kim's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) paper in Tensorflow.

## Requirements

- Python 3
- Tensorflow > 0.12
- Numpy

## Training

Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement

```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --eval_train --checkpoint_dir="./runs/1459637919/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.


## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
