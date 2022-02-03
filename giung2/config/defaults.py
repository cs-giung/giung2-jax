from giung2.config import CfgNode


_C = CfgNode()

_C.MODEL = CfgNode()
_C.MODEL.META_ARCHITECTURE = CfgNode()
_C.MODEL.META_ARCHITECTURE.NAME = 'ImageClassificationModelBase'

# preprocessing for ImageClassificationModelBase
_C.MODEL.PIXEL_MEAN = [0.0, 0.0, 0.0,]
_C.MODEL.PIXEL_STD  = [1.0, 1.0, 1.0,]

# ---------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------- #
_C.DATASETS = CfgNode()
_C.DATASETS.NAME = 'CIFAR10'

# root directory that contains datasets
_C.DATASETS.ROOT = './datasets/'

# random seed for SHUFFLE_INDICES
_C.DATASETS.SEED = 42

# MNIST, FashionMNIST
_C.DATASETS.MNIST = CfgNode()
_C.DATASETS.MNIST.SHUFFLE_INDICES = False
_C.DATASETS.MNIST.TRAIN_INDICES = [0, 50000,]
_C.DATASETS.MNIST.VALID_INDICES = [50000, 60000,]
_C.DATASETS.MNIST.DATA_AUGMENTATION = 'STANDARD_TRAIN_TRANSFORM'

# CIFAR10, CIFAR100
_C.DATASETS.CIFAR = CfgNode()
_C.DATASETS.CIFAR.SHUFFLE_INDICES = False
_C.DATASETS.CIFAR.TRAIN_INDICES = [0, 45000,]
_C.DATASETS.CIFAR.VALID_INDICES = [45000, 50000,]
_C.DATASETS.CIFAR.DATA_AUGMENTATION = 'STANDARD_TRAIN_TRANSFORM'

# TinyImageNet200
_C.DATASETS.TINY = CfgNode()
_C.DATASETS.TINY.SHUFFLE_INDICES = False
_C.DATASETS.TINY.TRAIN_INDICES = [0, 90000,]
_C.DATASETS.TINY.VALID_INDICES = [90000, 100000,]
_C.DATASETS.TINY.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# ImageNet1k
_C.DATASETS.IMAGENET = CfgNode()
_C.DATASETS.IMAGENET.ROOT = './datasets/ILSVRC2012/'
_C.DATASETS.IMAGENET.SHUFFLE_INDICES = True
_C.DATASETS.IMAGENET.TRAIN_INDICES = [0, 1231167,]
_C.DATASETS.IMAGENET.VALID_INDICES = [1231167, 1281167,]
_C.DATASETS.IMAGENET.DATA_AUGMENTATION = "STANDARD_TRAIN_TRANSFORM"

# ---------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CfgNode()
_C.MODEL.BACKBONE.NAME = 'PreResNet'

# ResNet, PreResNet
_C.MODEL.BACKBONE.RESNET = CfgNode()
_C.MODEL.BACKBONE.RESNET.IN_PLANES = 16
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK = CfgNode()
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_NORM_LAYER = False
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_ACTIVATION = False
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.USE_POOL_LAYER = False
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.CONV_KSP = [3, 1, 'SAME',]
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.POOL_KSP = [3, 2, 'VALID',]
_C.MODEL.BACKBONE.RESNET.BLOCK = 'BasicBlock'
_C.MODEL.BACKBONE.RESNET.SHORTCUT = 'ProjectionShortcut'
_C.MODEL.BACKBONE.RESNET.NUM_BLOCKS = [4, 4, 4,]
_C.MODEL.BACKBONE.RESNET.WIDEN_FACTOR = 1
_C.MODEL.BACKBONE.RESNET.CONV_LAYERS = 'Conv2d'
_C.MODEL.BACKBONE.RESNET.NORM_LAYERS = 'BatchNorm2d'
_C.MODEL.BACKBONE.RESNET.ACTIVATIONS = 'ReLU'

# ---------------------------------------------------------------------- #
# Classifier
# ---------------------------------------------------------------------- #
_C.MODEL.CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.NAME = 'SoftmaxClassifier'

# SoftmaxClassifier
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER = CfgNode()
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_CLASSES = 10
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.NUM_HEADS = 1
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.USE_BIAS = True
_C.MODEL.CLASSIFIER.SOFTMAX_CLASSIFIER.LINEAR_LAYERS = 'Linear'

# ---------------------------------------------------------------------- #
# Batch Normalization
# ---------------------------------------------------------------------- #
_C.MODEL.BATCH_NORMALIZATION = CfgNode()
_C.MODEL.BATCH_NORMALIZATION.MOMENTUM = 0.9
_C.MODEL.BATCH_NORMALIZATION.EPSILON = 1e-5

# ---------------------------------------------------------------------- #
# Filter Response Normalization
# ---------------------------------------------------------------------- #
_C.MODEL.FILTER_RESPONSE_NORMALIZATION = CfgNode()
_C.MODEL.FILTER_RESPONSE_NORMALIZATION.EPSILON = 1e-6
_C.MODEL.FILTER_RESPONSE_NORMALIZATION.USE_LEARNABLE_EPSILON = False

# ---------------------------------------------------------------------- #
# BatchEnsemble
# ---------------------------------------------------------------------- #
_C.MODEL.BATCH_ENSEMBLE = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.ENABLED = False

# the size of ensembles
_C.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE = 4

# initialization of rank-one factors
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.NAME = "normal"
_C.MODEL.BATCH_ENSEMBLE.ALPHA_INITIALIZER.VALUES = [1.0, 0.5,]
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.NAME = "normal"
_C.MODEL.BATCH_ENSEMBLE.GAMMA_INITIALIZER.VALUES = [1.0, 0.5,]
