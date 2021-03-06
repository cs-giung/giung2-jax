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

# data augmentation for training
_C.DATASETS.DATA_AUGMENTATION = 'STANDARD'

# MNIST, FashionMNIST
_C.DATASETS.MNIST = CfgNode()
_C.DATASETS.MNIST.SHUFFLE_INDICES = False
_C.DATASETS.MNIST.TRAIN_INDICES = [0, 50000,]
_C.DATASETS.MNIST.VALID_INDICES = [50000, 60000,]

# CIFAR10, CIFAR100
_C.DATASETS.CIFAR = CfgNode()
_C.DATASETS.CIFAR.SHUFFLE_INDICES = False
_C.DATASETS.CIFAR.TRAIN_INDICES = [0, 45000,]
_C.DATASETS.CIFAR.VALID_INDICES = [45000, 50000,]

# TinyImageNet200
_C.DATASETS.TINY = CfgNode()
_C.DATASETS.TINY.SHUFFLE_INDICES = False
_C.DATASETS.TINY.TRAIN_INDICES = [0, 90000,]
_C.DATASETS.TINY.VALID_INDICES = [90000, 100000,]

# ImageNet1k_x32, ImageNet1k_x64
_C.DATASETS.DOWNSAMPLED_IMAGENET = CfgNode()
_C.DATASETS.DOWNSAMPLED_IMAGENET.SHUFFLE_INDICES = True
_C.DATASETS.DOWNSAMPLED_IMAGENET.TRAIN_INDICES = [0, 1231167,]
_C.DATASETS.DOWNSAMPLED_IMAGENET.VALID_INDICES = [1231167, 1281167,]

# ImageNet1k
_C.DATASETS.IMAGENET = CfgNode()
_C.DATASETS.IMAGENET.SHUFFLE_INDICES = True
_C.DATASETS.IMAGENET.TRAIN_INDICES = [0, 1231167,]
_C.DATASETS.IMAGENET.VALID_INDICES = [1231167, 1281167,]
_C.DATASETS.IMAGENET.ROOT = './datasets/ILSVRC2012/'

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
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.CONV_KSP = [3, 1, 1,]
_C.MODEL.BACKBONE.RESNET.FIRST_BLOCK.POOL_KSP = [3, 2, 1,]
_C.MODEL.BACKBONE.RESNET.BLOCK = 'BasicBlock'
_C.MODEL.BACKBONE.RESNET.SHORTCUT = 'ProjectionShortcut'
_C.MODEL.BACKBONE.RESNET.NUM_BLOCKS = [4, 4, 4,]
_C.MODEL.BACKBONE.RESNET.WIDEN_FACTOR = 1
_C.MODEL.BACKBONE.RESNET.CONV_LAYERS = 'Conv2d'
_C.MODEL.BACKBONE.RESNET.NORM_LAYERS = 'BatchNorm2d'
_C.MODEL.BACKBONE.RESNET.ACTIVATIONS = 'ReLU'

# ResNeXt
_C.MODEL.BACKBONE.RESNEXT = CfgNode()
_C.MODEL.BACKBONE.RESNEXT.IN_PLANES = 64
_C.MODEL.BACKBONE.RESNEXT.GROUPS = 32
_C.MODEL.BACKBONE.RESNEXT.WIDTH_PER_GROUP = 4
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK = CfgNode()
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_NORM_LAYER = True
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_ACTIVATION = True
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.USE_POOL_LAYER = True
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.CONV_KSP = [7, 2, 3,]
_C.MODEL.BACKBONE.RESNEXT.FIRST_BLOCK.POOL_KSP = [3, 2, 1,]
_C.MODEL.BACKBONE.RESNEXT.BLOCK = 'BottleneckBlock'
_C.MODEL.BACKBONE.RESNEXT.SHORTCUT = 'ProjectionShortcut'
_C.MODEL.BACKBONE.RESNEXT.NUM_BLOCKS = [3, 4, 6, 3,]
_C.MODEL.BACKBONE.RESNEXT.WIDEN_FACTOR = 1
_C.MODEL.BACKBONE.RESNEXT.CONV_LAYERS = 'Conv2d'
_C.MODEL.BACKBONE.RESNEXT.NORM_LAYERS = 'BatchNorm2d'
_C.MODEL.BACKBONE.RESNEXT.ACTIVATIONS = 'ReLU'

# VGGNet
_C.MODEL.BACKBONE.VGGNET = CfgNode()
_C.MODEL.BACKBONE.VGGNET.DEPTH = 16
_C.MODEL.BACKBONE.VGGNET.IN_PLANES = 64
_C.MODEL.BACKBONE.VGGNET.MLP_HIDDENS = [4096, 4096,]
_C.MODEL.BACKBONE.VGGNET.CONV_LAYERS = 'Conv2d'
_C.MODEL.BACKBONE.VGGNET.NORM_LAYERS = 'NONE'
_C.MODEL.BACKBONE.VGGNET.ACTIVATIONS = 'ReLU'
_C.MODEL.BACKBONE.VGGNET.LINEAR_LAYERS = 'Linear'

# LeNet
_C.MODEL.BACKBONE.LENET = CfgNode()
_C.MODEL.BACKBONE.LENET.CONV_LAYERS = 'Conv2d'
_C.MODEL.BACKBONE.LENET.ACTIVATIONS = 'Sigmoid'
_C.MODEL.BACKBONE.LENET.LINEAR_LAYERS = 'Linear'

# VisionTransformer
_C.MODEL.BACKBONE.VIT = CfgNode()
_C.MODEL.BACKBONE.VIT.PATCH_SIZE = 16
_C.MODEL.BACKBONE.VIT.HIDDEN_SIZE = 768
_C.MODEL.BACKBONE.VIT.TRANSFORMER = CfgNode()
_C.MODEL.BACKBONE.VIT.TRANSFORMER.MLP_DIM = 3072
_C.MODEL.BACKBONE.VIT.TRANSFORMER.NUM_HEADS = 12
_C.MODEL.BACKBONE.VIT.TRANSFORMER.NUM_LAYERS = 12
_C.MODEL.BACKBONE.VIT.TRANSFORMER.DROPOUT_RATE = 0.0
_C.MODEL.BACKBONE.VIT.TRANSFORMER.ATTENTION_DROPOUT_RATE = 0.0

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
_C.MODEL.BATCH_NORMALIZATION.EPSILON = 1e-5
_C.MODEL.BATCH_NORMALIZATION.MOMENTUM = 0.9

# ---------------------------------------------------------------------- #
# Layer Normalization
# ---------------------------------------------------------------------- #
_C.MODEL.LAYER_NORMALIZATION = CfgNode()
_C.MODEL.LAYER_NORMALIZATION.EPSILON = 1e-5

# ---------------------------------------------------------------------- #
# Group Normalization
# ---------------------------------------------------------------------- #
_C.MODEL.GROUP_NORMALIZATION = CfgNode()
_C.MODEL.GROUP_NORMALIZATION.EPSILON = 1e-5
_C.MODEL.GROUP_NORMALIZATION.NUM_GROUPS = 32

# ---------------------------------------------------------------------- #
# Filter Response Normalization
# ---------------------------------------------------------------------- #
_C.MODEL.FILTER_RESPONSE_NORMALIZATION = CfgNode()
_C.MODEL.FILTER_RESPONSE_NORMALIZATION.EPSILON = 1e-6
_C.MODEL.FILTER_RESPONSE_NORMALIZATION.USE_LEARNABLE_EPSILON = False

# ---------------------------------------------------------------------- #
# Dropout
# ---------------------------------------------------------------------- #
_C.MODEL.DROPOUT = CfgNode()
_C.MODEL.DROPOUT.DROP_RATE = 0.5

# ---------------------------------------------------------------------- #
# BatchEnsemble
# ---------------------------------------------------------------------- #
_C.MODEL.BATCH_ENSEMBLE = CfgNode()

# the size of ensembles
_C.MODEL.BATCH_ENSEMBLE.ENSEMBLE_SIZE = 4

# initialization of rank-one factors
_C.MODEL.BATCH_ENSEMBLE.INITIALIZER = CfgNode()
_C.MODEL.BATCH_ENSEMBLE.INITIALIZER.NAME = "normal"
_C.MODEL.BATCH_ENSEMBLE.INITIALIZER.VALUES = [1.0, 1.0,]
