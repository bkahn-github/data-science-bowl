from config import Config

class BowlConfig(Config): 
    NAME = "bowl" 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 1 
    NUM_CLASSES = 1 + 1 
    IMAGE_MIN_DIM = 512 
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT)
    MEAN_PIXEL = [0, 0, 0] 
    LEARNING_RATE = 0.001 
#     LEARNING_RATE = 0.0001 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 256
    RESNET_ARCHITECTURE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    RPN_NMS_THRESHOLD = 0.7
    DETECTION_MAX_INSTANCES = 500
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    MEAN_PIXEL = [0, 0, 0] 
    WEIGHT_DECAY = 0.0001
    
# class BowlConfig(Config): 
#     NAME = "bowl" 
#     GPU_COUNT = 1 
#     IMAGES_PER_GPU = 1 
#     NUM_CLASSES = 1 + 1 
#     IMAGE_MIN_DIM = 256 
#     IMAGE_MAX_DIM = 512 
#     RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64) 
#     TRAIN_ROIS_PER_IMAGE = 500 
#     STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
#     VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT)
#     MEAN_PIXEL = [0, 0, 0] 
# #     LEARNING_RATE = 0.01 
#     LEARNING_RATE = 0.001 
# #     LEARNING_RATE = 0.0001 
#     USE_MINI_MASK = True 
#     MAX_GT_INSTANCES = 500

    
# from config import Config

# class BowlConfig(Config):
#     """Configuration for training on the toy shapes dataset.
#     Derives from the base Config class and overrides values specific
#     to the toy shapes dataset.
#     """
#     # Give the configuration a recognizable name
#     NAME = "bowl"

#     # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
#     # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 2

#     # Number of classes (including background)
#     NUM_CLASSES = 1 + 1 # background + nuclei

#     # Use small images for faster training. Set the limits of the small side
#     # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM = 512
#     IMAGE_MAX_DIM = 512

#     # Use smaller anchors because our image and objects are small
#     RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

#     # Reduce training ROIs per image because the images are small and have
#     # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
#     TRAIN_ROIS_PER_IMAGE = 600

#     STEPS_PER_EPOCH = None

#     # use small validation steps since the epoch is small
#     VALIDATION_STEPS = 5

#     USE_MINI_MASK = True

#     MAX_GT_INSTANCES = 256

#     DETECTION_MAX_INSTANCES = 512

#     RESNET_ARCHITECTURE = "resnet50"


bowl_config = BowlConfig()
bowl_config.display()
