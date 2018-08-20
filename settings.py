# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
## these are set I can't change them for some reason

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

### I'm not sure if I can have more than one queue? But let's try.
FACE_QUEUE = "face_queue"
IMAGE_QUEUE = "image_queue"
YOLO_QUEUE = "yolo_queue"