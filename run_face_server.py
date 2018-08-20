# import the necessary packages
#from keras.applications import ResNet50
#from keras.applications import imagenet_utils

import numpy as np
import settings
import helpers
import redis
import time
import json
from keras.engine import  Model
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from PIL import Image, ImageFilter


# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,port=settings.REDIS_PORT, db=settings.REDIS_DB)

def face_process():
	#vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') ## doesn't seem to be required. 
	print("* Loading model")
	model = VGGFace(model='vgg16')
	print("* Model loaded")

	while True:
		queue = db.lrange(settings.FACE_QUEUE, 0, settings.BATCH_SIZE - 1)
		imageIDs = []
		batch = None

		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			image = helpers.base64_decode_image(q["image"], settings.IMAGE_DTYPE, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH,settings.IMAGE_CHANS))

			# check to see if the batch list is None
			if batch is None:
				batch = image

			# otherwise, stack the data
			else:
				batch = np.vstack([batch, image])

			# update the list of image IDs
			imageIDs.append(q["id"])

		# check to see if we need to process the batch
		if len(imageIDs) > 0:
			# classify the batch
			print("* Batch size: {}".format(batch.shape))
			preds = model.predict(batch)
			results = utils.decode_predictions(preds)
			#print(results) ## this comes back with something so its the below structure giving me an issue. 
			# [[["b'A.J._Buckley'", 0.9768057], ["b'David_Denman'", 0.0013909286], ["b'Carmine_Giovinazzo'", 0.0010687601], ["b'Robert_Buckley'", 0.00093060045], ["b'Eddie_Cahill'", 0.00044030472]]]

			for (imageID, resultSet) in zip(imageIDs, results):
				print("imageID", imageID, resultSet)
				## imageID 63350aef-0ec3-4e1d-be99-3db36013a6d7
				output = []
				for(label, prob) in resultSet:
					r = {"label":label, "probability":float(prob)}
					output.append(r)
				db.set(imageID, json.dumps(output))

			# remove the set of images from our queue
			db.ltrim(settings.FACE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	face_process()