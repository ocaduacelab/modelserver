from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image

from darkflow.net.build import TFNet
import cv2
import numpy as np
import settings
import helpers
import flask
import redis
import uuid
import time
import json
import io

# initialize our Flask application and Redis server
app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)

## trying out some Yolo options 
#options = {"model": "cfg/yolov2-tiny.cfg", "load": "yolov2-tiny.weights", "threshold": 0.80}
#tfnet = TFNet(options)

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

@app.route("/")
def homepage():
	return "Welcome to the PyImageSearch Keras REST API!"

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image,(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image}
			db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

@app.route("/predictface", methods=["POST"])
def predict_face():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format and prepare it for
			# classification
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = prepare_image(image,(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))

			# ensure our NumPy array is C-contiguous as well,
			# otherwise we won't be able to serialize it
			image = image.copy(order="C")

			# generate an ID for the classification then add the
			# classification ID + image to the queue
			k = str(uuid.uuid4())
			image = helpers.base64_encode_image(image)
			d = {"id": k, "image": image}
			db.rpush(settings.FACE_QUEUE, json.dumps(d))

			# keep looping until our model server returns the output
			# predictions
			while True:
				# attempt to grab the output predictions
				output = db.get(k)

				# check to see if our model has classified the input
				# image
				if output is not None:
					# add the output predictions to our data
					# dictionary so we can return it to the client
					output = output.decode("utf-8")
					data["predictions"] = json.loads(output)

					# delete the result from the database and break
					# from the polling loop
					db.delete(k)
					break

				# sleep for a small amount to give the model a chance
				# to classify the input image
				time.sleep(settings.CLIENT_SLEEP)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


### this isn't passing to redis because I don't know how to deal w/ that outside of the Keras way outlined above. Sorry. 

@app.route("/yolopredict", methods=["POST"])
def yolo_predict():
	pass
	"""
	r = request
	npaar = np.fromstring(r.data, np.unit8)
	img = cv2.cvtColor(np.array(npaar), cv2.COLOR_RGB2BGR)
	result = tfnet.return_predict(img)
	print(result)
	return flask.jsonify(result)
	"""
    

if __name__ == "__main__":
	print("* Starting web service...")
	#app.run()
	app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)