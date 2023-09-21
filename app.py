from flask import Flask, render_template, request
from keras.models import load_model
import keras.utils as image
import pickle

app = Flask(__name__)

dic = {0 : 'Ulcer', 1 : 'Healthy'}

model = load_model('models/dfu.keras')

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i) # /255.0 
	i = i.reshape(1, 224,224,3)
	y_prob = model.predict([i]) 
	y_classes = y_prob.argmax(axis=-1)
	return dic[y_classes[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Deployment of the Diabetic Foot Ulcer model"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)