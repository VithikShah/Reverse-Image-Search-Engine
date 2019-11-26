import os
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from tensorflow.python.keras.models import Model
from keras.backend import clear_session
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import h5py
import tensorflow as tf

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads') # you'll need to create a folder named uploads

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image only!'), FileRequired('File was empty!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = './uploads/'+photos.url(filename)[37:]
        print(file_url)
        imgs = predict(file_url)
        return render_template('images.html',imgs=imgs)
    else:
        file_url = None
    return render_template('index.html', form=form, file_url=file_url)

def get_feature_vector(path, size=(224,224,)):
    img = Image.open(path)
    img = img.resize(size=size,resample=Image.LANCZOS)
    img = np.array(img)/255.0
    if len(img.shape)==2:
        img = np.repeat(img[:,:,np.newaxis], 3, axis=2)
    image_batch = np.expand_dims(img, axis=0)
    activations = vgg_model.predict(image_batch).reshape((4096,))
    return activations

def generate_h5_file():
	batch = []
	for file in os.listdir('../train2017'):
	    activations = get_feature_vector('../train2017/'+file)
	    batch.append(activations)
	activation_file = h5py.File("../vgg_activations.h5","w")
	activation_file.create_dataset('last_layer_activations',data=vgg_activations)
	activation_file.close()

def predict(path):
	features = get_feature_vector(path)
	l=[]
	for i in range(0,118288,22578):
	    nn = NearestNeighbors(n_neighbors=1,algorithm='brute')
	    end = i+22578
	    if end>118288:
	        end = 118288
	    nn.fit(fvec[i:end])
	    neigh = nn.kneighbors(X=np.array([features]),return_distance=False)[0][0]
	    l.append(i+neigh)
	    print(i+neigh)
	    #     print(l[-1])
	print("DONE")
	files = os.listdir('../train2017')
	files.sort()
	sol_images = []
	for i in l:
	    sol_images.append(files[i])
	return sol_images

if __name__ == '__main__':
	
	clear_session()
	#build model
	image_model = VGG16(include_top=True, weights='imagenet')
	global graph
	graph = tf.compat.v1.get_default_graph()

	VGG_last_layer = image_model.get_layer('fc2')
	vgg_model = Model(inputs = image_model.input, outputs = VGG_last_layer.output)
	print("model ready")

	#load h5 file
	activation_file = h5py.File("../vgg_activations.h5","r")
	fvec = list(activation_file['last_layer_activations'])
	print("h5 loaded")
	app.run()