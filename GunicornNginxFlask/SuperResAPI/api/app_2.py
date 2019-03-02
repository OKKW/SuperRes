# app.py - a minimal flask api using flask_restful
from flask import Flask, jsonify, request, redirect, url_for,render_template
from flask_restful import reqparse, abort, Api, Resource
import sys
import os.path
import json
import numpy as np
from PIL import Image  
import cv2
import torch
import architecture as arch
import os 
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

app = Flask(__name__,static_url_path = "/uploads", static_folder = "uploads")
api = Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

parser = reqparse.RequestParser()
parser.add_argument('query')


def allowed_file(filename):
	return '.' in filename and \
		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			process_input(request)
			return redirect(url_for('enhance'))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form method=post enctype=multipart/form-data>
		<p><input type=file name=file>
		<input type=submit value=Upload>
	</form>
	'''

def process_input(request):
	file = request.files['file']
	img = np.array(Image.open(request.files['file']))* 1.0 / 255

	model_path = 'models/interp_08.pth'

	device = torch.device('cpu')

	model = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')

	model.load_state_dict(torch.load(model_path), strict=True)
	model.eval()

	for k, v in model.named_parameters():
	    v.requires_grad = False
	model = model.to(device)

	img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
	img_LR = img.unsqueeze(0)
	img_LR = img_LR.to(device)

	output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
	output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
	output = (output * 255.0).round()

	#destRGB = cv2.cvtColor(output/255.0, cv2.COLOR_BGR2RGB)
	result=Image.fromarray(np.uint8(output))


	try:
		os.mkdir('uploads')
	except:
		pass

	try:
		os.mkdir('uploads/OriginalImages')
	except:
		pass

	try:
		os.mkdir('uploads/EnhancedImages')
	except:
		pass

	Image.open(request.files['file']).save(os.path.join('uploads', file.filename))
	result.save(os.path.join('uploads/EnhancedImages', file.filename))

	return 'SUCCESS'

@app.route("/enhance", methods=['POST'])
def enhance():
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']

		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file :
			process_input(request)
			

if __name__ == '__main__':
	app.run(debug=True,host='0.0.0.0')
