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

app = Flask(__name__)
api = Api(app)


parser = reqparse.RequestParser()
parser.add_argument('query')

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
				os.mkdir('Images')
			except:
				pass

			try:
				os.mkdir('Images/OriginalImages')
			except:
				pass

			try:
				os.mkdir('Images/EnhancedImages')
			except:
				pass

			Image.open(request.files['file']).save(os.path.join('Images/OriginalImages', file.filename))
			result.save(os.path.join('Images/EnhancedImages', file.filename))

			return 'SUCCESS'

if __name__ == '__main__':
	app.debug = True
	app.run(host='0.0.0.0')
