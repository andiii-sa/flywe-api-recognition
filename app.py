from flask import (Flask, jsonify)
from flask_restful import Resource, Api, reqparse
# from flask_mysqldb import MySQL
import numpy as np
import os

# import PIL

import tensorflow as tf
from tensorflow import keras

from werkzeug.datastructures import FileStorage

import ast 

app = Flask(__name__)
app.secret_key = 'cac829bb56e1928ef0212927fc5a6071'

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_HERITAGE'] = 'static/model_cnn/model_heritage_borobudur.h5'

batch_size = 32
img_height = 180
img_width = 180

def renamed_file(filename):
    return '.' in filename and filename.rsplit('/', 1)[1]


class Base(Resource):
    def __init__(self):
        self.reqparse = reqparse.RequestParser()
        self.reqparse.add_argument(
            'filename',
            required=True,
            help='filename wajib ada'
        )

        self.reqparse.add_argument(
            'code',
            required=True,
            help='code wajib ada'
        )

        self.reqparse.add_argument(
            'file',
            required=True,
            type=FileStorage,
            help='gambar wajib ada',
            location='files'
        )
        super().__init__()


class image(Base):
    def get(self):
        data = 'Tes GET'
        hasil = []
        hasil.append({
            'tes': data
        })
        return jsonify(hasil)

    def post(self):
        args = self.reqparse.parse_args()
        filename = args.get('filename')
        code = args.get('code')
        filename = renamed_file(filename)
        try:
            upload_file = args['file']  # This is FileStorage instance
            file = upload_file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            list = code
            list = ast.literal_eval(list)
            class_names = list

            # Recreate the exact same model, including its weights and the optimizer
            model = tf.keras.models.load_model(app.config['MODEL_HERITAGE'])

            img = keras.preprocessing.image.load_img(
                (os.path.join(app.config['UPLOAD_FOLDER'], filename)), target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            class_image = class_names[np.argmax(score)]
            persentase = 100 * np.max(score)

            temp_score = []
            for x in score: 
                temp_score.append(np.array(x)*100)
            dict_score = dict(zip(class_names,temp_score))
            sort_dict = sorted(dict_score.items(), key=lambda x: x[1], reverse=True)

            hasil = []
            for c in sort_dict:
                string_value = "{:f}".format(c[1])
                hasil.append({
                    'class_image': c[0],
                    'persentase': string_value
                })
            response = { 
                "status": 200,
                "message": "Success Classifikasi",
                "result" : hasil, 
            }
            pathDelete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.remove(pathDelete)
            return jsonify(response)
        except:
            response = { 
                "status": 500,
                "message": 'Terjadi kesalahan pada server API'
            }
            pathDelete = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.remove(pathDelete)
            return jsonify(response)


@app.route('/')
def index():
    return "<h1>Welcome to API Classification Image"


api = Api(app)
api.add_resource(image, '/api/v1/image-classification')

if __name__ == '__main__':
    app.run(debug=True)
