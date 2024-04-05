from flask import Flask, render_template, request, jsonify
import os
import base64
import cv2
from model_inference import Detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
output_file = os.path.join(app.root_path, 'static/uploads/output.jpg')
model = Detection(output_file)

@app.route("/", methods=["GET", "POST"])
def home_page():
    
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"])
def prediction():
    if request.method == 'POST':
        # Handle image upload and prediction
        image_data = request.get_json()['image']
        image_data = bytes(image_data, encoding='utf-8')
        image_data = base64.b64decode(image_data)
        # print(image_data)
        input_path = os.path.join(app.root_path, "static/uploads/input.jpg")
        file_path = save_input_image(input_path, image_data)

        # cv2.imwrite(input_path,image_data)
        detection_flag = model.inference(file_path)
        print("detection_flag", detection_flag)

        
        

        return jsonify({"output_file":"static/uploads/output.jpg"})

def save_input_image(filename_path, image_data):
    # Save the image data to a file
    
    with open(filename_path, 'wb') as f:
        f.write(image_data)

    return filename_path

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)