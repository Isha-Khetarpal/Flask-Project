from flask import Flask, request, render_template, redirect, send_file
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# Create Flask app with correct static and template folders
app = Flask(__name__, static_folder='static', template_folder='templates')

# Use /tmp for uploads in Vercel
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the YOLO model
model_path = os.path.join('model', 'best.pt')  # Path within the app folder
model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        file.save(uploaded_path)

        # Perform prediction
        results = model.predict(uploaded_path)
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')

        # Save the result image with annotations
        for result in results:
            img_with_annotations = result.plot()
            if isinstance(img_with_annotations, np.ndarray):
                img_with_annotations = Image.fromarray(img_with_annotations)
            img_with_annotations.save(result_path)

        return render_template('result.html', uploaded_image='uploaded_image.jpg', result_image='result_image.jpg')

@app.route('/display/<filename>')
def display_image(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, request, render_template, redirect, send_file
# from ultralytics import YOLO
# import os
# from PIL import Image
# import numpy as np

# app = Flask(__name__)

# # Ensure the uploads folder is created
# # Vercel requires the use of /tmp for temporary files
# app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Vercel's writable directory for temporary files
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Model file should be in your project directory (e.g., "model" folder in the root)
# model_path = './model/best.pt'  # Change this to match the model location in your project directory

# # Load the model using the relative path
# model = YOLO(model_path)

# @app.route('/')
# def index():
#     return render_template('upload.html')

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         uploaded_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
#         file.save(uploaded_path)
        
#         # Perform prediction
#         results = model.predict(uploaded_path)
#         result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_image.jpg')

#         # Save the result image with annotations
#         for result in results:
#             img_with_annotations = result.plot()
#             if isinstance(img_with_annotations, np.ndarray):
#                 img_with_annotations = Image.fromarray(img_with_annotations)
#             img_with_annotations.save(result_path)

#         # Render result page
#         return render_template('result.html', uploaded_image='uploaded_image.jpg', result_image='result_image.jpg')

# @app.route('/display/<filename>')
# def display_image(filename):
#     return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True)
