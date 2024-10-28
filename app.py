from flask import Flask, request, jsonify, render_template
import cv2
from skimage.metrics import structural_similarity as ssim
import os

app = Flask(__name__)

# Set the path for the master image
MASTER_IMAGE_PATH = 'images/20241018_074456.jpg' 

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/upload_user', methods=['POST'])
def upload_user():
    try:
        user_image = request.files['user_image']
        if not user_image:
            raise ValueError("No user image uploaded.")

        # Ensure uploads directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        user_image_path = 'uploads/user_image.jpg'
        user_image.save(user_image_path)

        # Compare the images
        master_image = cv2.imread(MASTER_IMAGE_PATH)
        user_image = cv2.imread(user_image_path)

        # Check if images are loaded properly
        if master_image is None:
            raise ValueError("Master image could not be loaded. Check the path and file.")
        if user_image is None:
            raise ValueError("User image could not be loaded. Check the uploaded file.")

        # Check image dimensions
        master_height, master_width = master_image.shape[:2]
        user_height, user_width = user_image.shape[:2]

        # Ensure both images are at least 7x7
        if master_height < 7 or master_width < 7 or user_height < 7 or user_width < 7:
            raise ValueError("One of the images is too small for SSIM calculation.")

        # Resize images to 300x300
        master_image = cv2.resize(master_image, (300, 300))
        user_image = cv2.resize(user_image, (300, 300))

        # Calculate minimum dimension for win_size
        min_dimension = min(master_image.shape[0], master_image.shape[1])
        win_size = min(11, min_dimension)  # Ensure win_size does not exceed dimensions

        # Calculate SSIM
        score, _ = ssim(master_image, user_image, full=True, channel_axis=-1)

        # Determine cleanliness result
        result = "The room is clean compared to the master image." if score >= 0.7 else "The room is not clean compared to the master image."

        return jsonify({'similarity_score': score, 'result': result})

    except Exception as e:
        print("Error:", str(e))  # Log the error for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
