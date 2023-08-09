from flask import Flask, request, redirect, url_for, session, flash, render_template, jsonify
import os
import detection_script
import statistics
from PIL import Image
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.secret_key = "hello"

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

network, class_names, class_colors = detection_script.load_network_at_begining()
all_detected_numbers = []
fil = []

@app.template_filter('enumerate')
def _enumerate(seq, start=0):
    return enumerate(seq, start)

@app.route('/')
def home():
    all_detected_numbers = []
    fil = []
    return render_template('home.html')


@app.route('/upload_single_image')
def upload_single_image():
    return render_template('upload_image.html', multiple=False)

@app.route('/upload_multiple_images')
def upload_multiple_images():
    return render_template('upload_images.html', multiple=True)



"""
@app.route('/login', methods=["POST","GET"])
def login():
    if request.method == "POST":
        user_name = request.form["nm"]
        session["user_name"] = user_name
        flash("Login is successful!")
        return redirect(url_for("user"))
    else:
        if "user_name" in session: # it means we are logged in
            flash("Already logged in")
            return redirect(url_for("user"))
        
        return render_template("login.html")

@app.route("/user")
def user():
    if "user_name" in session:
        user_name = session["user_name"]
        return render_template("user.html", user_name=user_name)
    else:
        flash("You are not logged in!")
        return redirect(url_for("login"))

@app.route("/logout")
def logout():
    #if "user_name" in session:
    #    user_nm = session["user_name"]
    flash("You have been logged out!")
    session.pop("user_name", None)
    return redirect(url_for("login"))
"""


@app.route('/upload_file', methods=['POST'])
def upload_file():

    if request.method == 'POST':
        # Check if a file was uploaded in the request
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join('static', 'uploads', f.filename)
            print("Upload folder is  ", filepath)

            # Save the uploaded file to the specified path
            f.save(filepath)

            # Predict detections
            res = detection_script.main_detect(filepath, network, class_names, class_colors)


            # Set filename, path and save predicted results image
            im = Image.fromarray(res[0].image)
            im = im.resize((640, 480))
            path_ = f.filename[:-4]
            path_ = "uploads/" + path_ + "_detect.jpg"
            im.save("static/" + path_)

            # get predicted numbers
            detected_numbers = res[0].detect_label

            return render_template('upload_file.html', filename_=path_, values=detected_numbers)

        elif 'save_button' in request.form["clicked_button"]:
            # Button 1 was clicked
            # Perform corresponding action (e.g., save results to a variable)
            saved_values = {}
            for i in range(5):
                entry_name = 'entry{}'.format(i)
                value = request.form.get(entry_name)
                saved_values[entry_name] = value
                a = 0

            flash("Image uploaded and processed successfully!")

        elif 'save_to_csv_button' in request.form["clicked_button"]:
            # Button 2 was clicked
            # Perform corresponding action (e.g., save results to a CSV file)
            flash("Results saved to CSV successfully!")

    return render_template('upload_file.html', filename_='', values=[0, 0, 0, 0, 0])



@app.route('/upload_files', methods=['POST'])
def upload_files():
    if request.method == 'POST':
        if 'file' in request.files:
            files = request.files.getlist('file')  # Get a list of uploaded files
            currentImageIndex = 0

            # List to store processed results of all images
            paths_to_files = []

            for f in files:
                if f.filename == '':
                    # Ignore empty file input
                    continue

                filename = secure_filename(f.filename)
                filepath = os.path.join('static', 'uploads', filename)
                print("Upload folder is  ", filepath)
                paths_to_files.append(filepath)

                # Save the uploaded file to the specified path
                f.save(filepath)
                
                # Predict detections
            results = detection_script.main_detect(paths_to_files, network, class_names, class_colors)

                # Set filename, path, and save predicted results image
            for res in results:
                im = Image.fromarray(res.image)
                im = im.resize((640, 480))
                path_ = res.filename[:-4] + "_detect.jpg"
                im.save(path_)

                # Get predicted numbers for this image
                detected_numbers = res.detect_label
                # Before rendering the template, replace None values with "-"

                while len(detected_numbers) < 5:
                    detected_numbers.append("-")

                all_detected_numbers.append(detected_numbers)
                fil.append(path_)


            return render_template('upload_files.html', filenames=fil, values=all_detected_numbers, currentImageIndex=currentImageIndex)
        
        elif 'save_button' in request.form or 'save_to_csv_button' in request.form:
            saved_values = {}
            for j in range(5):
                entry_name = 'entry{}'.format(j)
                currentImageIndex = request.form.get('currentImageIndex')
                idx = int(currentImageIndex)
                value = request.form.get(entry_name)
                saved_values[entry_name] = value
                all_detected_numbers[idx][j] = value

            flash("Image uploaded and processed successfully!")
            return render_template('upload_files.html', filenames=fil, values=all_detected_numbers, currentImageIndex=currentImageIndex)


@app.route('/statistics', methods=['POST'])
def statistics():
    gasmeter_values = []
    for gas_val in all_detected_numbers:
        concatenated_value = ''.join(gas_val)
        gasmeter_values.append(int(concatenated_value))
    
    statistics.do_statistics(gasmeter_values)

    return render_template('statistics.html', filename1="uploads/gas_meter.png", filename2="uploads/gas_meter_diff.png")


if __name__ == '__main__':
    app.run() 