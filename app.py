'''
import shutil
from glob import glob
from UIUtility import *
from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
from werkzeug.utils import secure_filename

check_dependencies()
MODEL = load_model_for_ui()
app = Flask(__name__)

# Add paths to app config
for name, path in initialize_database().items():
    app.config[name] = path
app.config["DB"] = "./db"

# Route for the dashboard page


@app.route('/')
def dashboard():
    return render_template('index.html')

# Route for creating a new project


@app.route('/new_project')
def new_project():
    # Logic for creating a new project
    return render_template('new_project.html')


@app.route('/upload_images', methods=['POST'])
def upload_images():
    if "files" not in request.files:
        return jsonify({"success": False, "error": "No files part in the request."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "No files uploaded."}), 400

    saved_files = []
    for file in files:
        if file.filename == "":
            return jsonify({"success": False, "error": "File name cannot be empty."}), 400

        # Save file to upload folder
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        try:
            file.save(file_path)
            saved_files.append(file_path)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": True, "saved_files": saved_files})


@app.route("/remove_temporary", methods=["POST"])
def remove_temporary():
    try:
        clear_dirs()
        return jsonify({"success": True, "message": "Temporary files removed."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate_masks", methods=["POST"])
def generate_masks():
    data = request.get_json()
    if not data or "saved_files" not in data:
        return jsonify({"success": False, "error": "No files provided"}), 400

    file_paths = data["saved_files"]
    mask_paths = model_pipeline(MODEL, file_paths)
    return jsonify({"success": True, "mask_paths": mask_paths})


@app.route("/save_mask", methods=["POST"])
def save_mask():
    data = request.get_json()
    project_name = data.get('project_name')
    filepaths = glob(app.config["IMAGE"] + "/*")
    images = load_drone_images(filepaths)
    masks = load_masks([os.path.join(app.config["MASKS"],
                       os.path.basename(path)) for path in filepaths])
    save_processed_images_mask(
        images, masks, filepaths, tmp=False, project_name=project_name)
    clear_dirs()
    return jsonify({"success": True, "message": "Mask saved successfully."})


@app.route('/get_existing_projects', methods=['GET'])
def get_existing_projects():
    try:
        # List directories in the projects directory
        projects = os.listdir(app.config["DB"])
        projects.remove('tmp')
        n_files_per_project = [
            len(glob(os.path.join(app.config["DB"], project) + "/image/*")) for project in projects]
        return jsonify({"projects": projects, "files": n_files_per_project})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/projects/<project_name>')
def existing_project(project_name):
    return render_template('project.html', project_name=project_name)


@app.route('/get_project_data/<project_name>')
def get_project_data(project_name):
    image_dir = os.path.join(app.config["DB"], project_name, "image")
    mask_dir = os.path.join(app.config["DB"], project_name, "masks")
    overlay_dir = os.path.join(app.config["DB"], project_name, "overlay")
    images = glob(image_dir + "/*")
    masks = glob(mask_dir + "/*")
    overlays = glob(overlay_dir + "/*")
    return jsonify({"images": images, "masks": masks, "overlays": overlays})


@app.route('/db/<project_name>/image/<filename>')
def serve_image_files(project_name, filename):
    # Decode the URL components to handle special characters
    project_path = os.path.join(app.config["DB"], project_name, "image")
    return send_from_directory(project_path, filename)


@app.route('/db/<project_name>/overlay/<filename>')
def serve_overlay_files(project_name, filename):
    # Decode the URL components to handle special characters
    project_path = os.path.join(app.config["DB"], project_name, "overlay")
    return send_from_directory(project_path, filename)


@app.route('/db/<project_name>/masks/<filename>')
def serve_mask_files(project_name, filename):
    # Decode the URL components to handle special characters
    project_path = os.path.join(app.config["DB"], project_name, "masks")
    return send_from_directory(project_path, filename)


@app.route('/db/tmp/masks/<filename>')
def serve_mask(filename):
    return send_from_directory(app.config["MASKS"], filename)

# ALERT SYSTEM


@app.route('/db/<project_name>/alert/<filename>')
def serve_alert(project_name, filename):
    project_path = os.path.join(app.config["DB"], project_name, "alert")
    return send_from_directory(project_path, filename)


@app.route("/clear_alert", methods=["POST"])
def clear_alert():
    try:
        # Get the project name from the request
        project_name = request.form.get("project_name")
        if not project_name:
            return jsonify(success=False, message="Project name is missing."), 400

        # Construct the path to the alert folder
        alert_path = os.path.join(app.config["DB"], project_name, "alert")

        # Remove the alert folder if it exists and recreate it
        if os.path.exists(alert_path):
            shutil.rmtree(alert_path)
        os.makedirs(alert_path, exist_ok=True)

        return jsonify(success=True, message="Alerts cleared successfully.")

    except Exception as e:
        return jsonify(success=False, message=f"An error occurred: {str(e)}"), 500


@app.route('/detect_changes', methods=['POST'])
def detect_changes():
    project_name = request.form.get("project_name")
    # Remove and create upload directory
    # shutil.rmtree(app.config["UPLOAD_FOLDER"], ignore_errors=True)
    # os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # shutil.rmtree(os.path.join(
    #     app.config["DB"], project_name, "alert"), ignore_errors=True)
    # os.makedirs(os.path.join(
    #     app.config["DB"], project_name, "alert"), exist_ok=True)

    file_paths = []
    for key in request.files:
        file = request.files[key]
        if file:
            filename = file.filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            file_paths.append(file_path)

    # Identify common paths
    filenames = [os.path.basename(name) for name in file_paths]
    image_names = os.listdir(os.path.join(
        app.config['DB'], project_name, "image"))
    common_paths = list(set(image_names) & set(filenames))

    # Generate the alerts for the common paths
    FILEs, TIRs, IR_NEWs, IR_REMOVEDs = detect_changes_in_masks(
        common_paths, project_name, app.config["UPLOAD_FOLDER"], MODEL)

    return jsonify(
        success=True,
        n_files=len(FILEs),
        FILEs=FILEs,
        TIRs=TIRs,
        IR_NEWs=IR_NEWs,
        IR_REMOVEDs=IR_REMOVEDs
    )


@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    project_name = request.form.get("project_name")
    filename = request.form.get("file_name")

    # Remove the specific file
    alert_path = os.path.join(app.config["DB"], project_name, "alert", filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    if os.path.exists(alert_path):
        os.remove(alert_path)
    if os.path.exists(file_path):
        os.remove(file_path)
        return jsonify(success=True, message="File removed successfully from the upload folder.")
    else:
        return jsonify(success=True, message="File removed successfully from the upload folder.")


if __name__ == '__main__':
    app.run(debug=True)
'''