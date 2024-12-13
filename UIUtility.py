
'''
import os
import cv2
import shutil
import smtplib
import numpy as np
from glob import glob
import tensorflow as tf
from email import encoders
import matplotlib.pyplot as plt
from utility import load_model
from twilio.rest import Client
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart

# Constants
BASE_PATHS = {
    "MODEL": "./model.keras",
    "DB": "./db",
    "BASE": "./db/tmp",
}

SUB_PATHS = {
    "UPLOAD": "upload",
    "ALERT": "alert",
    "IMAGE": "image",
    "MASKS": "masks",
    "OVERLAY": "overlay",
}

# Resolve absolute paths
FULL_PATHS = {key: os.path.join(
    BASE_PATHS["BASE"], path) for key, path in SUB_PATHS.items()}


def check_dependencies():
    try:
        os.system("conda activate sih")
    except:
        os.system("pip install -r requirements")


def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def initialize_database():
    create_directories(FULL_PATHS.values())
    clear_dirs()
    return get_paths()


def clear_dirs():
    for path in [FULL_PATHS["IMAGE"], FULL_PATHS["MASKS"], FULL_PATHS["ALERT"], FULL_PATHS["UPLOAD"]]:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


def get_paths():
    return {
        "IMAGE": FULL_PATHS["IMAGE"],
        "MASKS": FULL_PATHS["MASKS"],
        "ALERT": FULL_PATHS["ALERT"],
        "UPLOAD_FOLDER": FULL_PATHS["UPLOAD"],
    }


# Model Utilities
IMAGE_HEIGHT, IMAGE_WIDTH, N_IMAGE_CHANNELS = 256, 256, 3


def load_model_for_ui(model_path=BASE_PATHS["MODEL"]):
    return load_model(model_path)


def load_drone_image(image_path):
    image = tf.io.read_file(filename=image_path)
    processed_image = tf.image.decode_jpeg(
        contents=image, channels=N_IMAGE_CHANNELS)
    processed_image = tf.image.convert_image_dtype(
        image=processed_image, dtype=tf.float32)

    # Crop to square and resize
    image_shape = tf.shape(processed_image)
    height = image_shape[0]
    width = image_shape[1]
    min_dim = tf.minimum(height, width)

    if (height != IMAGE_HEIGHT) or (width != IMAGE_WIDTH):

        processed_image = tf.image.resize_with_crop_or_pad(
            processed_image, target_height=min_dim, target_width=min_dim)
        processed_image = tf.image.resize(
            images=processed_image, size=(IMAGE_WIDTH, IMAGE_HEIGHT))

    processed_image = tf.clip_by_value(
        processed_image, clip_value_min=0.0, clip_value_max=1.0)
    return tf.cast(processed_image, dtype=tf.float32)


def load_drone_images(filepaths):
    images_numpy = np.empty(shape=(
        len(filepaths), IMAGE_WIDTH, IMAGE_HEIGHT, N_IMAGE_CHANNELS), dtype=np.float32)
    index = 0
    for image_path in filepaths:
        image = load_drone_image(image_path)
        images_numpy[index] = image
        index += 1
    return images_numpy


def load_masks(filepaths):
    masks_numpy = np.empty(
        shape=(len(filepaths), IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float32)
    index = 0
    for image_path in filepaths:
        mask = load_drone_image(image_path)[:, :, 1]
        masks_numpy[index] = mask
        index += 1
    return masks_numpy


def generate_masks(images, model, mask_threshold=0.25):
    """Generate binary masks for the images using the model."""
    images = tf.data.Dataset.from_tensor_slices((images))
    images = images.batch(batch_size=16).prefetch(buffer_size=tf.data.AUTOTUNE)
    binary_masks = model.predict(images, verbose=1)
    rot_images = images.map(lambda x: tf.image.rot90(x, k=1))
    gen_masks_rotated = model.predict(rot_images, verbose=1)
    binary_masks_rotated = tf.image.rot90(gen_masks_rotated, k=3)
    combined_masks_rot = binary_masks_rotated + binary_masks
    combined_masks_rot = tf.cast(
        combined_masks_rot > mask_threshold, dtype=tf.float32)
    return combined_masks_rot


def create_new_db(project_name):
    base_dir = os.path.join(BASE_PATHS["DB"], project_name)
    paths = [base_dir, os.path.join(
        base_dir, "image"), os.path.join(base_dir, "masks"), os.path.join(base_dir, "alert"), os.path.join(base_dir, "overlay")]
    create_directories(paths)


def save_processed_images_mask(images, masks, filepaths, tmp=True, project_name=None):
    if tmp:
        image_output_dir = FULL_PATHS["IMAGE"]
        mask_output_dir = FULL_PATHS["MASKS"]
    else:
        image_output_dir = FULL_PATHS["IMAGE"]
        mask_output_dir = FULL_PATHS["MASKS"]

    if project_name:
        create_new_db(project_name)
        base_dir = os.path.join(BASE_PATHS["DB"], project_name)
        image_output_dir = os.path.join(base_dir, "image")
        mask_output_dir = os.path.join(base_dir, "masks")
        overlay_output_dir = os.path.join(base_dir, "overlay")
        overlay_output_dir = [os.path.join(
            overlay_output_dir, os.path.basename(path)) for path in filepaths]

    image_paths = [os.path.join(
        image_output_dir, os.path.basename(path)) for path in filepaths]
    mask_paths = [os.path.join(
        mask_output_dir, os.path.basename(path)) for path in filepaths]

    index = 0
    for image, mask, image_path, mask_path in zip(images, masks, image_paths, mask_paths):
        plt.imsave(fname=image_path, arr=image)
        plt.imsave(fname=mask_path, arr=tf.squeeze(mask), cmap="gray")

        if project_name:
            plt.imshow(plt.imread(image_path))
            plt.imshow(plt.imread(mask_path), cmap='gray', alpha=0.5)
            plt.axis('off')
            plt.savefig(overlay_output_dir[index])
            index += 1


def model_pipeline(model, filepaths, mask_threshold=0.25):
    images = load_drone_images(filepaths)
    masks = generate_masks(images, model, mask_threshold)
    save_processed_images_mask(images, masks, filepaths)
    mask_paths = [os.path.join(
        FULL_PATHS["MASKS"], os.path.basename(path)) for path in filepaths]
    return mask_paths


def compute_impact_ratio(mask, total_pixels):
    white_pixels = np.sum(mask)
    return white_pixels / total_pixels


def compute_new_and_removed_roads(original_mask, new_mask, total_pixels=256*256):
    # Ensure both masks are binary
    _, original_binary = cv2.threshold(
        original_mask, 0.5, 1, cv2.THRESH_BINARY)
    _, new_binary = cv2.threshold(new_mask.numpy(), 0.5, 1, cv2.THRESH_BINARY)

    # Perform morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    original_binary = cv2.morphologyEx(
        original_binary, cv2.MORPH_CLOSE, kernel)
    new_binary = cv2.morphologyEx(new_binary, cv2.MORPH_CLOSE, kernel)

    # Compute new roads & removed roads
    new_roads = np.maximum(new_binary - original_binary, 0)
    removed_roads = np.maximum(original_binary - new_binary, 0)

    # Compute Impact Ratio
    ir_new = compute_impact_ratio(new_roads, total_pixels)
    ir_removed = compute_impact_ratio(removed_roads, total_pixels)
    tir = ir_new + ir_removed

    return tir, new_roads, removed_roads, ir_new, ir_removed


def show_detailed_changes(original_mask, new_mask, new_roads, removed_roads, original_image, new_image, tir, alert_image_path):
    """
    Display detailed changes between masks and TIR.
    """
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 3, 1)
    plt.title("Original Mask")
    plt.imshow(original_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("New Mask")
    plt.imshow(new_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("New Image")
    plt.imshow(new_image)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Previous Roads Removed")
    plt.imshow(removed_roads, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("New Roads Constructed")
    plt.imshow(new_roads, cmap='gray')
    plt.axis('off')

    plt.suptitle(f"TIR (Total Impact Ratio): {tir:.2f}", fontsize=16)
    plt.tight_layout()

    plt.savefig(alert_image_path)


def handle_alert(file_name, original_mask, new_mask, original_image, new_image, roads_constructed, roads_removed, tir, ir_new, ir_removed, alert_image_path):
    """
    Handles the alerting system for significant changes in masks.
    """
    print(f"Alert: {file_name} detected significant changes. TIR: {tir}")

    # Show detailed changes between masks & Save the alert image

    show_detailed_changes(
        original_mask, new_mask,
        roads_constructed, roads_removed,
        original_image, new_image,
        tir, alert_image_path
    )

    # Notify via SMS and email
    sms_body = f"ALERT: {file_name} detected significant changes.TIR: {
        tir}\nNew Roads: {ir_new}, Removed Roads: {ir_removed}."
    email_body = f"\nSignificant changes detected in {file_name}.\nTIR: {tir}\nNew Roads: {
        ir_new}\nRemoved Roads: {ir_removed}.\nAttached is the updated mask for review."
    email_subject = "Road ALERT:"

    if ir_new >= ir_removed:
        sms_body += "\nRoad construction has been detected."
        email_body += "\nRoad construction has been detected."
        email_subject += "Road construction has been detected."
    else:
        sms_body += "\nRoad removal has been detected."
        email_body += "\nRoad removal has been detected."
        email_subject += "Road removal has been detected."

    # send_notifications(sms_body, email_subject, email_body, alert_image_path)


def send_notifications(sms_body, email_subject, email_body, image_path):
    send_sms(sms_body)
    send_email(email_subject, email_body, image_path)


def send_sms(body):
    try:
        # Replace with your Twilio credentials
        for sms_to in ["+919300680016", "+918969879979"]:
            client = Client("AC209bf06d7d35772f8c4c283e22266f01",
                            "de93e58b777ba725b2be7673690efb9d")
            message = client.messages.create(
                body=body, from_="+16814343297", to=sms_to)
            print(f"SMS sent successfully to {
                sms_to}. SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS. Error: {e}")


def send_email(subject, body, image_path=None):
    try:
        # Create the MIMEMultipart message object
        message = MIMEMultipart()
        mail_from = "Geotrackinnovators@gmail.com"
        mail_from_pswd = "zabv nogz vqee pyam"
        message['From'] = mail_from
        message['To'] = "saxenautkarsh722@gmail.com"
        message['Subject'] = subject

        # Attach the body of the email
        message.attach(MIMEText(body, 'plain'))

        # Attach file if image_path is provided and file exists
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as attachment:
                mime_base = MIMEBase('application', 'octet-stream')
                mime_base.set_payload(attachment.read())
            encoders.encode_base64(mime_base)
            mime_base.add_header(
                'Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            message.attach(mime_base)

        # Connect to the Gmail SMTP server
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(mail_from, mail_from_pswd)

        # Send the email
        server.sendmail(mail_from, "saxenautkarsh722@gmail.com", message.as_string())
        server.quit()

        print("Email has been sucessfully sent to " + "saxenautkarsh722@gmail.com")

    except Exception as e:
        print(f"Error sending email: {e}")


def detect_changes_in_masks(file_names, project_name, upload_dir, model, alert_threshold=0.05):
    image_paths = [os.path.join(
        BASE_PATHS["DB"], project_name, "image", filename) for filename in file_names]
    mask_paths = [path.replace("image", "masks") for path in image_paths]
    
    print(file_names, image_paths, mask_paths)

    og_images = load_drone_images(image_paths)
    og_masks = load_masks(mask_paths)

    new_image_paths = [os.path.join(upload_dir, name) for name in file_names]
    new_images = load_drone_images(new_image_paths)
    new_masks = generate_masks(new_images, model, 0.25)

    FILEs, TIRs, IR_NEWs, IR_REMOVEDs = [], [], [], []
    for file_name, original_mask, new_mask, original_image, new_image in zip(file_names, og_masks, new_masks, og_images, new_images):
        tir, roads_constructed, roads_removed, ir_new, ir_removed = compute_new_and_removed_roads(
            original_mask, new_mask)
        if tir >= alert_threshold:
            handle_alert(
                file_name,
                original_mask, new_mask,
                original_image, new_image,
                roads_constructed, roads_removed,
                tir, ir_new, ir_removed,
                alert_image_path=os.path.join(
                    BASE_PATHS["DB"], project_name, "alert", file_name)
            )
            FILEs.append(file_name)
            TIRs.append(tir)
            IR_NEWs.append(ir_new)
            IR_REMOVEDs.append(ir_removed)

    return FILEs, TIRs, IR_NEWs, IR_REMOVEDs

    
    '''