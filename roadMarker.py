# Import necessary libraries
'''
import os
import shutil
from twilio.rest import Client
import matplotlib.pyplot as plt
from glob import glob
import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from utility import *

# Main Class: Database Management and Alerts
class RoadManager:
    def __init__(self, database_path="./database", alert_threshold=0.05, model_path="./UNetRoadSegmentation_SIH_T1.keras"):
        self.model = load_model(model_path)
        self.database_path = database_path
        self.alert_threshold = alert_threshold
        self.mask_threshold = 0.2
        self.processed_images_path = os.path.join(self.database_path, "images")
        self.generated_masks_path = os.path.join(self.database_path, "masks")
        self.detected_alerts_path = os.path.join(self.database_path, "alerts")
        
        # Email configuration
        self.mail_from = "deepnets722@gmail.com"  # Sender email
        self.mail_from_pswd = "awdh lnni utxc wjbt"   # Sender email's app password
        self.mail_to = "saxenautkarsh722@gmail.com"    # Recipient email
        self.sms_to = "+919300680016"
        clear_console()

    # Database initialization
    def create_database(self):
        os.makedirs(self.database_path, exist_ok=True)
        os.makedirs(self.processed_images_path, exist_ok=True)
        os.makedirs(self.generated_masks_path, exist_ok=True)
        os.makedirs(self.detected_alerts_path, exist_ok=True)
        print(f"\nDatabase created or verified at: {self.database_path}")
    
    # Delete Database
    def delete_database(self):
        shutil.rmtree(self.database_path)
        print("Database deleted successfully!\n")

    # Filling the database with initial images and masks
    def fill_database(self, user_image_dir):
        self.create_database()
        if not os.listdir(self.processed_images_path) or not os.listdir(self.generated_masks_path):
            filepaths = glob(user_image_dir + "/*")
            if not filepaths:
                print("\nNo valid images found. Please try again.")
                return
            self.model_pipeline(filepaths)
            print("\nDatabase has been initialized with images and masks.")
        else:
            print("\nDatabase already contains data. Proceeding with updates...")

    def update_database(self, new_image_dir):
        new_image_paths = glob(os.path.join(new_image_dir, "*"))
        new_image_names = [os.path.basename(path) for path in new_image_paths]
        existing_image_names = [os.path.basename(path) for path in glob(self.processed_images_path + "/*")]

        # Identify new and common images
        new_images = set(new_image_names) - set(existing_image_names)
        common_images = set(new_image_names).intersection(existing_image_names)

        # Handle new images
        if new_images:
            print(f"\nAdding {len(new_images)} new images to the database.")
            new_image_paths = [os.path.join(new_image_dir, name) for name in new_images]
            self.model_pipeline(new_image_paths)
            print("New images and their masks have been added.")

        # Handle updates for common images
        if common_images:
            print(f"\n{len(common_images)} images already exist. Checking for changes...")
            common_image_paths = [os.path.join(new_image_dir, name) for name in common_images]
            self.check_changes(common_image_paths)

    def model_pipeline(self, filepaths):
        current_images = load_drone_images(filepaths)
        print("\nMasks are on the way. Please wait.")
        masks = generate_masks(current_images, self.model, self.mask_threshold)
        print("\nMasks are successfully generated.")
        save_processed_images_mask(current_images, masks, self.processed_images_path, self.generated_masks_path, filepaths)

    # Check changes in new images
    def check_changes(self, new_image_filepaths=None, direct_alert=False):
        if direct_alert:
            user_image_dir = input("\nEnter the image directory path: ")
            file_names = [os.path.basename(path) for path in glob(user_image_dir + "/*")]
            existing_files = [os.path.basename(path) for path in glob(self.processed_images_path + "/*")]
            common_files = set(file_names).intersection(existing_files)
            new_image_filepaths = [os.path.join(user_image_dir, name) for name in common_files]

        file_names = [os.path.basename(path) for path in new_image_filepaths]
        new_images = load_drone_images(new_image_filepaths)

        # Say the user that masks are on the way
        print("\nMasks are on the way. Please wait.")
        new_masks = generate_masks(new_images, self.model, self.mask_threshold)
        print("\nMasks are successfully generated.")
        
        _, old_masks, old_images = load_prev_imgs_masks(file_names, self.processed_images_path, self.generated_masks_path)

        for file_name, original_mask, new_mask, original_image, new_image in zip(file_names, old_masks, new_masks, old_images, new_images):
            tir, roads_constructed, roads_removed, ir_new, ir_removed = compute_new_and_removed_roads(original_mask, new_mask)
            if tir >= self.alert_threshold:
                self.handle_alert(
                    file_name, 
                    original_mask, new_mask, 
                    original_image, new_image, 
                    roads_constructed, roads_removed, 
                    tir, ir_new, ir_removed, 
                    direct_alert)

    def handle_alert(self, file_name, original_mask, new_mask, original_image, new_image, roads_constructed, roads_removed, tir, ir_new, ir_removed, direct_alert):
        """
        Handles the alerting system for significant changes in masks.
        """
        print(f"Alert: {file_name} detected significant changes. TIR: {tir}")

        # Show detailed changes between masks & Save the alert image
        alert_image_path = os.path.join(self.detected_alerts_path, f"alert_{file_name}")
        show_detailed_changes(
            original_mask, new_mask, 
            roads_constructed, roads_removed, 
            original_image, new_image, 
            tir, ir_new, ir_removed,  
            alert_image_path
        )
        
        # Handle direct vs indirect alerts
        if not direct_alert:

            # Notify via SMS and email
            sms_body = f"ALERT: {file_name} detected significant changes.TIR: {tir}\nNew Roads: {ir_new}, Removed Roads: {ir_removed}."
            email_body = f"\nSignificant changes detected in {file_name}.\nTIR: {tir}\nNew Roads: {ir_new}\nRemoved Roads: {ir_removed}.\nAttached is the updated mask for review."
            email_subject = "Road ALERT:"

            if ir_new>=ir_removed:
                sms_body+="\nRoad construction has been detected."
                email_body+="\nRoad construction has been detected."
                email_subject+="Road construction has been detected."
            else:
                sms_body+="\nRoad removal has been detected."
                email_body+="\nRoad removal has been detected."
                email_subject+="Road removal has been detected."

            # self.send_notifications(sms_body, email_subject, email_body, alert_image_path)            
            # self.user_mask_choice(file_name, new_mask)
        
    # Send SMS and Email notifications
    def send_notifications(self, sms_body, email_subject, email_body, image_path):
        self.send_sms(sms_body)
        self.send_email(email_subject, email_body, image_path)

    def send_sms(self, body):
        try:
            client = Client("AC209bf06d7d35772f8c4c283e22266f01", "de93e58b777ba725b2be7673690efb9d") # Replace with your Twilio credentials
            message = client.messages.create(body=body, from_="+16814343297", to=self.sms_to)
            print(f"SMS sent successfully to {self.sms_to}. SID: {message.sid}")
        except Exception as e:
            print(f"Failed to send SMS. Error: {e}")

    def send_email(self, subject, body, image_path=None):
        try:
            # Create the MIMEMultipart message object
            message = MIMEMultipart()
            message['From'] = self.mail_from
            message['To'] = self.mail_to
            message['Subject'] = subject
            
            # Attach the body of the email
            message.attach(MIMEText(body, 'plain'))
            
            # Attach file if image_path is provided and file exists
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as attachment:
                    mime_base = MIMEBase('application', 'octet-stream')
                    mime_base.set_payload(attachment.read())
                encoders.encode_base64(mime_base)
                mime_base.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
                message.attach(mime_base)
            
            # Connect to the Gmail SMTP server
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.mail_from, self.mail_from_pswd)
            
            # Send the email
            server.sendmail(self.mail_from, self.mail_to, message.as_string())
            server.quit()
            
            print("Email has been sucessfully sent to " + self.mail_to)
            
        except Exception as e:
            print(f"Error sending email: {e}")


    # Prompt the user for mask retention choice
    def user_mask_choice(self, file_name, new_mask):
        while True:
            user_choice = input(f"Do you want to keep the new mask for {file_name}? (y/n): ").strip().lower()
            if user_choice in {"y", "n"}:
                break
            print("\nInvalid input. Please enter 'y' or 'n'.")
        if user_choice == "y":
            mask_file_name = file_name.replace('.', '_mask.')
            mask_file_path = os.path.join(self.generated_masks_path, mask_file_name)
            new_mask = tf.squeeze(new_mask)
            plt.imsave(fname=mask_file_path, arr=new_mask, cmap="gray")
            print(f"New mask saved for {file_name} at {mask_file_path}.")
        else:
            print(f"Retaining the old mask for {file_name}. No changes made to the database.")
    
    def alert_authority(self):
        alert_image_paths = glob(os.path.join(self.detected_alerts_path, "*.JPG"))
        alert_csv_path = glob(os.path.join(self.detected_alerts_path, "*.csv"))[-1]
        alert_csv = pd.read_csv(alert_csv_path)

        # Prompt the user that CSV file does not exist cant send alert to authority ensure that file exists
        if not alert_csv_path:
            print("No CSV file found. Please ensure the CSV file exists in the detected_alerts_path.")
            return
        
        # Prompt the user that no alerts are found and exit the function
        if alert_csv.empty:
            print("No alerts found. No alerts to notify the authority.")
            return
        
        for alert_image_path in alert_image_paths:
            file_name = os.path.basename(alert_image_path)

            req_data = alert_csv[alert_csv.file_name  == file_name]
            tir = req_data["TIR"].values[0]
            ir_new = req_data["ir_new"].values[0]
            ir_removed = req_data["ir_removed"].values[0]

            print(tir, ir_new, ir_removed)

            # Notify via SMS and email
            sms_body = f"ALERT: {file_name} detected significant changes.TIR: {tir}\nNew Roads: {ir_new}, Removed Roads: {ir_removed}."
            email_body = f"\nSignificant changes detected in {file_name}.\nTIR: {tir}\nNew Roads: {ir_new}\nRemoved Roads: {ir_removed}.\nAttached is the updated mask for review."
            email_subject = "Road ALERT:"

            if ir_new>=ir_removed:
                sms_body+="\nRoad construction has been detected."
                email_body+="\nRoad construction has been detected."
                email_subject+="Road construction has been detected."
            else:
                sms_body+="\nRoad removal has been detected."
                email_body+="\nRoad removal has been detected."
                email_subject+="Road removal has been detected."

            self.send_notifications(sms_body, email_subject, email_body, alert_image_path)
    
    def start_view_manager(self):
        view_manager = ViewManager(self.database_path, self.generated_masks_path, self.detected_alerts_path)
        view_manager.start()



'''