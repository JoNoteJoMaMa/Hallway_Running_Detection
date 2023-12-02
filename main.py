import os
import time
from ultralytics import YOLO
import cv2
import torch
from datetime import datetime
import tkinter as tk

import supervision as sv

screenshot_interval = 2 
show_bounding_boxes = True

h_set = 11
m_set = 51 

# Use a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use Automatic Mixed Precision (AMP)
amp = True


# Create the main window
window = tk.Tk()
window.title("Y R U Running")

# Get the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set the window size
window_width = 700
window_height = 100

# Calculate the window position
x_position = (screen_width - window_width) // 2
y_position = (screen_height - window_height) // 2

# Set the window geometry
window.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Define the welcome message
welcome_message = "Welcome to the Y R U Running. Click 'START' to proceed."

# Create a label to display the welcome message
welcome_label = tk.Label(window, text=welcome_message, font=("Arial", 14))
welcome_label.pack(side="top", pady=(window_height // 4, 0))

# Create a "Yes" button to proceed
yes_button = tk.Button(window, text="START", command=lambda: start_detection())
yes_button.pack(side="top", pady=(10, 0))


#Set time
current_time = time.time()
now = datetime.now()
date = now.strftime("%d.%m.%Y")
hour = now.strftime("%H.%M.%S")

def start_detection():
    # Hide the welcome page
    welcome_label.destroy()
    yes_button.destroy()
    main()
    
    

def toggle_bounding_boxes():
    global show_bounding_boxes
    show_bounding_boxes = not show_bounding_boxes
    

def main():
    current_time = time.time()
    now = datetime.now()
    date = now.strftime("%d.%m.%Y")
    hour = now.strftime("%H.%M.%S")
            
    model = YOLO("runs/content/runs/detect/train2/weights/best.pt")
    
    box_annotator = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=0.5
    )
    
    last_screenshot_time = time.time()
    
    for result in model.track(source="New_HallWay_Running.mp4", stream=True, device=device, amp=amp):
        

        frame = result.orig_img
        detection = sv.Detections.from_ultralytics(result)        
        if result.boxes.id is not None:
            detection.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        
        # detection = detection[detection.class_id != 1]
    
        labels = [
            f"#{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detection
        ]   
    
        if show_bounding_boxes:
            frame = box_annotator.annotate(
                scene=frame,
                detections=detection,
                labels=labels)
            
        # Add message at the right bottom corner
        message = "Press 'b' to toggle bounding boxes"
        text_color = (0, 0, 255) 
        cv2.putText(frame, message, (1300, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        messageTime = f"Date: {datetime.now().strftime('%d/%m/%Y')} Time: {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(frame, messageTime, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        cv2.imshow("yolov8", frame)
        
        
        if detection[detection.class_id == 0]:
            current_time = time.time()
            now = datetime.now()
            date = now.strftime("%d.%m.%Y")
            hour = now.strftime("%H.%M.%S")
            if current_time - last_screenshot_time > screenshot_interval:
                # Take a screenshot
                screenshot = cv2.resize(frame, (1920, 1080))  # Adjust the size if needed
                screenshot_filename = os.path.join('imageSave', f"Person is running on Date {date} at Time {hour}.png")
                cv2.imwrite(screenshot_filename, screenshot)
                last_screenshot_time = current_time
                
                
        key = cv2.waitKey(30)
        if (key == 27):
            cv2.destroyAllWindows()
            window.destroy()
            # restart_gui()
            break
        elif key == ord('b'):  # Toggle bounding boxes with 'b' key
            toggle_bounding_boxes()

if __name__ == "__main__":
    window.mainloop()           
