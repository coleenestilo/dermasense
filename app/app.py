# Importing libraries
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import font as tkfont
import torch
import numpy as np
import os
from tkinter import ttk
import threading
import json
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))        # .../code/app
YOLO_DIR = os.path.abspath(os.path.join(
    BASE_DIR, '..', 'yolov5'))  # .../code/yolov5

# image uploader function
def imageUploader(self):
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        self.controller.image_path=path
        self.controller.show_frame("LoadingPage")

        # Running the model in a separate thread to avoid blocking the UI
        threading.Thread(target=self.controller.run_models).start()

    # if no file is selected, then we are displaying below message
    else:
        print("No file is Chosen !! Please choose a file.")

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")
    
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, UploadPage, ShowResults, LoadingPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")


        # Loading of models
        self.model_face_path = os.path.join(YOLO_DIR, 'runs', 'train', '300EpochsTraining_Faces4', 'weights', 'best.pt')
        self.model_acnewrinkles_path = os.path.join(YOLO_DIR, 'runs', 'train', '1000EpochsTraining_AcneWrinklesOnly4', 'weights', 'best.pt')
        self.model_hypacnescars_path = os.path.join(YOLO_DIR, 'runs', 'train', '1000EpochsTraining_HyperpigmentationAcneScarsOnly3', 'weights', 'best.pt')
        
    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

    def run_models(self):
        image_path = self.image_path
        results = {}
        class_counts = {}

        # Run model_face
        model_face = torch.hub.load(
            YOLO_DIR, 'custom', source='local', path=self.model_face_path)
        print("===== FACE MODEL LOADED =====")

        results_face = model_face(image_path)
        img = cv2.imread(image_path)  # Read the original image using OpenCV
        json_face = results_face.pandas().xyxy[0].to_json(orient="records")
        if len(json.loads(json_face)) > 0:
            
            # Run model_acnewrinkles
            model_acnewrinkles = torch.hub.load(
                YOLO_DIR, 'custom', source='local', path=self.model_acnewrinkles_path)
            print("===== ACNE WRINKLES MODEL LOADED =====")

            results_acnewrinkles = model_acnewrinkles(image_path)
            json_acnewrinkles = results_acnewrinkles.pandas().xyxy[0].to_json(orient="records")
            if len(json.loads(json_acnewrinkles)) > 0:
                print(json_acnewrinkles)
                results["Acne/Wrinkles"] = "Detected"
                img = self.draw_boxes(img, results_acnewrinkles,"Acne/Wrinkles")

            # Run model_hypacnescars
            model_hypacnescars = torch.hub.load(
                YOLO_DIR, 'custom', source='local', path=self.model_hypacnescars_path)
            print("===== HYPERPIGMENTATION ACNE SCARS MODEL LOADED =====")

            results_hypacnescars = model_hypacnescars(image_path)
            json_hypacnescars = results_hypacnescars.pandas().xyxy[0].to_json(orient="records")
            if len(json.loads(json_hypacnescars)) > 0:
                print(json_hypacnescars)
                results["Hyperpigmentation/Acne Scars"] = "Detected"
                img = self.draw_boxes(img, results_hypacnescars, "Hyperpigmentation/Acne Scars")
            
            # Get the detections
            data = json.loads(json_acnewrinkles) + json.loads(json_hypacnescars)
            class_counts = {}
            for detection in data:
                class_name = detection["name"]
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

        else:
            results["Face"] = "Not Detected"

        # Save results to a JSON file
        with open("results.json", "w") as json_file:
            json.dump(results, json_file)

        self.frames["ShowResults"].update_results(class_counts, img)
        self.show_frame("ShowResults")

    def draw_boxes(self, img, results, model_type):
        # Define colors for different classes
        color_map = {
            "Acne/Wrinkles": (255, 0, 0),  # Blue for acne/wrinkles
            "Hyperpigmentation/Acne Scars": (0, 0, 255)  # Red for hyperpigmentation/acne scars
        }

        color = color_map[model_type]
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{results.names[int(cls)]}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return img

class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="#DAD3BE")
        self.controller = controller
        f = tk.Frame(self, bg="#DAD3BE")
        f.place(relx=.5, rely=.5,anchor= tk.CENTER)
        label = tk.Label(f, text="DermaSense", font=controller.title_font, bg="#DAD3BE")
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(f, text="Start",bg="#B7B597", font=("Helvetica", "14"),
            command=lambda: controller.show_frame("UploadPage"), pady=10)
        button1.pack()

class UploadPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,bg="#DAD3BE")
        self.controller = controller
        f = tk.Frame(self, bg="#DAD3BE")
        f.place(relx=.5, rely=.5,anchor= tk.CENTER)

        label = tk.Label(f, text="Upload Image", font=controller.title_font, bg="#DAD3BE")
        label.pack(side="top", fill="x", pady=10)

        self.label = tk.Label(self)  # Add this line to create the label for displaying the image
        self.label.pack()

        self.button = tk.Button(f, text="Upload Image", font=("Helvetica", "14"),bg="#B7B597",
                           command=lambda:imageUploader(self), pady=10)
        self.button.pack()

class LoadingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,bg="#DAD3BE")
        self.controller = controller
        label = tk.Label(self, text="Loading... Please wait.", font=controller.title_font, bg="#DAD3BE")
        label.pack(side="top", fill="x", pady=10)
        progress_bar = ttk.Progressbar(self, length=200, mode='indeterminate')
        progress_bar.pack()
        progress_bar.start(10)

class ShowResults(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent,bg="#DAD3BE")
        self.controller = controller
        
        # Create a main frame to hold the image and results frames
        main_frame = tk.Frame(self,bg="#DAD3BE")
        main_frame.pack(fill="both", expand=True)
        
        # Create the image frame on the left side
        image_frame = tk.Frame(main_frame,bg="#DAD3BE")
        image_frame.pack(side="left", expand=True, fill="both", padx=10)
        image_frame.pack_propagate(False)
        self.image_label = tk.Label(image_frame, padx=10,bg="#DAD3BE")
        self.image_label.pack(expand=True)

        # Create the results frame on the right side
        results_frame = tk.Frame(main_frame,bg="#DAD3BE")
        results_frame.pack(side="right", expand=True, fill="both")
        results_frame.pack_propagate(False)
        self.label = tk.Label(results_frame, text="Results", font=controller.title_font,bg="#DAD3BE")
        self.label.pack(side="top", pady=10)
        self.results_text = tk.Label(results_frame, text="", font=controller.title_font,bg="#DAD3BE")
        self.results_text.pack(side="top", pady=10)
        self.retryBtn = tk.Button(results_frame, text="Retry",font=("Helvetica", "14"),bg="#B7B597",
                            command=lambda: controller.show_frame("UploadPage"))
        self.retryBtn.pack(side="top", pady=10)
    
        # Add the Save button to the results frame
        self.save_button = tk.Button(results_frame, text="Save Image", font=("Helvetica", "14"),bg="#B7B597", command=self.save_image)
        self.save_button.pack(side="top", pady=10)

        self.saved_image = None

    def update_results(self, results, results_image):
        results_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
        if results_str == "":
            results_str = "No Facial Features Detected\nCategory: Others"
        self.results_text.config(text=results_str)

        # Convert the OpenCV image to a format that Tkinter can display
        img = Image.fromarray(cv2.cvtColor(results_image, cv2.COLOR_BGR2RGB))
        img = img.resize((512, 512))
        self.saved_image = img  # Store the image for saving
        pic = ImageTk.PhotoImage(img)
        
        # Display the results image
        self.image_label.config(image=pic)
        self.image_label.image = pic

    def save_image(self):
        if self.saved_image:
            save_path = tk.filedialog.asksaveasfilename(defaultextension=".png",
                                                        filetypes=[("PNG files", "*.png"),
                                                                   ("All files", "*.*")])
            if save_path:
                self.saved_image.save(save_path)
                tk.messagebox.showinfo("Save Image", "Image saved successfully!")


# Main method
if __name__ == "__main__":

    # defining tkinter object
    app = SampleApp()

    # setting title and basic size to our App
    app.title("DermaSense")
    app.geometry("900x600")

    app.mainloop()
