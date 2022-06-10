from tkinter import *
from tkinter import ttk
from time import sleep

from gui_controller import Controller

class GUI():
    def __init__(self):
        app_name = "Facial Reconstruction from Videos"
        window_width = 1000
        window_height = 700
        font_infotext = ('Comic Sans MS',18,"bold")
        font_button_text = ('Comic Sans MS',12,"bold")
        background_app = '#345'
        self.controller = None
        
        self.root = Tk()
        self.root.title(app_name)
        self.root.config(bg=background_app)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.iconbitmap("./a_docs/icon.ico")
        self.user_info = ttk.Label(text="Welcome to the world of facial reconstruction from videos. What would you like to do with your video?")
        self.user_info.configure(font=font_infotext)
        self.user_info.pack(ipadx=10, ipady=10)
        
        # start window
        self.start_win = ttk.Frame(master=self.root)
        self.button_detect = ttk.Button(self.start_win, text="Detect faces", command=self.__detect)
        self.button_classify = ttk.Button(self.start_win, text="Classify found Faces", command=self.__classify)
        self.button_reconstruct = ttk.Button(self.start_win, text="Reconstruct found identities", command=self.__reconstruct)
        
        # self.button_detect.configure(font=font_button_text)
        self.button_detect.grid(column=0, row=1)
        self.button_classify.grid(column=1, row=1)
        self.button_reconstruct.grid(column=2, row=1)
        self.detect_win = None
        self.classify_win = None
        self.reconstruct_win = None
        self.start_win.pack()
        
        # save information
        self.info_about_the_pipeline = None #TODO
        self.progress_frame = None
        self.pb = None
        self.pb_label = None
        self.pb_var = IntVar()
        self.pb_max_len = 0

    def __classify(self):
        return
    
    def __detect(self):
        if self.detect_win is None:
            self.detection_win = ttk.Frame(self.root)
            self.detection_win.bind('<Return>', self.__show_start_window)
            
            self.detection_win.pack()
        #TODO: unshow start and show detection window
        self.detection_win.focus()
        self.controller.detection_gui()
    
    def __reconstruct(self):
        return
    
    def __show_start_window(self):
        #TODO: unshow detection and show start window
        self.start_win.focus()
    
    def __info_about_the_project(self):
        #TODO: pop up
        photo = PhotoImage(file='./a_docs/pipeline.png')
        image_label = ttk.Label(
            self.root,
            image=photo,
            padding=5
        )
    
    def __clear_history(self):
        self.progress_frame = None
        self.pb = None
        self.pb_label = None
        self.pb_var = IntVar()
        self.pb_max_len = 0
    
    def show_detection_progress(self, len):
        if self.progress_frame is None:
            self.progress_frame = ttk.Frame(self.detection_win)
            # progress bar at center
            self.progress_frame.columnconfigure(0, weight=1)
            self.progress_frame.rowconfigure(0, weight=1)
            self.pb = ttk.Progressbar(
                self.progress_frame, orient=HORIZONTAL, mode='determinate', length=len, variable = self.pb_var)
            self.pb_max_len = len
            self.pb.grid(row=0, column=0, sticky=EW, padx=10, pady=10)
            self.progress_frame.grid(row=0, column=0, sticky=NSEW)
            self.pb_label = ttk.Label(self.progress_frame, text="Computing...0%")
            self.pb_label.grid()
            self.progress_frame.pack()
        # self.pb.after(500, self.controller.monitor_model_thread, self.pb)
    
    def update_pb_value(self, curr_value):
        if self.pb_label is None:
            return
        else:
            val = int((curr_value/self.pb_max_len)*100)
            self.pb_var.set(val)
            self.pb_label["text"]=f"Computing...{val}%"
            
    
    def finish_detection_progress(self, output_path):
        self.pb_var.set(100)
        self.pb_label["text"]=f"Finished! You can find your outputs in\n{output_path}"
        
        #TODO: show visualization or something
    
    
    def error_in_progress(self, message):
        print(message)
        #TODO: pop up with error message
        ...
    
    def set_controller(self, controller):
        self.controller = controller
    
    def run(self):
        self.root.mainloop()
        
    
def main():
    try: 
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1) # against blurryness on windows
    finally:
        app = GUI()
        controller = Controller(view=app)
        app.set_controller(controller)
        app.run()


if __name__ == '__main__':
    main()
