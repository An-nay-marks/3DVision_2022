from textwrap import wrap
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox
from gui_controller import Controller

class GUI():
    def __init__(self):
        self.controller = None
        app_name = "Facial Reconstruction from Videos"
        window_width = 1000
        window_height = 800
        font_title = ('Comic Sans MS',12,"bold")
        font_button_text = ('Comic Sans MS', 11)
        font_info_text = ('Comic Sans MS', 10)
        font_info_about_project_text = ('Comic Sans MS', 10)
        background_color = "#%02x%02x%02x" % (0,102,102)
        background_color_light = "#%02x%02x%02x" % (224,224,224)
        run_button_color = "#%02x%02x%02x" % (255,128,0)
        info_button_color = "#%02x%02x%02x" % (0,153,0)
        self.root = Tk()
        self.root.title(app_name)
        self.root.protocol("WM_DELETE_WINDOW", self.__on_closing)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root_geometry = f'{window_width}x{window_height}+{center_x}+{center_y}'
        self.root.geometry(self.root_geometry)
        self.root.iconbitmap("./a_docs/icon.ico")
        s = ttk.Style()
        s.configure('TFrame', background=background_color)
        s.configure('TButton', background="white", font=font_button_text, padding=[5, 5, 5, 5])
        s.configure('TLabel', background=background_color, foreground="white", font=font_title)
        s.configure("Info.TLabel", background=background_color, foreground="white", font=font_info_text)
        s.configure("Info.TButton", background=background_color_light, foreground=info_button_color, font=font_info_text, padding=[10, 10, 10, 10], relief = "raised")
        s.configure("Run.TButton", background_color=background_color_light, foreground=run_button_color, font=font_title, relief = "raised")
        s.configure("InfoProject.TLabel", background=background_color, foreground="white", font=font_info_about_project_text)
        # start window
        self.start_win = ttk.Frame(master=self.root)
        self.user_info = ttk.Label(self.start_win, text="Welcome to the world of facial reconstruction from videos.\nWhat would you like to do with your video?", style="TLabel", padding=20, anchor="c")
        self.user_info.pack(padx=30, pady = 10, anchor="n", side=TOP)
        self.button_detect = ttk.Button(self.start_win, text="Detect Faces", command=self.__detect)
        self.button_classify = ttk.Button(self.start_win, text="Detect and Classify Faces", command=self.__classify)
        self.button_reconstruct = ttk.Button(self.start_win, text="Detect, Classify and Reconstruct Faces", command=self.__reconstruct)
        self.button_info = ttk.Button(self.start_win, text="About this Project", command=self.__info_about_the_project, style = "Info.TButton")
        # self.button_detect.configure(font=font_button_text)
        self.button_detect.pack(padx=30, pady = 10, anchor="ne", side=TOP)
        self.button_classify.pack(padx=30, pady = 10, anchor="ne", side=TOP)
        self.button_reconstruct.pack(padx=30, pady = 10, anchor="ne", side=TOP)
        self.button_info.pack(padx=30, pady = 30, anchor="sw", side=LEFT)
        self.start_win.pack(fill="both", expand=1, anchor="center")
        
        # detection window
        self.detection_win = ttk.Frame(self.root)
        self.detection_start_text = "DETECTION"
        
        # classification window
        self.classify_win = ttk.Frame(self.root)
        self.classification_start_text = "CLASSIFICATION"
        
        # reconstruction window
        self.reconstruct_win = ttk.Frame(self.root)
        self.reconstruction_start_text = "RECONSTRUCTION"
        
        # all variables that are used (needed to clean screen afterwards)
        self.progress_frame = None
        self.video_frame = None
        self.pb = None
        self.pb_label = None
        self.pb_var = IntVar()
        self.pb_var_1 = IntVar()
        self.pb_var_2 = IntVar()
        self.pb_max_len = 0
        self.pb_2 = None
        self.pb_2_label = None
        self.pb_2_max_len = 0
        self.curr_frame = self.start_win
        self.curr_user_info = self.user_info
        self.func = "start"
        self.open_video_button = None
        self.open_folder_button = None
        self.open_classified_folder_button = None
        self.button_back_to_start = None
        self.new_user_info = None
        self.radio_buttons = None
        self.run_button = None
        self.classifier_idx = StringVar()
        self.classifiers = None
        self.merge_strategy_idx = StringVar()
        self.merge_strategies = None
        self.detectors = None
        self.detector_idx = StringVar()
        self.load_patches = False
        self.load_classified_patches = False
    
    def __on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            self.controller.exit()
    
    def __detect(self):
        """Detection Button pressed"""
        self.curr_frame = self.detection_win
        self.func = "detect"
        self.curr_user_info = ttk.Label(self.curr_frame, text = self.detection_start_text, justify = "center", wraplength=400)
        self.curr_user_info.pack(padx=30, pady = 10)
        self.__init_return_button()
        self.video_frame = ttk.Label(self.curr_frame)
        self.open_video_button = ttk.Button(self.video_frame, text='Select Raw Video', command=self.__select_video)
        self.open_video_button.pack(padx=30, pady = 10, anchor="nw", side=LEFT)
        self.video_frame.pack()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
        
    def __classify(self):
        """Classification Button pressed"""
        self.curr_frame = self.classify_win
        self.func = "classify"
        self.curr_user_info = ttk.Label(self.curr_frame, text = self.classification_start_text, justify = "center", wraplength=400)
        self.curr_user_info.pack(padx=30, pady = 10)
        self.__init_return_button()
        self.video_frame = ttk.Label(self.curr_frame)
        self.open_video_button = ttk.Button(self.video_frame, text='Select Raw Video', command=self.__select_video)
        self.open_video_button.pack(padx=30, pady = 10, anchor="nw", side=LEFT)
        self.open_folder_button = ttk.Button(self.video_frame, text='Select Patches', command=self.__select_directory)
        self.open_folder_button.pack(padx=0, pady = 10, anchor="nw", side=LEFT)
        self.video_frame.pack()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
    
    def __reconstruct(self):
        """Reconstruct Button pressed"""
        self.curr_frame = self.reconstruct_win
        self.func = "reconstruct"
        self.curr_user_info = ttk.Label(self.curr_frame, text = self.reconstruction_start_text, justify = "center", wraplength=400)
        self.curr_user_info.pack(padx=30, pady = 10)
        self.__init_return_button()
        self.video_frame = ttk.Label(self.curr_frame)
        self.open_video_button = ttk.Button(self.video_frame, text='Select Raw Video', command=self.__select_video)
        self.open_video_button.pack(padx=30, pady = 10, anchor="nw", side=LEFT)
        self.open_folder_button = ttk.Button(self.video_frame, text='Select Patches', command=self.__select_directory)
        self.open_folder_button.pack(padx=0, pady = 10, anchor="nw", side=LEFT)
        self.open_classified_folder_button = ttk.Button(self.video_frame, text='Select Classified Patches', command=self.__select_classified_directory)
        self.open_classified_folder_button.pack(padx=30, pady = 10, anchor="nw", side=LEFT)
        self.video_frame.pack()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
    
    def __init_return_button(self):
        if self.button_back_to_start is not None: # needed to change location of button
            self.button_back_to_start.pack_forget()
        self.button_back_to_start = ttk.Button(self.curr_frame, text="Back to Home", command=self.__show_start_window, style="Info.TButton")
        self.button_back_to_start.pack(padx=20, pady = 20)
    
    def __show_start_window(self):
        self.__clear_history()
        self.start_win.pack(fill="both", expand=1)
        self.curr_frame = self.start_win
        self.curr_user_info = self.user_info
        
        self.func = "start"

    def __info_about_the_project(self):
        infos = ttk.Frame(self.root)
        img = PhotoImage(master=infos, file='./a_docs/Pipeline.png')
        text="Hi, we are Deniz, Lucas and Anne. Our pipeline uses video material to create a person-specific face reconstruction. It uses the fact, that a person usually appears multiple times within a video and elevates that additional information. We start with face detection, offering two different face detectors (SCRFD, Yolov5Face) that extract the facial patches. Afterwards, the pipeline uses clustering methods (DBScan, Agglomerative, k-means) or classifiers such as the VGGNet pretrained on celebrity faces, to identify which facial patch belongs to a certain individual. Finally, we use DECA to create an animatable model of each individual. The last step can be improved by *merging* information of each person. Therefore, we extract the DECA parameters for each face patch with its Encoder. Our pipeline offers to average all DECA parameters or only the shape parameters of an individual.\nWe also trained a Multi-Layer-Perceptron to be predictive of the quality of a given face patch and then predict a weighted average for the DECA parameters. The (weighted) average parameters are then used for the reconstruction.\n\n If you have any question, you can reach out to us or open a Github issue. More information about the project and the performance of the specific architectures can be found in our final report on Github. We hope you have fun with the pipeline!"
        text_label = ttk.Label(infos, text = text, wraplength=600, style="InfoProject.TLabel")
        image_label = ttk.Label(infos, image=img)
        image_label.image = img
        image_label.config(image=img)
        image_label.pack(padx = 5, pady = 5, side=LEFT, anchor="nw")
        text_label.pack(padx=10, pady=10, side=TOP, anchor="nw")
        back_button = ttk.Button(infos, text="back to start menu", command=lambda:[self.__show_start_window(), infos.pack_forget(), self.root.geometry(self.root_geometry)])
        back_button.pack(padx=10, pady=10, side=TOP, anchor="s")
        infos.pack(fill="both", expand=1)
        self.root.geometry("")
        self.curr_frame.pack_forget()
    
    def __show_run_button(self, fun, text):
        self.run_button = ttk.Button(self.curr_frame, text=text, command=lambda:[fun(), self.__unpack_previous_frame()], style="Run.TButton")
        self.run_button.pack(ipadx = 20, ipady = 20, padx = 20, side=TOP, anchor="center")
    
    def __show_radio_buttons(self):
        if self.radio_buttons is not None:
            self.radio_buttons.pack_forget()
            self.run_button.pack_forget()
        self.radio_buttons = ttk.Label(self.curr_frame)
        if self.func == "detect" or ( (self.func=="classify" or self.func=="reconstruct") and not self.load_patches):
            self.detectors = ttk.Label(self.radio_buttons)
            label = ttk.Label(self.detectors, text="Detector:")
            label.pack(padx = 30, pady = 10)
            for idx, detector in enumerate(self.controller.get_detectors()):
                new_radio_button = ttk.Radiobutton(self.detectors, text = detector, value = str(idx), variable = self.detector_idx)
                new_radio_button.pack(padx = 5, pady = 5, ipadx=2, ipady=2)
            self.detector_idx.set(str(0))
            self.detectors.pack(padx=10, pady=30, ipadx=10, ipady=10, side=LEFT, anchor="nw")
            fun = self.controller.detection_gui
            text = "Run Detection"
        if self.func == "classify" or (self.func == "reconstruct" and not self.load_classified_patches):
            self.classifiers = ttk.Label(self.radio_buttons)
            label = ttk.Label(self.classifiers, text="Classifier:")
            label.pack(padx = 30, pady = 10)
            for idx, classifier in enumerate(self.controller.get_classifiers()):
                new_radio_button = ttk.Radiobutton(self.classifiers, text = classifier, value = str(idx), variable = self.classifier_idx)
                new_radio_button.pack(padx = 5, pady = 5, ipadx=2, ipady=2)
            self.classifier_idx.set(str(0))
            self.classifiers.pack(padx=10, pady=30, ipadx=10, ipady=10,side=LEFT, anchor="nw")
            fun = self.controller.classify_gui
            text = "Run Classification"
        if self.func == "reconstruct":
            self.merge_strategies = ttk.Label(self.radio_buttons)
            label = ttk.Label(self.merge_strategies, text="DECA Merge Strategy:")
            label.pack(padx = 30, pady = 10)
            for idx, strategy in enumerate(self.controller.get_merge_strategies()):
                new_radio_button = ttk.Radiobutton(self.merge_strategies, text = strategy, value = str(idx), variable = self.merge_strategy_idx)
                new_radio_button.pack(padx = 5, pady = 5, ipadx=2, ipady=2)
            self.merge_strategy_idx.set(str(0))
            self.merge_strategies.pack(padx=10, pady=30, ipadx=10, ipady=10, side=LEFT, anchor="nw")
            fun = self.controller.reconstruction_gui
            text = "Run Reconstruction"
        self.radio_buttons.pack()
        self.__show_run_button(fun, text)
    
    def __select_video(self):
        filename = fd.askopenfilename(title='Select a Video', initialdir='/')
        if filename =="": return
        self.open_video_button["command"] = self.__change_selected_video
        self.controller.set_in_path(filename)
        self.load_patches = False
        self.load_classified_patches = False
        if self.func == "detect":
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}', style="Info.TLabel")
                self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
            else:
                self.new_user_info["text"] = f'Selected File\n{filename}'
            self.__show_radio_buttons()
        elif self.func == "classify":
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}', style="Info.TLabel")
                self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
            else:
                self.new_user_info["text"] = f'Selected File\n{filename}'
            self.__show_radio_buttons()
        elif self.func == "reconstruct":
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}', style="Info.TLabel")
                self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
            else:
                self.new_user_info["text"] = f'Selected File\n{filename}'
            self.__show_radio_buttons()
    
    def __select_directory(self):
        self.load_patches = True
        self.load_classified_patches = False
        dir_name = fd.askdirectory(title='Select a Directory with Patches (in folder "out")', initialdir='/')
        if dir_name =="": return
        self.open_folder_button["command"] = self.__change_selected_directory
        self.controller.set_in_path(dir_name)
        if self.func == "classify":
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Directory\n{dir_name}', style="Info.TLabel")
                self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
            else:
                self.new_user_info["text"] = f'Selected Directory\n{dir_name}'
            self.__show_radio_buttons()
        elif self.func == "reconstruct":
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Patch Directory\n{dir_name}', style="Info.TLabel")
                self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
            else:
                self.new_user_info["text"] = f'Selected Patch Directory\n{dir_name}'
            self.__show_radio_buttons()
    
    def __select_classified_directory(self):
        self.load_patches = True
        self.load_classified_patches = True
        dir_name = fd.askdirectory(title='Select a Directory with already classified Patches', initialdir='/')
        if dir_name =="": return
        self.open_classified_folder_button["command"] = self.__change_selected_classified_directory
        self.controller.set_in_path(dir_name)
        if self.new_user_info is None:
            self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Classified Patch Directory\n{dir_name}', style="Info.TLabel")
            self.new_user_info.pack(padx=30, pady = 10, anchor="nw")
        else:
            self.new_user_info["text"] = f'Selected Classified Patch Directory\n{dir_name}'
        self.__show_radio_buttons()
    
    def __change_selected_video(self):
        filename = fd.askopenfilename(title='Select a Video', initialdir='/')
        self.new_user_info["text"] = f'Selected File\n{filename}'
        self.load_patches = False
        self.load_classified_patches = False
        self.controller.set_in_path(filename)
        self.__show_radio_buttons()
    
    def __change_selected_directory(self):
        dir_name = fd.askdirectory(title='Select a Directory with already extracted Patches', initialdir='/')
        self.new_user_info["text"] = f'Selected Patch Directory\n{dir_name}'
        self.load_patches = True
        self.load_classified_patches = False
        self.controller.set_in_path(dir_name)
        self.__show_radio_buttons()
    
    def __change_selected_classified_directory(self):
        dir_class_name = fd.askdirectory(title='Select a Directory with already classified Patches', initialdir='/')
        self.new_user_info["text"] = f'Selected Classified Patches Directory\n{dir_class_name}'
        self.load_patches = False
        self.load_classified_patches = True
        self.controller.set_in_path(dir_class_name)
        self.__show_radio_buttons()
    
    def __clear_history(self): # needed to discard changes, if there is time: TODO: make code object oriented
        vars_clear = [self.progress_frame, self.pb, self.pb_label, self.pb_2, self.pb_2_label, self.open_video_button, self.video_frame, self.button_back_to_start, self.new_user_info, self.run_button, self.open_folder_button, self.open_classified_folder_button, self.classifiers, self.merge_strategies, self.detectors, self.radio_buttons]
        for var in vars_clear:
            if var is not None:
                var.pack_forget()
                var = None
        self.pb_var = IntVar()
        self.pb_var_1 = IntVar()
        self.pb_var_2 = IntVar()
        self.pb_max_len = 0
        self.pb_2_max_len = 0
        self.curr_frame.pack_forget()
        self.curr_frame = self.start_win
        self.curr_user_info = self.user_info
        self.func = "start"
        self.detector_idx = StringVar()
        self.merge_strategy_idx = StringVar()
        self.detectory_idx = StringVar()
        self.load_patches = False
        self.load_classified_patches = False
    
    def __unpack_previous_frame(self):
        # user pressed run button --> clear everything but the heading
        vars_to_forget = [self.open_video_button, self.video_frame, self.button_back_to_start, self.run_button, self.new_user_info, self.open_folder_button, self.open_classified_folder_button, self.classifiers, self.merge_strategies, self.detectors, self.radio_buttons]
        for vari in vars_to_forget:
            if vari is not None:
                vari.pack_forget()
        if self.func == "detect":
            self.curr_user_info["text"] = "DETECTION"
        elif self.func == "classify":
            self.curr_user_info["text"] = "CLASSFICATION"
        if self.func == "reconstruct":
            self.curr_user_info["text"] = "RECONSTRUCTION"
        
    
    def __visualize_classification(self):
        # TODO
        return
            
    def show_progress(self, len, func):
        if func == "Detecting":
            self.progress_frame = ttk.Frame(self.curr_frame)
            self.progress_frame.columnconfigure(0, weight=1)
            self.progress_frame.rowconfigure(0, weight=1)
            self.progress_frame.pack(padx=30, pady = 10)
            self.pb = ttk.Progressbar(
                self.progress_frame, orient=HORIZONTAL, mode='determinate', length=100, variable = self.pb_var)
            self.pb_max_len = len
            self.pb.pack()
            self.pb_label = ttk.Label(self.progress_frame, text=f"{func}...0%")
            self.pb_label.pack()
        elif func == "Reconstructing":
            if self.progress_frame is None:
                self.progress_frame = ttk.Frame(self.curr_frame)
                self.progress_frame.columnconfigure(0, weight=1)
                self.progress_frame.rowconfigure(0, weight=1)
                self.progress_frame.pack(padx=30, pady = 10)
            self.pb_2 = ttk.Progressbar(
                self.progress_frame, orient=HORIZONTAL, mode='determinate', length=100, variable = self.pb_var_2)
            self.pb_2_max_len = len
            self.pb_2.pack()
            self.pb_2_label = ttk.Label(self.progress_frame, text=f"{func}...0%")
            self.pb_2_label.pack()        
    
    def update_pb_value(self, curr_value, func):
        if func == "Detecting" and self.pb_label is None:
            return
        if func == "Reconstructing" and self.pb_2_label is None:
            return
        if func == "Detecting":
            pb_max_len = self.pb_max_len
            pb_label = self.pb_label
            pb_var = self.pb_var
        else:
            pb_max_len = self.pb_2_max_len
            pb_label = self.pb_2_label
            pb_var = self.pb_var_2
        val = int((curr_value/pb_max_len)*100)
        pb_var.set(val)
        pb_label["text"]=f"{func}...{val}%"
    
    def show_undefined_progress(self, func):
        if self.progress_frame is None:
            self.progress_frame = ttk.Frame(self.curr_frame)
            self.progress_frame.columnconfigure(0, weight=1)
            self.progress_frame.rowconfigure(0, weight=1)
            self.progress_frame.pack()
        self.pb = ttk.Progressbar(self.progress_frame, orient=HORIZONTAL, mode='indeterminate', length=100, variable=self.pb_var_1)
        self.pb.pack(padx=30, pady = 10)
        self.pb_var_1.set(20)
        self.pb.start()
        self.pb_label = ttk.Label(self.progress_frame, text=f"{func}...")
        self.pb_label.pack()
        
    def finish_detection_progress(self, user_text, home_button):
        self.pb_var.set(100)
        self.pb_label["text"]=f"Finished detection! {user_text}"
        if home_button:
            self.button_back_to_start.pack()
    
    def finish_classification_progress(self, user_text, home_button):
        self.pb_var_1.set(100)
        self.pb.stop()
        self.pb["mode"]='determinate'
        self.pb_label["text"]=f"Finished classification! {user_text}"
        if home_button:
            self.button_back_to_start.pack()
        
        self.visualize_classification_button = ttk.Button(self.progress_frame, text="Visualize Classification Results", command=self.__visualize_classification)
    
    def finish_reconstruction_progress(self, user_text):
        self.pb_var_2.set(100)
        self.pb_2_label["text"]=f"Finished detection! {user_text}"
        self.button_back_to_start.pack()
    
    def error_in_progress(self, exception):
        print(exception)
        messagebox.showinfo("Error", f"Whoooops, there has been an error:\n{exception}\n\nPlease ensure your video contains detectable faces and you specified the correct directories.")
        self.__show_start_window()
    
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
