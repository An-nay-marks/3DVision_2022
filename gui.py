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
        window_height = 700
        font_infotext = ('Comic Sans MS',12,"bold")
        font_button_text = ('Comic Sans MS',10)
        self.root = Tk()
        self.root.title(app_name)
        self.root.protocol("WM_DELETE_WINDOW", self.__on_closing)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.iconbitmap("./a_docs/icon.ico")
        s = ttk.Style()
        s.configure('TFrame', background="black")
        s.configure('TButton', background="white", font=font_button_text)
        s.configure('TLabel', background="black", foreground="white", font=font_infotext)
        # start window
        self.start_win = ttk.Frame(master=self.root, style='TFrame')
        self.user_info = ttk.Label(self.start_win, text="Welcome to the world of facial reconstruction from videos.\nWhat would you like to do with your video?", style="TLabel")
        self.user_info.grid(column=0, row=0, columnspan=3, ipadx = 10, ipady = 10, sticky = EW)
        self.button_detect = ttk.Button(self.start_win, text="Detect Faces", command=self.__detect, style='TButton')
        self.button_classify = ttk.Button(self.start_win, text="Detect and Classify Faces", command=self.__classify, style='TButton')
        self.button_reconstruct = ttk.Button(self.start_win, text="Detect, Classify and Reconstruct Faces", command=self.__reconstruct, style='TButton')
        # self.button_detect.configure(font=font_button_text)
        self.button_detect.grid(column=0, row=1)
        self.button_classify.grid(column=1, row=1)
        self.button_reconstruct.grid(column=2, row=1)
        self.start_win.pack(fill="both", expand=1)
        
        # detection window
        self.detection_win = ttk.Frame(self.root)
        self.detection_start_text = "DETECTION\nPlease select a video:"
        
        # classification window
        self.classify_win = ttk.Frame(self.root)
        self.classification_start_text = "CLASSIFICATION\nPlease specify what you want to do:"
        
        # reconstruction window
        self.reconstruct_win = ttk.Frame(self.root)
        self.reconstruction_start_text = "CLASSIFICATION\nPlease specify what you want to do:"
        
        # all variables that are used (needed to clean screen afterwards)
        self.info_about_the_pipeline = None #TODO
        self.progress_frame = None
        self.pb = None
        self.pb_label = None
        self.pb_var = IntVar()
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
        self.run_button = None
        self.classifier_idx = StringVar()
        self.classifiers = []
        self.merge_strategy_idx = StringVar()
        self.merge_strategies = []
        #TODO: select detector
        self.detectors = []
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
        self.curr_user_info = ttk.Label(self.curr_frame, style="TLabel", text = self.detection_start_text)
        self.curr_user_info.pack()
        self.open_video_button = ttk.Button(self.curr_frame, text='Select a Video', command=self.__select_video)
        self.open_video_button.pack()
        self.__init_return_button()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
        
    def __classify(self):
        """Classification Button pressed"""
        self.curr_frame = self.classify_win
        self.func = "classify"
        self.curr_user_info = ttk.Label(self.curr_frame, style="TLabel", text = self.classification_start_text)
        self.curr_user_info.pack()
        self.open_video_button = ttk.Button(self.curr_frame, text='Select a Raw Video', command=self.__select_video)
        self.open_video_button.pack()
        self.open_folder_button = ttk.Button(self.curr_frame, text='Select Patches Folder', command=self.__select_directory)
        self.open_folder_button.pack()
        self.__init_return_button()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
    
    def __reconstruct(self):
        """Reconstruct Button pressed"""
        self.curr_frame = self.reconstruct_win
        self.func = "reconstruct"
        self.curr_user_info = ttk.Label(self.curr_frame, style="TLabel", text = self.reconstruction_start_text)
        self.curr_user_info.pack()
        self.open_video_button = ttk.Button(self.curr_frame, text='Select a Raw Video', command=self.__select_video)
        self.open_video_button.pack()
        self.open_folder_button = ttk.Button(self.curr_frame, text='Select Patches Folder', command=self.__select_directory)
        self.open_folder_button.pack()
        self.open_classified_folder_button = ttk.Button(self.curr_frame, text='Select Classified Patches Folder', command=self.__select_classified_directory)
        self.open_classified_folder_button.pack()
        self.__init_return_button()
        self.curr_frame.pack(fill="both", expand=1)
        self.start_win.pack_forget()
    
    def __init_return_button(self):
        if self.button_back_to_start is not None: # needed to change location of button
            self.button_back_to_start.pack_forget()
        self.button_back_to_start = ttk.Button(self.curr_frame, text="Back to Home", command=self.__show_start_window, style='TButton')
        self.button_back_to_start.pack()
    
    def __show_start_window(self):
        self.__clear_history()
        self.start_win.pack(fill="both", expand=1)
        self.curr_frame = self.start_win
        self.curr_user_info = self.user_info
        self.func = "start"

    def __info_about_the_project(self):
        #TODO: pop up
        photo = PhotoImage(file='./a_docs/pipeline.png')
        image_label = ttk.Label(
            self.root,
            image=photo,
            padding=5
        )
    
    def __show_run_button(self, fun, text):
        self.run_button = ttk.Button(self.curr_frame, text=text, command=lambda:[fun(), self.button_back_to_start.pack_forget()])
        self.run_button.pack()
    
    def __show_radio_buttons(self):
        if self.func == "classify":
            self.classifiers = []
            for idx, classifier in enumerate(self.controller.get_classifiers()):
                new_radio_button = ttk.Radiobutton(self.curr_frame, text = classifier, value = str(idx), variable = self.classifier_idx)
                new_radio_button.pack(padx = 5, pady = 5)
                self.classifiers.append(new_radio_button)
            self.classifier_idx.set(str(0))
            fun = self.controller.classify_gui
            text = "Run Classification"
            self.__show_run_button(fun, text)
        elif self.func == "reconstruct":
            if not self.load_classified_patches:
                self.classifiers = []
                for idx, classifier in enumerate(self.controller.get_classifiers()):
                    new_radio_button = ttk.Radiobutton(self.curr_frame, text = classifier, value = str(idx), variable = self.classifier_idx)
                    new_radio_button.pack(padx = 5, pady = 5)
                    self.classifiers.append(new_radio_button)
                self.classifier_idx.set(str(0))
            self.merge_strategies = []
            for idx, merge in enumerate(self.controller.get_merge_strategies()):
                new_radio_button = ttk.Radiobutton(self.curr_frame, text = merge, value = str(idx), variable = self.merge_strategy_idx)
                new_radio_button.pack(padx = 5, pady = 5)
                self.merge_strategies.append(new_radio_button)
            self.merge_strategy_idx.set(str(0))
            fun = self.controller.reconstruction_gui
            text = "Run Reconstruction"
            self.__show_run_button(fun, text)
    
    def __select_video(self):
        filename = fd.askopenfilename(title='Select a Video', initialdir='/')
        if filename =="": return
        self.open_video_button["text"] = 'Change Video'
        self.open_video_button["command"] = self.__change_selected_video
        self.controller.set_in_path(filename)
        if self.func == "detect":
            self.curr_user_info["text"] = 'DETECTION\nChange video or run the detection'
            self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}')
            self.new_user_info.pack()
            fun = self.controller.detection_gui
            text = "Run Detection"
            self.__show_run_button(fun, text)
        elif self.func == "classify":
            self.curr_user_info["text"] = 'CLASSIFICATION\nChange your selection or run the classification'
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}')
                self.new_user_info.pack()
            else:
                self.new_user_info["text"] = f'Selected File\n{filename}'
            self.load_patches = False
            if self.classifiers == []:
                self.__show_radio_buttons()
        elif self.func == "reconstruct":
            self.curr_user_info["text"] = 'Reconstruction\nChange your selection and merge strategy or run the reconstruction'
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected File\n{filename}')
                self.new_user_info.pack()
            else:
                self.new_user_info["text"] = f'Selected File\n{filename}'
            self.load_patches = False
            self.load_classified_patches = False
            if self.classifiers == []:
                self.__show_radio_buttons()
    
    def __select_directory(self):
        dir_name = fd.askdirectory(title='Select a Directory with already extracted Patches', initialdir='/')
        if dir_name =="": return
        self.open_folder_button["text"] = 'Change Directory'
        self.open_folder_button["command"] = self.__change_selected_directory
        self.controller.set_in_path(dir_name)
        if self.func == "classify":
            self.curr_user_info["text"] = 'CLASSIFICATION\nChange your selection or run the classification'
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Directory\n{dir_name}')
                self.new_user_info.pack()
            else:
                self.new_user_info["text"] = f'Selected Directory\n{dir_name}'
            self.load_patches = True
            if self.classifiers == []:
                self.__show_radio_buttons()
        elif self.func == "reconstruct":
            self.curr_user_info["text"] = 'Reconstruction\nChange your selection and merge strategy or run the reconstruction'
            if self.new_user_info is None:
                self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Patch Directory\n{dir_name}')
                self.new_user_info.pack()
            else:
                self.new_user_info["text"] = f'Selected Patch Directory\n{dir_name}'
            self.load_patches = True
            self.load_classified_patches = False
            if self.merge_strategies == []:
                self.__show_radio_buttons()
            
    
    def __select_classified_directory(self):
        dir_name = fd.askdirectory(title='Select a Directory with already classified Patches', initialdir='/')
        if dir_name =="": return
        self.open_folder_button["text"] = 'Change Classified Patches Directory'
        self.open_folder_button["command"] = self.__change_selected_classified_directory
        self.controller.set_in_path(dir_name)
        self.curr_user_info["text"] = 'Reconstruction\nChange your selection or run the reconstruction'
        if self.new_user_info is None:
            self.new_user_info = ttk.Label(self.curr_frame, text=f'Selected Classified Patch Directory\n{dir_name}')
            self.new_user_info.pack()
        else:
            self.new_user_info["text"] = f'Selected Classified Patch Directory\n{dir_name}'
        self.load_patches = False
        self.load_classified_patches = True
        if self.classifiers != []:
            self.__unpack_radio_buttons_classifiers()
        if self.merge_strategies == []:
            self.__show_radio_buttons()
    
    def __change_selected_video(self):
        filename = fd.askopenfilename(title='Select a Video', initialdir='/')
        self.new_user_info["text"] = f'Selected File\n{filename}'
        self.load_patches = False
        self.load_classified_patches = False
        self.controller.set_in_path(filename)
    
    def __change_selected_directory(self):
        dir_name = fd.askopenfilename(title='Select a Directory with already extracted Patches', initialdir='/')
        self.new_user_info["text"] = f'Selected Patch Directory\n{dir_name}'
        self.load_patches = True
        self.load_classified_patches = False
        self.controller.set_in_path(dir_name)
    
    def __change_selected_classified_directory(self):
        dir_class_name = fd.askopenfilename(title='Select a Directory with already classified Patches', initialdir='/')
        self.new_user_info["text"] = f'Selected Classified Patches Directory\n{dir_class_name}'
        self.load_patches = False
        self.load_classified_patches = True
        self.controller.set_in_path(dir_class_name)
    
    def __clear_history(self): # needed to discard changes, if there is time: TODO: make code object oriented
        vars_clear = [self.progress_frame, self.pb, self.pb_label, self.pb_2, self.pb_2_label, self.open_video_button, self.button_back_to_start, self.new_user_info, self.run_button, self.curr_user_info, self.open_folder_button, self.open_classified_folder_button]
        for var in vars_clear:
            if var is not None:
                var.pack_forget()
                var = None
        self.pb_var = IntVar()
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
        self.classifiers = []
        self.merge_strategies = []
        self.detectors = []
        self.load_patches = False
        self.load_classified_patches = False
    
    def __unpack_previous_frame(self):
        vars_to_forget = [self.new_user_info, self.run_button, self.open_video_button, self.open_folder_button, self.open_classified_folder_button, *self.classifiers, *self.detectors, *self.merge_strategies]
        for vari in vars_to_forget:
            if vari is not None:
                vari.pack_forget()
    
    def __unpack_radio_buttons_detectors(self):
        for radio_button in self.detectors:
            radio_button.pack_forget()
            self.run_button.pack_forget()
    
    def __unpack_radio_buttons_classifiers(self):
        for radio_button in self.classifiers:
            radio_button.pack_forget()
            self.run_button.pack_forget()
    
    def __unpack_radio_buttons_merge_strategies(self):
        for radio_button in self.merge_strategies:
            radio_button.pack_forget()
            self.run_button.pack_forget()
    
    def __visualize_classification(self):
        # TODO
        return
            
    def show_progress(self, len, func):
        if func == "Detecting":
            self.progress_frame = ttk.Frame(self.curr_frame)
            self.progress_frame.columnconfigure(0, weight=1)
            self.progress_frame.rowconfigure(0, weight=1)
            self.progress_frame.pack()
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
                self.progress_frame.pack()
            self.pb_2 = ttk.Progressbar(
                self.progress_frame, orient=HORIZONTAL, mode='determinate', length=100, variable = self.pb_var_2)
            self.pb_2_max_len = len
            self.pb_2.pack()
            self.pb_2_label = ttk.Label(self.progress_frame, text=f"{func}...0%")
            self.pb_2_label.pack()            
            
        if self.func == "detect":
            self.curr_user_info["text"] = "DETECTION"
            self.__unpack_radio_buttons_detectors()
        elif self.func == "classify":
            self.curr_user_info["text"] = "CLASSFICATION"
            self.__unpack_radio_buttons_detectors()
            self.__unpack_radio_buttons_classifiers()
            self.open_folder_button.pack_forget()
        else:
            self.curr_user_info["text"] = "RECONSTRUCTION"
            self.__unpack_radio_buttons_detectors()
            self.__unpack_radio_buttons_classifiers()
            self.__unpack_radio_buttons_merge_strategies()
            self.open_folder_button.pack_forget()
            self.open_classified_folder_button.pack_forget()
            
        self.__unpack_previous_frame()
    
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
        self.pb = ttk.Progressbar(self.progress_frame, orient=HORIZONTAL, mode='indeterminate', length=100)
        self.pb.pack()
        self.pb.start()
        self.pb_label = ttk.Label(self.progress_frame, text=f"{func}...")
        self.pb_label.pack()
        if self.func == "detect":
            self.curr_user_info["text"] = "DETECTION"
        elif self.func == "classify":
            self.curr_user_info["text"] = "CLASSFICATION"
        else:
            self.curr_user_info["text"] = "RECONSTRUCTION"
        self.__unpack_previous_frame()
        
    def finish_detection_progress(self, user_text, home_button):
        self.pb_var.set(100)
        self.pb_label["text"]=f"Finished detection! {user_text}"
        if home_button:
            self.button_back_to_start.pack()
    
    def finish_classification_progress(self, user_text, home_button):
        self.pb_var.set(100)
        self.pb.stop()
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
