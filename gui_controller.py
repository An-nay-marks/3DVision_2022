from threading import Thread
from time import sleep

class Controller():
    def __init__(self, view):
        self.view = view
        self.model = None # change model according to wanted functionality
        self.model_thread = None
        # for global info from the view
        self.in_path = ""
        self.out_path = ""
    
    # def monitor_model_thread(self):
    #     if self.model_thread.is_alive():
    #         self.view.show_detection_progress()
    #     else:
    #         print(self.model_output_message)
    #         if self.model_output_message == "Done":
    #             self.view.finish_detection_progress(self.model_output)
    #         else:
    #             self.view.error_in_progress(self.model_output_message)
    
    def detection_gui(self):
        self.model_thread = Thread(target = self.__detect)
        self.model_thread.start()
        # self.monitor_model_thread()
        return
    
    def set_in_path(self, in_path):
        self.in_path = in_path
    
    def set_out_path(self, out_path):
        self.out_path = out_path
    
    def __detect(self):
        try:
            #TODO: call pipeline stuff
            
            self.view.show_detection_progress(50000)
            for i in range(50000):
                print("bla")
                if i%25000 == 0:
                    self.view.update_pb_value(i)
            self.view.finish_detection_progress(self.out_path)
        except Exception as e:
            self.view.error_in_progress(e.message)
    