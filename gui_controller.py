from threading import Thread
from detection import scrfd
from detect import initialize_detector
from classify import initialize_classifier
from pipeline.offline_pipeline import OfflinePipeline
from utils_3DV import *

class Controller():
    def __init__(self, view):
        self.view = view
        self.model_thread = None
        # for global info from the view
        self.in_path = ""
        self.curr_function = ""
    
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
        self.curr_function = "Detecting"
        self.model_thread = Thread(target = self.__detect)
        self.model_thread.start()
        return
    
    def classify_gui(self):
        self.model_thread = Thread(target=lambda:self.__classify())
        self.model_thread.start()
    
    def set_in_path(self, in_path):
        self.in_path = in_path
    
    def __detect(self, home_button=True, save_patches=True, user_text_on_finish = None):
        self.curr_function = "Detecting"
        try:
            detector = scrfd.SCRFaceDetector(f'{ROOT_DIR}/data/model_files/scrfd_34g.onnx')
            provider = initialize_video_provider(self.in_path)
            run_name = get_current_datetime_as_str()
            pipeline = OfflinePipeline(provider, run_name, None, detector)
            self.view.show_progress(pipeline.get_source(), self.curr_function)
            out_path = pipeline.target_dir
            pipeline.detect(self, save_patches)
            if user_text_on_finish is None:
                user_text_on_finish = f"You can find your patches in\n{out_path}"
            self.view.finish_detection_progress(user_text=user_text_on_finish, home_button=home_button)
            return pipeline, run_name, out_path
        except Exception as e:
            self.view.error_in_progress(e)
    
    def __classify(self):
        classifier_idx = int(self.view.classifier_idx.get())
        classifier_name = OFFLINE_CLASSIFIERS[classifier_idx]
        use_patches = self.view.load_patches
        try:
            classifier = initialize_classifier(classifier_name)
            if not use_patches:
                pipeline, run_name, out_dir = self.__detect(home_button=False, save_patches = False, user_text_on_finish="")
                pipeline.classifier = classifier
                pipeline.source = out_dir
                load_patches = False
            else:
                data_src = self.in_path
                run_name = get_current_datetime_as_str()
                pipeline = OfflinePipeline(data_src, run_name, None, None, classifier)
                load_patches = True
            self.curr_function = "Classifying"
            out_path = pipeline.target_dir
            self.view.show_undefined_progress(self.curr_function)
            pipeline.classify(load_patches=load_patches)
            pipeline.save_classification()
            self.view.finish_classification_progress(out_path, home_button=True)
        except Exception as e:
            print(e)
            self.view.error_in_progress(e)
    
    def __reconstruct(self):
        classifier_idx = int(self.view.classifier_idx.get())
        classifier_name = OFFLINE_CLASSIFIERS[classifier_idx]
        use_patches_classification = self.view.load_patches
        merge_idx = int(self.view.merge_idx.get())
        merge_name = MERGE_STRATEGIES[merge_idx]
        use_patches_reconstruction = self.view.load_patches
    
    def status(self, num_frames):
        self.view.update_pb_value(num_frames, self.curr_function)
    
    def get_classifiers(self):
        return OFFLINE_CLASSIFIERS
    
    def get_detectors(self):
        return DETECTORS
    
    def get_optimizations(self):
        return MERGE_STRATEGIES
        
    