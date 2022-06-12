from threading import Thread
from detection import scrfd
from detect import initialize_detector
from classify import initialize_classifier
from reconstruct import initialize_deca
from pipeline.offline_pipeline import OfflinePipeline
from utils_3DV import *

class Controller():
    def __init__(self, view):
        self.view = view
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
        model_thread = Thread(target = self.__detect)
        model_thread.start()
        return
    
    def classify_gui(self):
        self.curr_function = "Classifying"
        model_thread = Thread(target = self.__classify)
        model_thread.start()
    
    def reconstruction_gui(self):
        self.curr_function = "Reconstructing"
        model_thread = Thread(target = self.__reconstruct)
        model_thread.start()
    
    def set_in_path(self, in_path):
        self.in_path = in_path
    
    def __detect(self, home_button = True, save_patches = True, user_text_on_finish = None):
        self.curr_function = "Detecting"
        try:
            detector = scrfd.SCRFaceDetector(f'{ROOT_DIR}/data/model_files/scrfd_34g.onnx')
            provider = initialize_video_provider(self.in_path)
            run_name = get_current_datetime_as_str()
            pipeline = OfflinePipeline(provider, run_name, None, detector)
            self.view.show_progress(pipeline.get_source_detect(), "Detecting")
            out_path = pipeline.target_dir
            pipeline.detect(self, save_patches)
            if user_text_on_finish is None:
                user_text_on_finish = f"You can find your patches in\n{out_path}"
            self.view.finish_detection_progress(user_text=user_text_on_finish, home_button=home_button)
            return pipeline, run_name
        except Exception as e:
            self.view.error_in_progress(e)
    
    def __classify(self, home_button=True, save_classification = True, user_text_on_finish = None):
        classifier_idx = int(self.view.classifier_idx.get())
        classifier_name = OFFLINE_CLASSIFIERS[classifier_idx]
        use_patches = self.view.load_patches
        try:
            classifier = initialize_classifier(classifier_name)
            if not use_patches:
                pipeline, run_name = self.__detect(home_button=False, save_patches = False, user_text_on_finish="")
                pipeline.classifier = classifier
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
            if save_classification:
                pipeline.save_classification()
            if user_text_on_finish is None:
                user_text_on_finish = f"You can find your classified patches in\n{out_path}"
            self.view.finish_classification_progress(user_text = user_text_on_finish, home_button=home_button)
            return pipeline, run_name
        except Exception as e:
            print(e)
            self.view.error_in_progress(e)
    
    def __reconstruct(self):
        merge_idx = int(self.view.merge_strategy_idx.get())
        merge_name = MERGE_STRATEGIES[merge_idx]
        use_classified_patches_instead_of_classification = self.view.load_classified_patches
        try:
            deca = initialize_deca(merge_name)
            if use_classified_patches_instead_of_classification:
                data_src = self.in_path
                run_name = get_current_datetime_as_str()
                pipeline = OfflinePipeline(data_src, run_name, None, deca=deca)
            else:
                pipeline, run_name = self.__classify(home_button=False, save_classification=False, user_text_on_finish="")
                pipeline.deca = deca
            self.view.show_progress(pipeline.get_source_reconstruct(), "Reconstructing")
            pipeline.reconstruct(self)
            out_path = pipeline.target_dir
            user_text_on_finish = f"You can find your reconstructions in \n{out_path}"
            self.view.finish_reconstruction_progress(user_text = user_text_on_finish)
        except Exception as e:
            print(e)
            self.view.error_in_progress(e)
         
    
    def status(self, num_frames):
        self.view.update_pb_value(num_frames, self.curr_function)
    
    def get_classifiers(self):
        return OFFLINE_CLASSIFIERS
    
    def get_detectors(self):
        return DETECTORS
    
    def get_merge_strategies(self):
        return MERGE_STRATEGIES
    
    def exit(self):
        os._exit(1)
        
    