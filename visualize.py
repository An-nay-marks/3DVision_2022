from utils_3DV import *
from pipeline.pipeline_utils import *
from recognition import arcface, vgg
from sklearn.decomposition import PCA
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from classify import initialize_classifier
import random


def classify_and_visualize_for_offline_classifiers(run_name, classifier = None, path_to_raw_patches=None, use_pca=False):
    """Classifies the patches and Visualizes the patches according to their identity. It saves the visualization in the target folder.
    Args:
        classifier (Classifier): classifier name
        path_to_raw_patches (_type_, optional): Path to the folder containing the raw patches. Defaults to None.
        use_pca (bool, optional): Whether to embed the encodings with PCA and visualize the blob. Otherwise it will choose 10 randomly classified images to display in a grid form. Defaults to False.
    """
    classifier = initialize_classifier(classifier)
    # if path_to_raw_patches is None: #TODO: delete after testing
    #   data_src = initialize_video_provider(path_to_classified_patches)
    #   patches, identities = load_classified_patches(data_src)
    #   patches, identities = load_classified_patches(path_to_classified_patches) # delete after testing
    #else:
    target_folder = f"{OUT_DIR}/{run_name}"
    print("Loading Raw Patches ...")
    patches = load_raw_patches(path_to_raw_patches)
    if use_pca:
        patches = np.asarray(patches, dtype=object)
        print("Encoding Raw Patches ...")
        if isinstance(classifier, vgg.VGGFaceClassifier):
            # arcface doesn't have explicit encoding
            encodings = arcface.ArcFaceR100(f'{ROOT_DIR}/data/model_files/arcface_r100.pth').encode(patches)
        else:
            encodings = classifier.encode_all(patches)
        print("Classifying Encodings ...")
        identities, best_idx = classifier.classify_all(patches)
        patches = patches[best_idx]
        identities = identities[best_idx]
        unique_ids = np.unique(identities)
        print(f"Plotting patches for {len(unique_ids)} classes ...")
        for unique_id in tqdm(unique_ids):
            unique_id_broadcasted = np.full(fill_value=unique_id, shape=len(identities))
            face_mask = np.where(identities == unique_id_broadcasted, True, False)
            unique_encodings = np.asarray(encodings)[face_mask].tolist()
            unique_patches = np.asarray(patches)[face_mask].tolist()
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(unique_encodings)
            
            # some matplotlib settings
            plt.clf()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.grid(False)
            ax.set_xticks([]) #get rid of lines
            ax.set_yticks([])
            min_x = min(embedding[:,0])
            max_x = max(embedding[:,0])
            min_y = min(embedding[:,1])
            max_y = max(embedding[:,1])
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            
            for i in range(len(embedding)):
                # display every image at its embedded coordinate in the plot
                x = embedding[i, 0]
                y = embedding[i, 1]
                bb = Bbox.from_bounds(x, y, 2, 2)
                bb2 = TransformedBbox(bb, ax.transData)
                bbox_image = BboxImage(bb2,
                                    norm=None,
                                    origin=None,
                                    clip_on=False)
                bbox_image.set_data(unique_patches[i][:,:,[2,1,0]]) # convert channels from opencv to matplotlib
                ax.add_artist(bbox_image)
            plt.axis('off')
            #plt.show()
            plt.savefig(f"{target_folder}/id_{unique_id}_visualization")
    else:
        # no embedding --> just visualize in a grid 20 random images
        patches = np.asarray(patches, dtype=object)
        identities, best_idx = classifier.classify_all(patches)
        #patches = patches[best_idx]
        #identities = identities[best_idx]
        fig, plt = __get_plt_for_grid_vis(patches, identities)
        plt.axis('off')
        plt.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        # Selecting the axis-Y making the right and left axes False
        plt.tick_params(axis='y', which='both', right=False,
                        left=False, labelleft=False)
        # Iterating over all the axes in the figure
        # and make the Spines Visibility as False
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)
        plt.savefig(f"{target_folder}/grid_visualization_of_classes.png", bbox_inches='tight', pad_inches=0)
        plt.show()
    print(f"...Done! Check out your visualization: {target_folder}")

def visualize_classified_patches_in_grid(target_folder, path_to_classified_patches):
    """Visualizes already classified patches in a grid of class_numbers x 20

    Args:
        target_folder (string): target path
        path_to_classified_patches (_type_): _description_
    """
    patches, identities = load_classified_patches(path_to_classified_patches)
    fig, plt = __get_plt_for_grid_vis(patches, identities)
    plt.axis('off')
    plt.savefig(f"{target_folder}/grid_visualization_of_classes_1.png")

def __get_plt_for_grid_vis(patches, identities):
    unique_identities = np.unique(identities)
    rows = len(unique_identities)
    print(f"Plotting patches for {rows} classes ...")
    fig, axs = plt.subplots(rows, 20)
    for row_idx, unique_id in enumerate(tqdm(unique_identities)):
        unique_id_broadcasted = np.full(fill_value=unique_id, shape=len(identities))
        face_mask = np.where(identities == unique_id_broadcasted, True, False)
        unique_patches = np.asarray(patches)[face_mask].tolist()
        num_imgs_of_class = min(len(unique_patches), 20)
        # pick 20 (or less if less classes) random patches of that class
        random_idxs = [random.randint(0,len(unique_patches)-1) for i in range(num_imgs_of_class)]
        for col_idx in range(20):
            if rows == 1: # matplotlib automatically squeezes dimensions
                ax = axs[col_idx]
            else:
                ax = axs[row_idx, col_idx]
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.grid(False)
            #ax.set_xticks([]) #get rid of lines
            #ax.set_yticks([])
            if col_idx < num_imgs_of_class: # only plot available images
                random_idx = random_idxs[col_idx] # get specific random index
                img = unique_patches[random_idx][:,:,[2,1,0]] # get patch at random index and permute channels from opencv to fit matplotlib
                ax.imshow(img)
            else:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                for spine in ['top', 'right', 'left', 'bottom']:
                    ax.spines[spine].set_visible(False)
    return fig, plt
    

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # classify_and_visualize_for_offline_classifiers(classifier = "agglomerative", run_name="test_samples_visualization", path_to_raw_patches=f"{ROOT_DIR}\\out\\test_samples", use_pca=True)
    classify_and_visualize_for_offline_classifiers(classifier = "vgg", run_name="vgg_classification", path_to_raw_patches=f"{ROOT_DIR}\\out\\test_samples_Deniz", use_pca=False)
    # visualize_classified_patches_in_grid(run_name="test_samples_visualization_grid", path_to_classified_patches=f"{ROOT_DIR}\\out\\test_samples")