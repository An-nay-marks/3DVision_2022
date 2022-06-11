from reconstruction.improving_deca.dataset import augment_patches

if __name__=="__main__":
    augment_patches(downscale = 0.01, start_counter_offset = 0) # include some very bad data, because that is often the case for single frames from videos
    augment_patches(downscale = 0.01, start_counter_offset = 1, flip=True)
    augment_patches(downscale = 0.05, start_counter_offset = 2)
    augment_patches(downscale = 0.05, start_counter_offset = 3, flip=True)
    augment_patches(downscale = 0.2, start_counter_offset = 4)
    augment_patches(downscale = 0.2, start_counter_offset = 5, flip=True)
    augment_patches(downscale = 0.5, start_counter_offset = 6)
    augment_patches(downscale = 0.5, start_counter_offset = 7, flip=True)
    augment_patches(downscale = 0.8, start_counter_offset = 8)
    augment_patches(downscale = 0.8, start_counter_offset = 9, flip=True)