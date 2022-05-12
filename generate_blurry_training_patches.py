from reconstruction.improving_deca.dataset import generate_blurry_patches

if __name__=="__main__":
    generate_blurry_patches(downscale = 0.01, start_counter_offset = 0) # include some very bad data, because that is often the case for single frames from videos
    generate_blurry_patches(downscale = 0.05, start_counter_offset = 1)
    generate_blurry_patches(downscale = 0.2, start_counter_offset = 2)
    generate_blurry_patches(downscale = 0.5, start_counter_offset = 3)
    generate_blurry_patches(downscale = 0.8, start_counter_offset = 4)