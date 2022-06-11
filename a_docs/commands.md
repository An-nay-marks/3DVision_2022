# For EULER:
- connect to VPN
- login via: `ssh <username>@euler.ethz.ch`
- direct to project directory:
        `cd /cluster/project/infk/courses/252-0579-00L/group11/gitrepo/3DVision_2022/`
- run example command:
        `bsub -R "rusage[mem=10000,ngpus_excl_p=8]" "python main.py -f reconstruction -s video.mp4 -r testrun"`
        The last part always consists of the usual command line instruction
- check for job: `bjobs`
- an output logfile will be created automatically in the project ROOT folder for each run, there you can also check for error messages

# For Code:
- only do reconstruction with mean optimization: `python reconstruct.py -r "test_reconstruction_mean" -lc <path_to_samples> -op "mean"`
- only do reconstruction with mean shape optimization: `python reconstruct.py -r "test_reconstruction_mean_shape" -lc "<path_to_samples> -op "mean_shape"`
- classification with extracted patches: `python classify.py -lr ""