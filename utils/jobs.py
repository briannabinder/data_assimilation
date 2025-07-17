import os, argparse, shutil, yaml

def print_job_header(file, log_path, partition, mem):

    print("#!/bin/bash", file=file)
    print(f"#SBATCH --partition={partition}", file=file) # epyc-64, main, gpu
    # print('#SBATCH --constraint=epyc-7542' ,file=file)
    print("#SBATCH --nodes=1", file=file)
    print("#SBATCH --ntasks=1", file=file)
    print("#SBATCH --cpus-per-task=8", file=file)
    # print('#SBATCH --gpus-per-task=a40:1' ,file=file)
    print(f"#SBATCH --mem={mem}GB", file=file) 
    print("#SBATCH --time=48:00:00", file=file)
    print(f"#SBATCH --output={log_path}", file=file)
    # print('#SBATCH --mail-type=all', file=file)
    # print('#SBATCH --mail-user=bjbinder@usc.edu', file=file)
    print("",file=file)

    print('eval "$(conda shell.bash hook)"', file=file)
    print("conda activate da", file=file)
    print("",file=file)

def print_baseline_jobs(file, exp_id, ensemble_sizes, include_truth=True):

    for ensemble_size in ensemble_sizes:
        print(f"python3 main.py --exp_id {exp_id} ENKF --ensemble_size {ensemble_size}", file=file)
        print(f"python3 main.py --exp_id {exp_id} SIR --ensemble_size {ensemble_size}", file=file)
    
    if include_truth:
        print(f"python3 main.py --exp_id {exp_id} SIR --ensemble_size 100000", file=file)

def print_kde_jobs(file, exp_id, ensemble_sizes, h_x_min, h_x_max, h_y):

    for ensemble_size in ensemble_sizes:
        print(f"python3 main.py --exp_id {exp_id} KDE --ensemble_size {ensemble_size} --h_x_min {h_x_min} --h_x_max {h_x_max} --h_y {h_y}", file=file)

def print_postprocess_jobs(file, exp_id):

    if exp_id == "L63_01" or exp_id == "L63_02" or exp_id == "L63_03":
        print(f"python3 utils/statistics.py --exp_id {exp_id} --stat W2", file=file)

    elif exp_id == "L96_01":
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Postprocess experiment results.")
    parser.add_argument("--exp_id", type=str, help="Experiment ID (e.g., L63_03)")
    parser.add_argument("--filter", type=str, default="KDE", help="Filter name (default: KDE)")
    parser.add_argument("--partition", type=str, default="gpu", help="CARC partition")
    parser.add_argument("--mem", type=int, default=48, help="CARC max memory")
    args = parser.parse_args()
    
    exp_id, filter_name = args.exp_id, args.filter
    partition, mem = args.partition, args.mem
    exp_dir = f"./exps/{exp_id}/"

    config_path = exp_dir + "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        ensemble_sizes = config['sweep_params']['ensemble_sizes']

    jobs_dir = exp_dir + "jobs/"
    os.makedirs(jobs_dir, exist_ok=True)

    logs_dir = exp_dir + "logs/"
    os.makedirs(logs_dir, exist_ok=True)

    job_count = 1

    if job_count == 1:
        file = open(jobs_dir + f"job{job_count}.slurm", 'w')
        print_job_header(file, logs_dir + f"log{job_count}.slurm", args.partition, args.mem)
        print_baseline_jobs(file, exp_id, ensemble_sizes)
        file.close()
        job_count = job_count + 1
    
    if filter_name == "KDE":
        for h_y in config['sweep_params']['kde']['h_ys']:
            for h_x_min in config['sweep_params']['kde']['h_x_mins']:
                for h_x_max in config['sweep_params']['kde']['h_x_maxs']:
                    file = open(jobs_dir + f"job{job_count}.slurm", 'w')
                    print_job_header(file, logs_dir + f"log{job_count}.slurm", args.partition, args.mem)
                    print_kde_jobs(file, exp_id, ensemble_sizes, h_x_min, h_x_max, h_y)
                    file.close()
                    job_count = job_count + 1

    elif filter_name == "NNFLOW":
        pass
    else:
        raise ValueError(f"Unsupported filter: {filter_name}")

    # POSTPROCESS
    file = open(jobs_dir + f"postprocess.slurm", 'w')
    print_job_header(file, logs_dir + f"postprocess.slurm", args.partition, args.mem)
    print_postprocess_jobs(file, exp_id)
    file.close()

    print(f"{job_count-1} jobs saved at {jobs_dir}")
    print("Done!")  
    
    # postprocess(args.exp_id, args.filter, args.ensemble_size)
