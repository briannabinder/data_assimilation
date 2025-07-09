
filter = "KDE" # SKDE

# exp_id = "TOY_01"
# ensemble_sizes = [50, 100, 200, 250, 300, 500, 1000]
# h_x_mins       = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# h_x_maxs       = [3, 5, 10, 15, 20, 25, 35, 50]
# h_ys           = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

exp_id = "L63_03"
ensemble_sizes = [250, 500, 1000]
h_x_mins       = [0.1, 0.2, 0.5, 1, 2, 5, 10]
h_x_maxs       = [30, 60, 100, 300, 500]
h_ys           = [0.1, 0.5, 1, 2, 5, 10, 12, 15, 20, 40]

def print_job_header(job_count):
    print("#!/bin/bash", file=f)
    print("#SBATCH --partition=epyc-64", file=f) # epyc-64, main
    #print('#SBATCH --constraint=epyc-7542' ,file=f)
    print("#SBATCH --nodes=1", file=f)
    print("#SBATCH --ntasks=1", file=f)
    print("#SBATCH --cpus-per-task=8", file=f)
    # print('#SBATCH --gpus-per-task=a40:1' ,file=f)
    print("#SBATCH --mem=48GB", file=f) 
    print("#SBATCH --time=48:00:00", file=f)
    print(f"#SBATCH --output=./exps/{exp_id}/outputs/output{job_count}.out", file=f)
    #print('#SBATCH --mail-type=all', file=f)
    #print('#SBATCH --mail-user=bjbinder@usc.edu', file=f)
    print("",file=f)

    # print("module purge", file=f)
    # print("module load legacy/CentOS7", file=f)
    # print("module load gcc/8.3.0", file=f)
    # print("module load cuda/11.1-1", file=f)
    # print("module load cudnn/8.0.4.30-11.1", file=f)
    # print("",file=f)

    print('eval "$(conda shell.bash hook)"', file=f)
    print("conda activate da", file=f)
    print("",file=f)

job_count = 1

if job_count == 1:

    f = open(f"./exps/{exp_id}/jobs/job{job_count}.slurm", 'w')
    print_job_header(job_count)

    for ensemble_size in ensemble_sizes:

        print(f"python3 main.py --exp_id {exp_id} ENKF --ensemble_size {ensemble_size}", file=f)
        print(f"python3 main.py --exp_id {exp_id} SIR --ensemble_size {ensemble_size}", file=f)
        
    print(f"python3 main.py --exp_id {exp_id} SIR --ensemble_size 100000", file=f)

    print("",file=f)
    f.close()
    job_count = job_count + 1

for h_y in h_ys:
    for h_x_min in h_x_mins:
        for h_x_max in h_x_maxs:

            f = open(f"./exps/{exp_id}/jobs/job{job_count}.slurm", 'w')
            print_job_header(job_count)

            for ensemble_size in ensemble_sizes:

                if filter == 'KDE':
                    print(f"python3 main.py --exp_id {exp_id} {filter} --ensemble_size {ensemble_size} --h_x_min {h_x_min} --h_x_max {h_x_max} --h_y {h_y}", file=f)
                elif filter == 'SKDE':
                    print(f"python3 main.py --exp_id {exp_id} {filter} --ensemble_size {ensemble_size} --h_x_min {h_x_min} --h_x_max {h_x_max} --h_y {h_y}", file=f)

            print("",file=f)
            f.close()
            job_count = job_count + 1

print(f"{job_count-1} jobs saved at ./exps/{exp_id}/jobs/")
print("Done!")

# Useful CARC commands
# watch -n 1 myqueue 
# sbatch ./exps/exp_id/jobs/train1.slurm <- To run on CARC
# for i in {1..100}; do sbatch "./exps/L63_03/jobs/job$i.slurm"; done