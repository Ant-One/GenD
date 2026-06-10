from run_exp import entry
from src.exp.clip import experiments

if __name__ == "__main__":
    exp_list = list(experiments.keys())
    print(f"Lancement du benchmark CLIP : {exp_list}")
    entry(exp_names=exp_list, test=True)