from run_exp import entry
from src.exp.gend_clip import experiments

if __name__ == "__main__":
    # Récupère tous les noms d'expériences de gend.py
    exp_list = list(experiments.keys())
    for exp_name in exp_list:
        if "test" in exp_name:
            print(f"Lancement de : {exp_name}")
            entry(exp_names=exp_name, test=True)
        else:
            print(f"Skipped {exp_name} as it is not a test")