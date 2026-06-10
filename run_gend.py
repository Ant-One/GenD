from run_exp import entry
from src.exp.gend import experiments

if __name__ == "__main__":
    # Récupère tous les noms d'expériences de gend.py
    exp_list = list(experiments.keys())
    print(f"Lancement de : {exp_list}")
    entry(exp_names=exp_list, test=True)