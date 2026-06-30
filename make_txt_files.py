import os
import subprocess

def run_bash_commands():
    base_dir = "/home/antoine/gend-datasets/datasets/test/CDFv3-x1.3-th0.5-all/real"
    output_dir = "/home/antoine/GenD/config/datasets/test-from-gend/CDFv3"
    
    # Étape 1 : Créer le dossier de destination si nécessaire
    # (Obligatoire, sinon la redirection bash '>' retournera une erreur "No such file or directory")
    os.makedirs(output_dir, exist_ok=True)

    # Étape 2 : Vérifier que le dossier source existe
    if not os.path.exists(base_dir) or not os.path.isdir(base_dir):
        print(f"Erreur : Le dossier '{base_dir}' est introuvable.")
        return

    # Étape 3 : Parcourir le dossier 'fake'
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # S'assurer qu'il s'agit bien d'un sous-répertoire
        if os.path.isdir(subdir_path):
            
            # Étape 4 : Construire la commande Bash exacte
            # Les chemins sont mis entre guillemets pour éviter que le shell ne plante 
            # si un nom de dossier contient des espaces.
            cmd = f'find "{base_dir}/{subdir}"/* -type f | sort > "{output_dir}/CDFv3_{subdir}_real.txt"'
            
            print(f"Exécution de la commande : {cmd}")
            
            # Étape 5 : Exécuter la commande dans le shell
            # shell=True est nécessaire pour interpréter le pipe (|), le glob (*) et la redirection (>)
            result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True, cwd="/")
            
            # Affichage des erreurs éventuelles générées par Bash
            if result.returncode != 0:
                print(f"Erreur lors de l'exécution pour '{subdir}':\n{result.stderr}")

if __name__ == "__main__":
    run_bash_commands()
