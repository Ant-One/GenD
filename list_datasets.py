from pathlib import Path

def print_file_list(directory_path):
    # Définir le chemin du dossier
    dossier = Path(directory_path)
    
    # Vérifier que le dossier existe bien
    if not dossier.exists() or not dossier.is_dir():
        print(f"Erreur : Le dossier '{directory_path}' est introuvable.")
        return []

    # Créer la liste Python en filtrant pour ne garder que les fichiers (exclut les sous-dossiers)
    # .name récupère uniquement le nom du fichier (ex: 'image.png'), pas le chemin complet
    liste_fichiers = [fichier.name for fichier in dossier.iterdir() if fichier.is_file()]
    
    # Imprimer la liste de façon lisible, ligne par ligne (façon code Python)
    print("fichiers = [")
    for nom in liste_fichiers:
        print(f'    "config/datasets/test-from-gend/CDFv3/{nom}",')
    print("]")
    
    # Retourne la liste pour pouvoir l'utiliser ailleurs dans votre code si besoin
    return liste_fichiers

if __name__ == "__main__":
    # Remplacez "nom_de_votre_dossier" par le chemin réel de votre dossier
    print_file_list("config/datasets/test-from-gend/CDFv3")
