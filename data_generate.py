import os
import pandas as pd

# Función para encontrar rutas a las subcarpetas que contienen imágenes y proyecciones
def find_subfolder_paths(case_path, dose_type, folder_type):
    for root, dirs, files in os.walk(case_path):
        for dir_name in dirs:
            # Comprobar si la carpeta contiene el tipo y la dosis especificados
            if dose_type.lower() in dir_name.lower() and folder_type.lower() in dir_name.lower():
                return os.path.join(root, dir_name)
    return None

# Ruta principal donde se encuentran los casos
root_path = '/home/data/LDCT/manifest-1648648375084/LDCT-and-Projection-data/'

# Inicializar una lista para almacenar la información de cada caso
data = []

# Iterar sobre cada caso en la carpeta principal
for case_name in os.listdir(root_path):
    case_path = os.path.join(root_path, case_name)
    if os.path.isdir(case_path):  # Asegurarse de que solo se procesen directorios
        # Encontrar rutas para cada tipo de carpeta
        full_dose_images_path = find_subfolder_paths(case_path, 'Full Dose', 'Images')
        low_dose_images_path = find_subfolder_paths(case_path, 'Low Dose', 'Images')
        full_dose_projections_path = find_subfolder_paths(case_path, 'Full Dose', 'Projections')
        low_dose_projections_path = find_subfolder_paths(case_path, 'Low Dose', 'Projections')
        
        # Agregar la información al DataFrame
        data.append([case_name, full_dose_images_path, low_dose_images_path, full_dose_projections_path, low_dose_projections_path])

# Crear el DataFrame con los datos recopilados
df = pd.DataFrame(data, columns=['Case', 'SDCT', 'LDCT', 'SDRAW', 'LDRAW'])

import os 

if not os.path.isdir('./CombinationDataset'):
    os.mkdir('./CombinationDataset')

df.to_csv('./CombinationDataset/train.txt', sep='\t', header=False, index=False)

if not os.path.isdir('./DefaultDataset'):
    os.mkdir('./DefaultDataset')

df = df.drop(['LDRAW', 'SDRAW'], axis=1)
df.to_csv('./DefaultDataset/train.txt', sep='\t', header=False, index=False)
