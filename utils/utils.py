import os
import pydicom
import numpy as np
import pandas as pd

from PIL import Image
from functools import partial
from multiprocessing import Pool, cpu_count

'''
lot_id Function

Generates unique identifiers for each multi-file image split lot.

Inputs:
	- df: (pd.DataFrame) Dataframe containing the repeated values
	- case_column: (String) Name of column to split repeated values
	- number_column: (String) Column containing path-list.
	
Output:
	- Encoded Identifier: (String) Encoded ID. Info:
		- Case		<I>
		- Split 	<S>
		- First File	<F>
		- Last File	<T>
		- Slice Count	<C>
	
Example: IN001S5F12T16C5
	- Case Index N001
	- Split 5
	- From File 12
	- To File 16
	- Counting 5 Slices
'''
def lot_id(df, case_column, number_column):
    # Agrupar por la columna 'Case' y calcular el primer y último elemento para la columna 'Number'
    grouped = df.groupby(case_column)
    
    # Iterar sobre cada grupo
    for case, group in grouped:
        for i, row in group.iterrows():
            # Obtener el primer y último elemento de la lista
            first_elem = os.path.basename(row[number_column][0]).split('.')[0]
            last_elem = os.path.basename(row[number_column][-1]).split('.')[0]
            # Generar el nuevo nombre
            new_name = f"I{case}S{i}F{first_elem}T{last_elem}C{len(row[number_column])}"
            # Asignar el nuevo nombre a la columna 'NewName'
            df.at[i, case_column] = new_name
    return df

'''
n_slice_split Function

Returns every n-consecutive-path combination from a given directory

Inputs:
	- directory: (String) Path to directory.
	- split: (Int) Slice count n.
	
Outputs:
	- lst (list<list<String>>) List containing every n-path consecutive combination.
'''
def n_slice_split(directory, split=3):
    if os.path.exists(directory) and os.path.isdir(directory):
        lst = sorted([os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])
        if split < 0:
            split = max(len(lst), 1)
        return [lst[i:i+split] for i in range(len(lst) - split + 1)]
    else:
        return []
        
'''
load_image Method

Single-file image loader.

Inputs:
	- path: (String) Path to file.
	- id: (String) Image identification name or id, default=None.
	
Outputs:
	- data: (dict)
		Keys:
			- Image: (ndarray) Loaded image.
			- Metadata: (dict) File metadata.
			- Id: (String) Image Id.
'''
def load_image(path, id=None):
	ext = path[-4:]
	
	if ext == '.dcm':
		image = pydicom.dcmread(path)
		
		# Ignore images and missing keys
		metadata = {str(element.name): str(element.value) \
			for element in image \
			if element.name != 'Pixel Data'}
			
		return {'Image': image.pixel_array,
			'Metadata': metadata,
			'Id': id if id else path}
			
	elif ext in ['.npz', '.npy']:
		return {'Image': np.load(path),
			'Metadata': None,
			'Id': id if id else path}
			
	elif ext in ['.pt', '.pth']:
		return {'Image': torch.load(path).numpy(),
			'Metadata': None,
			'Id': id if id else path}
	else:
		return {'Image': np.array(Image.open(path)),
			'Metadata': None,
			'Id': id if id else path}
	
'''
load_composite Method

Multi-file image loader.

Inputs:
	- path_list: (list<String>) Paths to files.
	- id: (String) Image set identification name or id, default=None.
	- dim: (Int) Image dimension for multi-file input. Only 2 or 3 allowed. Default=3.
	- metadata: (String) Keep Metadata. Options: 'first', 'last', None.
	
Outputs:
	- data: (dict)
		Keys:
			- Image: (ndarray) Loaded stack image.
			- Metadata: (dict) File metadata. 
			- Id: (String) Image set identification name or id.
'''
def load_composite(path_list, id=None, dim: int=3, metadata='first', multi_cpu=False):
	assert dim in (2,3), "Dimension dim in load() must be an integer between 2 and 3"
	assert metadata in ['first', 'last', None], f"Metadata option unavailable: {metadata}"
	
	files = []

	# Cargar imágenes en paralelo
	if multi_cpu:
		with Pool(processes=cpu_count()) as pool:
			# Leer las imágenes en paralelo
			files = pool.map(load_image, path_list)	
	else:
		files = [load_image(f) for f in path_list]

	# Filtrar los resultados nulos (si algún archivo no se pudo cargar)
	files = [f for f in files if f is not None]
			
	# Ordenar las imágenes por 'Id' en orden ascendente
	files.sort(key=lambda x: x['Id'])
	
	if metadata:
		metadata = files[0 if metadata=='first' else -1]['Metadata']

	# Extraer las imágenes en el orden correcto
	files = [f['Image'] for f in files]

	image = np.stack(files) if dim == 3 else np.hstack(files)
		
	return {'Image': image,
		'Metadata': metadata,
		'Id': id}
		
'''
load Method

Default image loader.

Inputs:
	- path: (String or list<String>) Path to files or directory.
	- id: (String) Image identification name or id.
	- dim: (Int) Image dimension for multi-file input. Only 2 or 3 allowed. Default=3.
	
Outputs:
	- data: (dict)
		Keys:
			- Image: (ndarray) Loaded stack image.
			- Metadata: (dict) File metadata. Keeps last.
			- Id: (String) Image set identification name or id.
'''

def load(path, id, dim: int=3):
	if type(path) == str:
		if os.path.isdir(path):
			files = [os.path.join(path, name) for name in os.listdir(path)]
			return load_composite(files, id, dim)
		return load_image(path, id)
	return load_composite(path, id, dim)
