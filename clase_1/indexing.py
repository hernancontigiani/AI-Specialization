__author__ = "Hernan Contigiani"

import numpy as np


class Indexer():
    '''Clase para identificación de usuarios'''
    def __init__(self, ids):

        # [15, 12, 14, 10, 1, 2, 1]
        # [4, 5, 3, 1, 2, 0] unique pos
        # [0, 1, 2, 3, 4, 5] unique pos order
        # [15, 12, 14, 10, 1, 2] unique ids en orden original

        _, unique_pos = np.unique(ids, return_index=True)    # Extraigo las pos de los ids unicos

        # Ordeno la posicion para extraer en el mismo orden inicial del array
        uniques_argsort = unique_pos.argsort()
        uniques_pos_sort = unique_pos[uniques_argsort]
        
        # Genero el array de uniques en el orden original
        uniques = ids[uniques_pos_sort]



        idxs = np.arange(uniques.size) # A cada idx le asigno un idx
        
        # Generar un array para contener todos los posibles ids, aquellos
        # no especificados se le asignará el idx = -1        
        id2idx = np.ones(uniques.max() + 1, dtype=np.int64) * -1
        id2idx[uniques] = idxs

        self.id2idx = id2idx
        self.idx2id = uniques

    def get_user_id(self, idx):
        return self.idx2id[idx]
    
    def get_user_idx(self, id):
        return self.id2idx[id]
