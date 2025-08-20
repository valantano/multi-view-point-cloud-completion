import torch
import numpy as np


class ModuleCollector:
    """
    Used to store the attention weights and indices of a single module of the PointAttN Architecture.
    """

    def __init__(self, name: str, module_infos: dict):
        self.name: str = name
        self.module_infos: dict = module_infos
        self.indices = []       # list  [{'k_v_ids': {'pcd_ref': pcd_name, inds: np.array}, 'q_ids': {...}}, ...] 
        self.weights = []       # list of np.arrays

    def get_infos(self) -> dict:
        """
        :return: dictionary with the information of the modules {'name': name, 'module_infos': {...}, 
        'layers': {0: 'Layer_0 -- weights: (512, 256) -- kv_inds: (256,) - pcd_ref -- q_inds: (256,) - pcd_ref', 1: ...}}
        """
        layers = {}
        for i in range(len(self.weights)):
            layers[i] = f"Layer_{i} -- weights: {self.weights[i].shape} \
                -- kv_inds: {self.indices[i]['k_v_ids']['inds'].shape} - {self.indices[i]['k_v_ids']['pcd_ref']} \
                    -- q_inds: {self.indices[i]['q_ids']['inds'].shape} - {self.indices[i]['q_ids']['pcd_ref']}"
        return {'name': self.name, 'module_infos': self.module_infos, 'layers': layers}

    def __str__(self) -> str:
        """
        String representation of the module collector
        :return: string representation
        """
        header = "--" * 20 + self.name + "--" * 20 + "\n"
        m_info = "--".join([f"{key}={value}" for key, value in self.module_infos.items()]) + "\n"
        w_info = "\n".join([f"  {i}: {self.weights[i].shape}" for i in range(len(self.weights))])
        i_info = "\n".join([f" {i}: k_v_ids: pcd_ref={self.indices[i]['k_v_ids']['pcd_ref']}, \
                            inds={self.indices[i]['k_v_ids']['inds'].shape} | \
                                q_ids: pcd_ref={self.indices[i]['q_ids']['pcd_ref']}, \
                                    inds={self.indices[i]['q_ids']['inds'].shape}" for i in range(len(self.indices))])
        str_repr = header + m_info + w_info + "\n" + i_info + "\n"
        str_repr += "--" * 40 + "-" * len(self.name) + "\n"
        return str_repr
    
    def add_weights(self, weights: np.ndarray):
        """
        Add weights to the module
        :param weights: numpy array of weights
        """
        self.weights.append(weights)

    def add_indices(self, k_v_ids: np.ndarray, k_v_ref: str, q_ids: np.ndarray, q_ref: str):
        """
        Add indices to the module
        :param k_v_ids: numpy array of key value ids for an attention block
        :param k_v_ref: reference to what point cloud the k_v_ids belong to
        :param q_ids: numpy array of query ids for an attention block
        :param q_ref: reference to what point cloud the q_ids belong to
        """
        self.indices.append({'k_v_ids': {'pcd_ref': k_v_ref, 'inds': k_v_ids}, 'q_ids': {'pcd_ref': q_ref, 'inds': q_ids}})
    
    def is_balanced(self) -> bool:
        """
        Check if the weights and indices are balanced
        :return: True if balanced, False otherwise
        """
        n_weights = len(self.weights)
        n_indices = len(self.indices)
        return n_weights == n_indices
    
    def __len__(self) -> int:
        """
        Get the number of weights in the module
        :return: number of weights
        """
        assert self.is_balanced(), f"Mismatch between weights and indices. Weights: {len(self.weights)}, \
            Indices: {len(self.indices)}"
        return len(self.weights)
    
    def get_indices(self, layer_id) -> dict:
        """
        Get the indices of the module
        :param layer_id: layer id of the module
        :return: indices of the module
        """
        if layer_id < len(self.indices):
            return self.indices[layer_id]
        else:
            raise IndexError(f"Layer id {layer_id} out of range. Total layers: {len(self.indices)}")
        
    def get_weights(self, layer_id) -> np.ndarray:
        """
        Get the weights of the module
        :param layer_id: layer id of the module
        :return: weights of the module
        """
        if layer_id < len(self.weights):
            return self.weights[layer_id]
        else:
            raise IndexError(f"Layer id {layer_id} out of range. Total layers: {len(self.weights)}")
    

    def get_last_added_indices(self) -> dict:
        """
        Workaround function to get access to the indices of the different FE modules used in SAM
        Get the last added indices
        :return: last added indices
        """
        if self.indices:
            return self.indices[-1]
        else:
            raise IndexError("No indices in the module.")

    

class AttnWeightsCollector:
    """
    Uses ModuleCollector to store attention weights and indices of each module of the PointAttN Architecture.
    """

    def __init__(self):
        self.attn_storage: list[ModuleCollector] = []
        self.in_out: dict = dict()
        self.activated = False

    def add_in_out(self, in_out: dict):
        self.in_out = in_out

    def __str__(self) -> str:
        str_repr = "Attention Weights Collector:\n"
        for module in self.attn_storage:
            str_repr += str(module) + "\n"
        return str_repr
    
    def last(self) -> ModuleCollector:
        if len(self.attn_storage) > 0:
            return self.attn_storage[-1]
        else:
            return None
        
    def __len__(self) -> int:
        return len(self.attn_storage)
    
    def get_infos(self) -> dict:
        """
        :return: dictionary with the information of the modules {   0: {'name': name, 'module_infos': {...}, 
        'layers': {0: 'Layer_0 -- weights: (512, 256) -- kv_inds: (256,) - pcd_ref -- q_inds: (256,) - pcd_ref', 1: ...}}, 
        """
        collector_infos = {}
        for i in range(len(self.attn_storage)):
            module = self.attn_storage[i]
            module_infos = module.get_infos()
            collector_infos[i] = module_infos

        return collector_infos

    def add_module(self, name: str, module_infos: dict):
        if self.activated:
            if len(self) > 0:
                assert self.last().is_balanced(), f"Mismatch between weights and indices in the previous module. \
                    Weights: {len(self.last().weights)}, Indices: {len(self.last().indices)}"

            self.attn_storage.append(ModuleCollector(name, module_infos))

    def add_weights(self, weights: torch.Tensor):
        if self.activated:
            self.last().add_weights(weights.cpu().numpy())

    def add_indices(self, k_v_ids: torch.Tensor, k_v_ref: str, q_ids: torch.Tensor, q_ref: str):
        if self.activated:
            if type(k_v_ids) == torch.Tensor:
                k_v_ids = k_v_ids.cpu().numpy()
            if type(q_ids) == torch.Tensor:
                q_ids = q_ids.cpu().numpy()
            self.last().add_indices(k_v_ids, k_v_ref, q_ids, q_ref)

    def get_last_added_indices(self) -> dict:  # workaround to get the indices of the feature_extractor_intermediate
        """
        Workaround function to get access to the indices of the different FE modules used in SAM
        """
        module = self.attn_storage[-2]

        return module.get_last_added_indices()
    
    def workaround_double_SAM(self) -> tuple[dict, dict]:
        """
        Workaround function to get access to the indices of the different FE modules used in SAM
        """
        module1 = self.attn_storage[-1]

        module2 = self.attn_storage[-2]

        return module1.get_last_added_indices(), module2.get_last_added_indices()


    def get_module(self, module_idx) -> ModuleCollector:
        if module_idx < len(self.attn_storage):
            return self.attn_storage[module_idx]
        else:
            raise IndexError(f"Module index {module_idx} out of range. Total modules: {len(self.attn_storage)}")

    def activate(self):
        self.activated = True
    
    def deactivate(self):
        self.activated = False

    def clear(self):
        self.in_out = dict()
        self.attn_storage = []
        self.activated = False
