from base.scaphoid_utils.logger import print_log, get_table_str


class AverageMeter:

    """
    Keeps track of most recent, average, sum, count and std of an item.
    Stores the current value, the sum and the count of the values of specified items
    """
    def __init__(self, items='', init_later=False):
        """
        :param items: list of items to keep track of e.g. ['loss', 'accuracy']
        """
        self.initialized = False

        self.items = None
        self.n_items = 0
        self._val = None    # stores current value of each item; list of size (n_items)
        self._sum = None    # stores sum of each item; list of size (n_items)
        self._count = None  # stores count of each item; list of size (n_items)
        self._sq_sum = None  # stores sum of squares of each item; list of size (n_items)


        if not init_later:
            self.initialize(items)
        

    def initialize(self, items):
        """
        Helper func to initialize the items even after the object has been created
        :param items: list of items to keep track of e.g. ['loss', 'accuracy']
        """
        if type(items) == str:
            items = [items]

        elif type(items) == dict:
            items = list(items.keys())

        self.items = {}
        for i, val in enumerate(items):
            self.items[val] = i     # dict to keep track of index of each item

        self.n_items = len(items)
        self.reset()
        self.initialized = True

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items
        self._sq_sum = [0] * self.n_items

    def get_epoch_log_dict(self, val=True):
        log = {}
        for item_name, idx in self.items.items():
            if val:
                metric_name = 'Metric/' + item_name
            else:
                metric_name = item_name.replace('/', '/Epoch/')
            log[metric_name] = self.avg(idx)
        
        for log_name, value in log.items():
            if 'CDL' in log_name:
                log[log_name] = value * 1000
        return log
    
    def __str__(self):
        header = ['Vars'] + [name.replace('Loss/', '') for name in self.get_names()]
        values = ['_val'] + ["%.4f" % v for v in self._val]
        sums = ['_sum'] + ["%.4f" % s for s in self._sum]
        counts = ['_count'] + ["%.4f" % c for c in self._count]
        table, separat_rows = get_table_str(header, values, sums, counts)
        return table

    def get_names(self):
        """
        Returns the names of the items
        :return: list of item names
        """
        return list(self.items.keys())

    def update_via_dict(self, metric_dict):
        assert isinstance(metric_dict, dict), f"metric_dict should be a dict, but got {type(metric_dict)}"
        if not self.initialized:
                self.initialize(metric_dict)
        # print(f"Updating {metric_dict}")
        for metric_name, metric_values in metric_dict.items():
            # print(f"Updating {metric_name} with {metric_values}")
            if metric_values.dim() == 0:
                # print(f"dim=0: {metric_values}")
                self.update(metric_values.item(), metric_name)
            elif metric_values.dim() == 1:
                # print(f"dim=1: {metric_values}")
                for _, metric_value in enumerate(metric_values):
                    self.update(metric_value, metric_name)
            else:
                raise ValueError(f"metric_values should be a scalar or a 1D tensor, but got {metric_values.dim()} \
                                 dimensions")

    def update(self, values, item_name=None):
        if isinstance(values, dict):
            # dict like {Loss/Sparse: [0.1, 0.2], Loss/Total: [0.3, 0.4]} or {Loss/Sparse: 0.1, Loss/Total: [0.3, 0.4]}
            raise NotImplementedError("Updating via dict is not implemented yet")

        if isinstance(values, list):
            assert len(values) == self.n_items, f"Length of values {len(values)} does not match number of \
                items {self.n_items}"
            for idx, v in enumerate(values):
                self.update_single(v, idx)
        else:
            if item_name is not None:
                idx = self.items.get(item_name, None)
                if idx is None:
                    raise ValueError(f"Item name {item_name} not found in items {self.items}")
            else:
                idx = 0
            self.update_single(values, idx)

    def update_single(self, value, idx):
        """
        Updates the value of the item at the given index
        :param value: value to update
        :param idx: index of item to update
        """
        if not self.initialized:
            raise ValueError("AverageMeter not initialized. Call initialize() first.")
        if idx < 0 or idx >= self.n_items:
            raise IndexError(f"Index {idx} out of range for items {self.items}")
        
        self._val[idx] = value
        self._sum[idx] += value
        self._count[idx] += 1
        self._sq_sum[idx] += value ** 2

    def std(self, idx=None):
        """
        Gets the sample standard deviation of the item at the given index or name
        :param idx: index or name of item to get std of 
        """
        if isinstance(idx, str):
            idx = self.items.get(idx, None)
        if idx is None:
            return self.__compute_sample_std(0) if self.n_items == 1 else [
                self.__compute_sample_std(i) for i in range(self.n_items)
            ]
        else:
            return self.__compute_sample_std(idx)

    def __compute_sample_std(self, idx: int) -> float:
        """
        Private helper function to compute sample standard deviation
        :param idx: index of item to compute std for
        """
        if self._count[idx] < 2:
            return 0.0
        c, sq_sum, sum = self._count[idx], self._sq_sum[idx], self._sum[idx]
        return ((sq_sum - (sum ** 2) / c) / (c-1)) ** 0.5

    def sum(self, idx=None):
        """
        Gets the sum of the item at the given index or name
        :param idx: index or name of item to get sum of 
        """
        if isinstance(idx, str):
            idx = self.items.get(idx, None)
        if idx is None:
            return self._sum[0] if self.n_items == 1 else [self._sum[i] for i in range(self.n_items)]
        else:
            return self._sum[idx]

    def val(self, idx=None):
        """
        Gets the value of the item at the given index or name
        :param idx: index or name of item to get value of 
        """
        if isinstance(idx, str):
            idx = self.items.get(idx, None)
        if idx is None:
            return self._val[0] if self.n_items == 1 else [self._val[i] for i in range(self.n_items)]
        else:
            return self._val[idx]

    def count(self, idx=None):
        """
        :param idx: index or name of item to get value of 
        """
        if isinstance(idx, str):
            idx = self.items.get(idx, None)
        if idx is None:
            return self._count[0] if self.n_items == 1 else [self._count[i] for i in range(self.n_items)]
        else:
            return self._count[idx]

    def avg(self, idx=None):
        """
        :param idx: index or name of item to get value of 
        """
        if isinstance(idx, str):
            idx = self.items.get(idx, None)

        if idx is None:
            return self._sum[0] / self._count[0] if self.n_items == 1 else [
                self._sum[i] / self._count[i] for i in range(self.n_items)
            ]
        else:
            return self._sum[idx] / self._count[idx]
        