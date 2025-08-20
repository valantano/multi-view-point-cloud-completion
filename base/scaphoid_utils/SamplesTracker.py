import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, "..", '..'))

print(f"Current Directory: {current_dir}")
print(f"Main Directory: {main_dir}")

# Add the main directory to sys.path such that submodules can be imported
if main_dir not in sys.path:
    sys.path.append(main_dir)

from base.scaphoid_utils.logger import get_table_str


class DoubleSampleTracker:
    """
    This tracker is used to track the best and worst samples.
    It is a subclass of SamplesTracker, but it ensures that the best samples are at the beginning.
    """
    
    def __init__(self, max_samples: int=10, smaller_is_better=True):
        self.worst_tracker = SamplesTracker(max_samples, track='worst', smaller_is_better=smaller_is_better)
        self.best_tracker = SamplesTracker(max_samples, track='best', smaller_is_better=smaller_is_better)


    def add_sample(self, sample_id: int, score: float):
        self.worst_tracker.add_sample(sample_id, score)
        self.best_tracker.add_sample(sample_id, score)

    def __len__(self):
        assert len(self.worst_tracker) == len(self.best_tracker), "Worst and Best trackers should have the same length"
        return len(self.worst_tracker)
    
    def __str__(self):
        worst_str = self.worst_tracker.__str__()
        best_str = self.best_tracker.__str__()
        return f"Worst Samples:\n{worst_str}\nBest Samples:\n{best_str}"

    


class SamplesTracker:

    def __init__(self, max_samples: int=10, track='worst', smaller_is_better=True):
        self.max_samples = max_samples
        self.sample_ids = []
        self.scores = []
        if track not in ['worst', 'best']:
            raise ValueError(f"track must be either 'worst' or 'best', got {track}")
        if track == 'worst':
            self.smaller_is_better = smaller_is_better
        else:
            self.smaller_is_better = not smaller_is_better

    def __len__(self):
        return len(self.sample_ids)

    def __ensure_ordering(self):
        """
        sample_ids should be sorted such that the best score is at position 0.
        """
        if len(self) == 0:
            return
        if self.smaller_is_better:
            sorted_pairs = sorted(zip(self.scores, self.sample_ids))
        else:
            sorted_pairs = sorted(zip(self.scores, self.sample_ids), reverse=True)
        if sorted_pairs:
            self.scores, self.sample_ids = map(list, zip(*sorted_pairs))
        else:
            self.scores, self.sample_ids = [], []        

    def __len__(self):
        return len(self.sample_ids)
    
    def __str__(self):
        sample_ids = ['X' for _ in range(self.max_samples)]
        scores = ['X' for _ in range(self.max_samples)]
        for i, (sample_id, score) in enumerate(zip(self.sample_ids, self.scores)):
            sample_ids[i] = str(sample_id)
            scores[i] = f"{score:.4f}"
        return get_table_str(sample_ids, scores)[0]
        
        
    def add_sample(self, sample_id: int, score: float):
        self.sample_ids.append(sample_id)
        self.scores.append(score)
        self.__ensure_ordering()
        if len(self.sample_ids) > self.max_samples:
            self.sample_ids = self.sample_ids[1:]
            self.scores = self.scores[1:]



if __name__ == "__main__":
    tracker = DoubleSampleTracker(max_samples=5, smaller_is_better=True)
    tracker.add_sample(1, 0.5)
    print(tracker)
    tracker.add_sample(2, 0.3)
    print(tracker)
    tracker.add_sample(3, 0.7)
    print(tracker)
    tracker.add_sample(4, 0.2)
    print(tracker)
    tracker.add_sample(5, 0.1)
    print(tracker)
    tracker.add_sample(6, 0.4)
    print(tracker)

    assert len(tracker) == 5, "Tracker should have 5 samples"
    assert tracker.worst_tracker.sample_ids == [4, 2, 6, 1, 3], "Worst tracker should have the correct order"
    assert tracker.best_tracker.sample_ids == [1, 6, 2, 4, 5], "Best tracker should have the correct order"
    assert tracker.worst_tracker.scores == [0.2, 0.3, 0.4, 0.5, 0.7], "Worst tracker should have the correct scores"
    assert tracker.best_tracker.scores == [0.5, 0.4, 0.3, 0.2, 0.1], "Best tracker should have the correct scores"

