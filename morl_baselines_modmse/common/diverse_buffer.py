"""Diverse Experience Replay Buffer. Code extracted from https://github.com/axelabels/DynMORL."""
from dataclasses import dataclass

import numpy as np


MAIN_TREE = 0


class SumTree:
    """Implementation of a SumTree with multiple trees covering the same data array.

    Adapted from: https://github.com/jaara/AI-blog/blob/master/SumTree.py.
    Each transition is stored along with its trace_id.
    """

    def __init__(self, capacity):
        """Initializes the SumTree.

        Args:
            capacity: The maximum number of transitions to store in the SumTree
        """
        self.capacity = capacity
        self.write = 0
        self.trees = {}
        self.main_tree = MAIN_TREE
        self.data = np.repeat(((None, None, None),), capacity, axis=0)
        self.updates = {}

    def copy_tree(self, trg_i, src_i=MAIN_TREE):
        """Copies src_i's priorities into a new tree trg_i.

        Args:
            trg_i:Target tree identifier
            src_i: Source tree identifier (default: {MAIN_TREE})
        """
        if trg_i not in self.trees:
            self.trees[trg_i] = np.copy(self.trees[src_i])
            self.updates[trg_i] = 0

    def create(self, i):
        """Create tree i, either by copying the main tree if it exists, or by creating a new tree from scratch.

        Args:
            i: The new tree's identifier
        """
        if i is None:
            i = self.main_tree
        if i not in self.trees and self.main_tree in self.trees:
            self.copy_tree(i, self.main_tree)
        elif i not in self.trees:
            self.trees[i] = np.zeros(2 * self.capacity - 1)
            self.updates[i] = 0

    def _propagate(self, idx: int, change: float, tree_id=None):
        """Propagate priority changes to root.

        Args:
            idx: Node to propagate from
            change: Priority difference to propagate
            tree_id: Which tree the change applies to)
        """
        tree_id = tree_id if tree_id is not None else self.main_tree

        parent = (idx - 1) // 2

        self.trees[tree_id][parent] += change

        if parent != 0:
            self._propagate(parent, change, tree_id)

    def _retrieve(self, idx: int, s: float, tree_id=None):
        """Retrieve the node covering offset s starting from node idx.

        Args:
            idx: Note to start from
            s: offset
            tree_id: Which tree the priorities relate to

        Returns:
            node covering the offset
        """
        tree_id = tree_id if tree_id is not None else self.main_tree

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.trees[tree_id]):
            return idx

        if s <= self.trees[tree_id][left]:
            return self._retrieve(left, s, tree_id)
        else:
            return self._retrieve(right, s - self.trees[tree_id][left], tree_id)

    def total(self, tree_id=None):
        """Returns the tree's total priority.

        Args:
            tree_id: Tree identifier

        Returns:
            Total priority
        """
        tree_id = tree_id if tree_id is not None else self.main_tree

        return self.trees[tree_id][0]

    def average(self, tree_id=None):
        """Return the tree's average priority, assumes the tree is full.

        Args:
            tree_id: Tree identifier

        Returns:
            Average priority
        """
        return self.total(tree_id) / self.capacity

    def add(self, priorities: dict, data: tuple, write=None):
        """Adds a data sample to the SumTree at position write.

        Args:
            priorities: Dictionary of priorities, one key per tree
            data: Transition to be added
            write: Position to write to

        Returns:
            Tuple containing the replaced data as well as the node's index in the tree.
        """
        write = write if write is not None else self.write
        idx = write + self.capacity - 1

        # Save replaced data to eventually save in secondary memory
        replaced_data = np.copy(self.data[write])
        replaced_priorities = {tree: np.copy(self.trees[tree][idx]) for tree in self.trees}
        replaced = (replaced_data, replaced_priorities)

        # Set new priorities
        for i, p in priorities.items():
            self.update(idx, p, i)

        # Set new data
        self.data[write] = data
        return replaced, idx

    def update(self, idx: int, p, tree_id=None):
        """For a given index, update priorities for the given trees.

        Args:
            idx: Node's position in the tree
            p: Dictionary of priorities or priority for the given tree_id
            tree_id: Tree to be updated

        Keyword Arguments:
            tree_id {object} -- Tree to be updated (default: {None})
        """
        if type(p) == dict:
            for k in p:
                self.update(idx, p[k], k)
            return
        tree_id = tree_id if tree_id is not None else self.main_tree

        change = p - self.trees[tree_id][idx]

        self.trees[tree_id][idx] = p
        self._propagate(idx, change, tree_id)

    def get(self, s: float, tree_id=None):
        """Get the node covering the given offset.

        Args:
            s: Offset to retrieve
            tree_id: Tree to retrieve from

        Returns:
            Containing the index, the priority and the transition
        """
        tree_id = tree_id if tree_id is not None else self.main_tree

        idx = self._retrieve(0, s, tree_id)

        return self.get_by_id(idx, tree_id)

    def get_by_id(self, idx: int, tree_id=None):
        """Get the node at the given index.

        Args:
            idx: Index to retrieve
            tree_id: Tree to retrieve from

        Returns:
            A tuple containing the index, the priority and the transition
        """
        tree_id = tree_id if tree_id is not None else self.main_tree
        dataIdx = idx - self.capacity + 1

        return idx, self.trees[tree_id][idx], self.data[dataIdx]


class DiverseMemory:
    """Prioritized Replay Buffer with integrated secondary Diverse Replay Buffer. Code extracted from https://github.com/axelabels/DynMORL."""

    def __init__(
        self,
        main_capacity: int,
        sec_capacity: int = 0,
        trace_diversity: bool = True,
        crowding_diversity: bool = True,
        value_function=lambda trace, trace_id, memory_indices: np.random.random(1),
        e: float = 0.01,
        a: float = 2,
    ):
        """Initializes the DiverseMemory.

        Args:
            main_capacity: Normal prioritized replay capacity
            sec_capacity: Size of the secondary diverse replay buffer, if 0, the buffer functions as a normal prioritized Replay Buffer (default: {0})
            trace_diversity: Whether diversity should be enforced at trace-level (True) or at transition-level (False)
            crowding_diversity: Whether a crowding distance is applied to compute diversity
            value_function: When applied to a trace, this function should return the trace's value to be used in the crowding distance computation
            e: epsilon to be added to errors (default: {0.01})
            a: Power to which the error will be raised, if a==0, functionality is reduced to a replay buffer without prioritization (default: {2})
        """
        self.len = 0
        self.trace_diversity = trace_diversity
        self.value_function = value_function
        self.crowding_diversity = crowding_diversity
        self.e = e
        self.a = a
        self.capacity = main_capacity + sec_capacity
        self.tree = SumTree(self.capacity)
        self.main_capacity = main_capacity
        self.sec_capacity = sec_capacity
        self.secondary_traces = []

    def _getPriority(self, error):
        """Compute priority from error.

        Args:
            error {float} -- error

        Returns:
            float -- priority
        """
        return (error + self.e) ** self.a

    def _getError(self, priority):
        """Given a priority, computes the corresponding error.

        Args:
            priority {float} -- priority

        Returns:
            float -- error
        """
        return priority ** (1 / self.a) - self.e

    def dupe(self, trg_i, src_i):
        """Copies the tree src_i into a new tree trg_i.

        Args:
            trg_i: target tree identifier
            src_i: source tree identifier
        """
        self.tree.copy_tree(trg_i, src_i)

    def main_mem_is_full(self):
        """Because of the circular way in which we fill the memory, checking whether the current write position is free is sufficient to know if the memory is full."""
        return self.tree.data[self.tree.write][1] is not None

    def extract_trace(self, start: int):
        """Determines the end of the trace starting at position start.

        Args:
            start: Trace's starting position

        Returns:
            The trace's end position
        """
        trace_id = self.tree.data[start][0]

        end = (start + 1) % self.main_capacity

        if not self.trace_diversity:
            return end
        if trace_id is not None:
            while self.tree.data[end][0] == trace_id:
                end = (end + 1) % self.main_capacity
                if end == start:
                    break

        return end

    def add(self, error, sample, trace_id=None, pred_idx=None, tree_id=None):
        """Add the sample to the replay buffer, with a priority proportional to its error. If trace_id is provided, the sample and the other samples with the same id will be treated as a trace when determining diversity.

        Args:
            error: Error
            sample: The transition to be stored
            trace_id: The trace's identifier (default: {None})
            tree_id: The tree for which the error is relevant (default: {None})

        Returns:
            The index of the node in which the sample was stored
        """
        self.len = min(self.len + 1, self.capacity)
        self.tree.create(tree_id)

        sample = (
            trace_id,
            sample,
            None if pred_idx is None else (pred_idx - self.capacity + 1),
        )

        # Free up space in main memory if necessary
        if self.main_mem_is_full():
            end = self.extract_trace(self.tree.write)
            self.move_to_sec(self.tree.write, end)

        # Save sample into main memory
        if type(error) is not dict:
            error = {tree_id: error}
        idx = self.add_sample(sample, error, self.tree.write)
        self.tree.write = (self.tree.write + 1) % self.main_capacity

        return idx

    def remove_trace(self, trace):
        """Removes the trace from the main memory.

        Args:
            trace: List of indices for the trace
        """
        _, trace_idx = trace
        for i in trace_idx:
            self.tree.data[i] = (None, None, None)

            idx = i + self.tree.capacity - 1
            for tree in self.tree.trees:
                self.tree.update(idx, 0, tree)

    def get_trace_value(self, trace_tuple):
        """Applies the value_function to the trace's data to compute its value.

        Args:
            trace_tuple: Tuple containing the trace and the trace's indices

        Returns:
            The trace's value
        """
        trace, write_indices = trace_tuple
        if not self.trace_diversity:
            assert len(trace) == 1
        trace_id = trace[0][0]
        trace_data = [t[1] for t in trace]

        return self.value_function(trace_data, trace_id, write_indices)

    def sec_distances(self, traces):
        """Give a set of traces, this method computes each trace's crowding distance.

        Args:
            traces: List of trace tuples

        Returns:
            List of distances
        """
        values = [self.get_trace_value(tr) for tr in traces]
        if self.crowding_diversity:
            distances = crowd_dist(values)
        else:
            distances = values
        return [(i, d) for i, d in enumerate(distances)], values

    def get_sec_write(self, secondary_traces, trace, reserved_idx=None):
        """Given a trace, find free spots in the secondary memory to store it by recursively removing past traces with a low crowding distance."""
        if reserved_idx is None:
            reserved_idx = []

        if len(trace) > self.sec_capacity:
            return None

        if len(reserved_idx) >= len(trace):
            return reserved_idx[: len(trace)]

        # Find free spots in the secondary memory
        # TODO: keep track of free spots so recomputation isn't necessary
        free_spots = [
            i + self.main_capacity for i in range(self.sec_capacity) if (self.tree.data[self.main_capacity + i][1]) is None
        ]

        if len(free_spots) > len(reserved_idx):
            return self.get_sec_write(secondary_traces, trace, free_spots[: len(trace)])

        # Get crowding distance of traces stored in the secondary buffer
        idx_dist, _ = self.sec_distances(secondary_traces)

        # Highest density = lowest distance
        i, _ = min(idx_dist, key=lambda d: d[1])

        _, trace_idx = secondary_traces[i]
        reserved_idx += trace_idx

        self.remove_trace(secondary_traces[i])

        del secondary_traces[i]
        return self.get_sec_write(secondary_traces, trace, reserved_idx)

    def move_to_sec(self, start: int, end: int):
        """Move the trace spanning from start to end to the secondary replay buffer.

        Args:
            start: Start position of the trace
            end: End position of the trace
        """
        # Recover trace that needs to be moved
        if end <= start:
            indices = np.r_[start : self.main_capacity, 0:end]
        else:
            indices = np.r_[start:end]
        if not self.trace_diversity:
            assert len(indices) == 1
        trace = np.copy(self.tree.data[indices])
        priorities = {tree_id: self.tree.trees[tree_id][indices + self.tree.capacity - 1] for tree_id in self.tree.trees}

        # Get destination indices in secondary memory
        write_indices = self.get_sec_write(self.secondary_traces, trace)

        # Move trace to secondary memory if enough space was freed
        if write_indices is not None and len(write_indices) >= len(trace):
            for i, (w, t) in enumerate(zip(write_indices, trace)):
                self.tree.data[w] = t

                idx = w + self.tree.capacity - 1
                for tree_id in priorities:
                    p = priorities[tree_id][i]
                    self.tree.update(idx, p, tree_id)

                if i > 0:
                    self.tree.data[w][2] = write_indices[i - 1]
            if not self.trace_diversity:
                assert len(trace) == 1
            self.secondary_traces.append((trace, write_indices))
        # elif self.sec_capacity>0:
        #     print("No space found for trace", trace[0][0],", discarding...",file=sys.stderr)

        # Remove trace from main memory
        self.remove_trace((None, indices))

    def add_sample(self, transition, error, write=None):
        """Stores the transition into the priority tree.

        Args:
            transition: Tuple containing the trace id, the sample and the previous sample's index
            error: Dictionary containing the error for each tree
            write: Index to write the transition to
        """
        p = {k: self._getPriority(error[k]) for k in error}
        _, idx = self.tree.add(p, transition, write=write)
        return idx

    def add_tree(self, tree_id):
        """Adds a secondary priority tree.

        Args:
            tree_id: The secondary tree's id
        """
        self.tree.create(tree_id)

    def get_data(self, include_indices: bool = False):
        """Get all the data stored in the replay buffer.

        Args:
            include_indices: Whether to include each sample's position in the replay buffer (default: {False})

        Returns:
            The data
        """
        all_data = list(np.arange(self.capacity) + self.capacity - 1), list(self.tree.data)
        indices = []
        data = []
        for i, d in zip(all_data[0], all_data[1]):
            if (d[1]) is not None:
                indices.append(i)
                data.append(d[1])
        if include_indices:
            return indices, data
        else:
            return data

    def sample(self, n: int, tree_id=None):
        """Sample n transitions from the replay buffer, following the priorities of the tree identified by tree_id.

        Args:
            n: Number of transitions to sample
            tree_id: identifier of the tree whose priorities should be followed (default: {None})

        Returns:
            pair of (indices, transitions)
        """
        if n < 1:
            return None, None, None
        batch = np.zeros((n,), dtype=np.ndarray)
        ids = np.zeros(n, dtype=int)
        priorities = np.zeros(n, dtype=float)
        segment = self.tree.total(tree_id) / n
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            (idx, p, data) = self.tree.get(s, tree_id)
            while (data[1]) is None or (idx - self.capacity + 1 >= self.capacity):
                s = np.random.uniform(0, self.tree.total(tree_id))
                (idx, p, data) = self.tree.get(s, tree_id)
            ids[i] = idx
            batch[i] = data[1]
            priorities[i] = p
        return ids, batch, priorities

    def update(self, idx: int, error: float, tree_id=None):
        """Given a node's idx, this method updates the corresponding priority in the tree identified by tree_id.

        Args:
            idx: Node's index
            error: New error
            tree_id: Identifies the tree to update (default: {None})
        """
        if tree_id is None:
            tree_id = self.tree.main_tree
        if type(error) is not dict:
            error = {tree_id: error}
        p = {k: self._getPriority(error[k]) for k in error}
        self.tree.update(idx, p, tree_id)

    def get(self, indices: list):
        """Given a list of node indices, this method returns the data stored at those indices.

        Args:
            indices: List of indices

        Returns:
            array of transitions
        """
        indices = np.array(indices, dtype=int) - self.capacity + 1
        return self.tree.data[indices][:, 1]

    def get_error(self, idx, tree_id=None):
        """Given a node's idx, this method returns the corresponding error in the tree identified by tree_id.

        Args:
            idx: Node's index
            tree_id: Identifies the tree to update (default: {None})

        Returns:
            Error
        """
        tree_id = self.tree.main_tree if tree_id is None else tree_id
        priority = self.tree.trees[tree_id][idx]
        return self._getError(priority)


def crowd_dist(evals: list):
    """Given a list of vectors, this method computes the crowding distance of each vector, i.e. the sum of distances between neighbors for each dimension.

    Args:
        evals: list of vectors

    Returns:
        list of crowding distances
    """

    @dataclass
    class Point:
        data: np.ndarray
        distance: float
        i: int

    points = np.array([Point() for _ in evals])
    dimensions = len(evals[0])
    for i, d in enumerate(evals):
        points[i].data = d
        points[i].i = i
        points[i].distance = 0.0

    # Compute the distance between neighbors for each dimension and add it to
    # each point's global distance
    for d in range(dimensions):
        points = sorted(points, key=lambda p: p.data[d])
        spread = points[-1].data[d] - points[0].data[d]
        for i, p in enumerate(points):
            if i == 0 or i == len(points) - 1:
                p.distance += float("inf")
            else:
                p.distance += (points[i + 1].data[d] - points[i - 1].data[d]) / spread

    # Sort points back to their original order
    points = sorted(points, key=lambda p: p.i)
    distances = np.array([p.distance for p in points])

    return distances
