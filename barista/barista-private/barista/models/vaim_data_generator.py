import torch
import collections
import numpy as np
import os
import itertools
import h5py
from torch.utils.data import DataLoader, IterableDataset

from ..utilities import utilities as utils
import random

utils.set_random_seed(42)

class RemageDataset(IterableDataset):
    def __init__(self, hdf5_dir, 
                parameters,
                batch_size=3000, 
                files_per_batch=20,
        ):
        """
        - hdf5_dir: Directory containing HDF5 files.
        - batch_size: Number of samples per batch (3,400).
        - files_per_batch: Number of files used in each batch (34).
        """
        super().__init__()
        self.hdf5_dir = hdf5_dir
        self.batch_size = batch_size
        self.files_per_batch = files_per_batch
        self.rows_per_file = batch_size // files_per_batch
        self.epoch_counter = 0  # Tracks row block
        self.total_batches = 0
        self.parameters = parameters
        self.hist_bins=(0, 1000, 100)
        self.d_base = "stp/det001"
        self.v_base = "stp/vertices"

        # List and sort all HDF5 files
        self.files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".hdf5")])
        self.num_files = len(self.files)
        self.dataset_size =0 
        # Total row cycles per file to complete an epoch
        self.nrows = self.get_max_number_of_rows()
        self.total_cycles_per_epoch = self.nrows // self.rows_per_file  # nrows / k rows per batch = c cycles per full dataset pass
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.dataset_size
    
    def get_max_number_of_rows(self):
        max_rows = 0
        self.dataset_size = 0 
        for file in self.files:
            with h5py.File(file, "r") as hdf:
                    num_rows = hdf["stp/particles/entries"][()]
                    self.dataset_size += num_rows
                    # Update max row count if this file has more rows
                    if num_rows > max_rows:
                        max_rows = num_rows
            if num_rows==0:
                print(f"WARNING! {file} has row size 0. Either no data or target key doesn't match.")
        if max_rows == 0:
                raise ValueError("ERROR! Data is either empty or target key doesn't match.")
        return max_rows
    
    def shuffle_files(self):
        """Shuffle the file order at the start of each full dataset pass (epoch)."""
        random.shuffle(self.files)
        self.epoch_counter = 0  # Reset row counter

    def load_vertex_event_data_in_batches(self,filename, start, end):
        """
        Generator that yields batches of features and labels:
        - Features: vertex-level [x, y, z, time]
        - Labels: one-hot vector (if detector hit) or all zeros (if no hit)

        Yields:
            X_batch: np.ndarray of shape (chunk_size, feature_size)
            y_batch: np.ndarray of shape (chunk_size, n_bins)
        """
        bin_edges = np.linspace(*self.hist_bins)
        n_bins = len(bin_edges) - 1

        with h5py.File(filename, "r") as f:
            # Vertex features

            v_base = self.parameters["feature"]["base"]
            tmp = self.parameters["feature"]["evtid_name"]
            v_evtid = f[f"{v_base}/{tmp}"][()]
            x=[]
            
            for d in self.parameters["feature"]["datasets"]:
                x.append(f[f"{v_base}/{d}"])

            # Detector info: load once into memory
            d_base = self.parameters["target"]["base"]
            tmp = self.parameters["target"]["evtid_name"]

            d_evtid = f[f"{d_base}/{tmp}"][()]

            tmp = self.parameters["target"]["datasets"]
            d_edep = f[f"{d_base}/{tmp}"][()]

            unique_d_evtids, inverse_indices = np.unique(d_evtid, return_inverse=True)
            edep_per_event = np.zeros(len(unique_d_evtids))
            np.add.at(edep_per_event, inverse_indices, d_edep)
            edep_map = dict(zip(unique_d_evtids, edep_per_event))

            evt_batch = v_evtid[start:end]
            # Read chunk
            # Collect all slices
            features = [i[start:end] for i in x]  # list of 1D arrays
            # Stack along axis=1 so that each row is a sample, each column is a feature
            X = np.stack(features, axis=1)

            # Create label array
            #Y = np.zeros((len(evt_batch), n_bins), dtype=np.float32)
            #for i,evt in enumerate(evt_batch):
            #    if evt in edep_map:
            #        total_edep = edep_map[evt]
            #        bin_index = np.digitize(total_edep, bin_edges) - 1
            #        if 0 <= bin_index < n_bins:
            #            Y[i, bin_index] = 1.0
            Y=np.zeros(len(evt_batch))
            for i,evt in enumerate(evt_batch):
                if evt in edep_map:
                    total_edep = edep_map[evt]
                    if total_edep > 790. and total_edep < 810.:
                        Y[i]=1.

            return X, Y

    def __iter__(self):
        #print(f"Starting structured HDF5 loading for row block {self.epoch_counter}...")
        self.total_batches = 0
        cycle_idx = 0
        used_rows = 0  # Track number of rows used

        while cycle_idx < self.total_cycles_per_epoch:
            for i in range(0, len(self.files), self.files_per_batch):  # Loop over file chunks
                
                batch = []
                selected_files = self.files[i:i + self.files_per_batch]

                # Select the next sequential k rows per file
                start_idx = self.epoch_counter * self.rows_per_file
                end_idx = start_idx + self.rows_per_file
                for j, file in enumerate(selected_files):
                        features, target = self.load_vertex_event_data_in_batches(file, start_idx, end_idx)
                        # Stack rows from this file
                        file_data = np.hstack([features, target])
                        batch.extend(file_data.tolist())
                        used_rows += len(file_data)
                # end loop over single file
                # Yield batch of batch-size shuffled samples
                random.shuffle(batch)
                yield torch.tensor(batch, dtype=torch.float32)

                self.total_batches += 1
            # end loop over file chunk
            cycle_idx += 1

            # Move to next row block
            self.epoch_counter += 1
            self.total_batches += cycle_idx+1
            # end loop of row chunk after all files read in

            # If all files and rows (from k*i to k*(i+1)) are read, reshuffle files for the row block
            if self.epoch_counter >= self.total_cycles_per_epoch:
                print("Finished full dataset pass. Starting new epoch! ")
                self.shuffle_files()
                break

NetRegressionDescription = collections.namedtuple(
    "NetRegressionDescription", ("query", "target_y")
)

class RemageDataGeneration(object):
    """
    """
    def __init__(
        self,
        config_file,
        path_to_files,
        batch_size,
    ):
        self._context_ratio = config_file["net_settings"]["context_ratio"]
        self._batch_size = batch_size
        self.path_to_files = path_to_files
        self.dataloader="None"
        self.config_file=config_file

        ## needs to be updated

        self.feature_size = config_file["simulation_settings"]["feature_size"]
        self.target_size = config_file["simulation_settings"]["target_size"]
        self.parameters={'feature': {'base': config_file["simulation_settings"]["feature_base"],'evtid_name': config_file["simulation_settings"]["feature_evtid_name"],'datasets': config_file["simulation_settings"]["feature_datasets"]}, 
                         'target':  {'base': config_file["simulation_settings"]["target_base"], 'evtid_name': config_file["simulation_settings"]["target_evtid_name"],'datasets': config_file["simulation_settings"]["target_datasets"]}}

    def set_loader(self):
        dataset = RemageDataset(self.path_to_files, parameters=self.parameters, batch_size=self._batch_size, files_per_batch=self.config_file["net_settings"]["files_per_batch"])
        self.dataloader = DataLoader(dataset, batch_size=None, num_workers=self.config_file["net_settings"]["number_of_walkers"], prefetch_factor=2) 

    def format_batch_for_net(self,batch, context_is_subset=True):
        """
        Formats a batch into the query format required for net training with dynamic batch splitting.
        Parameters:
        - batch (torch.Tensor): Input batch of shape (batch_size, feature_dim).
        - total_batch_size (int): Expected full batch size (default: 3000).
        - context_ratio (float): Ratio of context points (default: 1/3).
        - target_ratio (float): Ratio of target points (default: 2/3).

        Returns:
        - NetRegressionDescription(query=((batch_context_x, batch_context_y), batch_target_x), target_y=batch_target_y)
        """

        batch_size = batch.shape[0]  # Actual batch size (may be < 3000)
        
        # Dynamically compute num_context and num_target
        num_context = int(batch_size * self._context_ratio)
        num_target = batch_size - num_context  # Ensure it sums to batch_size
        
        # Shuffle the batch to ensure randomness
        batch = batch[torch.randperm(batch.shape[0])]
        
        # Split batch into input (X) and target (Y) features
        batch_x = batch[:,:self.feature_size]  # All features except last column (input features)
        batch_y = batch[:,self.feature_size:self.feature_size+self.target_size]   # Last column is the target (output values)

        if context_is_subset:
            # **Context is taken as the first num_context points from target**
            batch_target_x = batch_x  # Target is the entire batch
            batch_target_y = batch_y  # Target outputs are the entire batch

            batch_context_x = batch_target_x[:num_context]  # Context is a subset of target
            batch_context_y = batch_target_y[:num_context]  # Context outputs
        else:
            # **Context and target are independent splits**
            batch_context_x = batch_x[:num_context]  # Context inputs
            batch_context_y = batch_y[:num_context]  # Context outputs
            batch_target_x = batch_x[num_context:num_context + num_target]  # Target inputs
            batch_target_y = batch_y[num_context:num_context + num_target]  # Target outputs

        # Ensure y tensors have correct dimensions (convert from 1D to 2D if needed)
        batch_context_y = batch_context_y.view(-1, 1) if batch_context_y.ndim == 1 else batch_context_y
        batch_target_y = batch_target_y.view(-1, 1) if batch_target_y.ndim == 1 else batch_target_y
        
        if batch_context_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_context_x = batch_context_x.unsqueeze(0)
        if batch_context_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_context_y = batch_context_y.unsqueeze(0)

        if batch_target_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_target_x = batch_target_x.unsqueeze(0)
        if batch_target_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_target_y = batch_target_y.unsqueeze(0)
        # Construct the query tuple
        query = ((batch_context_x, batch_context_y), batch_target_x)
        
        # Return the properly formatted object
        return NetRegressionDescription(query=query, target_y=batch_target_y)

    def get_batch(self,batch_idx):
        """
        Retrieves a specific batch from an iterable DataLoader.

        Parameters:
        - dataloader (torch.utils.data.DataLoader): The DataLoader object.
        - batch_idx (int): The index of the batch to retrieve.

        Returns:
        - The requested batch.
        """
        batch = next(itertools.islice(self.dataloader, batch_idx, None))
        return self.format_batch_for_net(batch)

    def get_dataloader(self):
        return self.dataloader