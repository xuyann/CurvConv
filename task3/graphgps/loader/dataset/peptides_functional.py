import hashlib
import os.path as osp
import pickle
import shutil

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, download_url
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import numpy as np

class PeptidesFunctionalDataset(InMemoryDataset):
    def __init__(self, root='datasets', smiles2graph=smiles2graph,
                 transform=None, pre_transform=None):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'peptides-functional')

        self.url = 'https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1'
        self.version = '701eb743e899f4d793f0e13c8fa5a1b4'  # MD5 hash of the intended dataset file
        self.url_stratified_split = 'https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1'
        self.md5sum_stratified_split = '5a0114bdadc80b94fc7ae974f13ef061'

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            # if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
            #     shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'peptide_multi_class_dataset.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            path_split1 = download_url(self.url_stratified_split, self.root)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir,
                                       'peptide_multi_class_dataset.csv.gz'))
        smiles_list = data_df['smiles']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            # G=nx.Graph()
            # for node in range(graph['num_nodes']):
            #     G.add_node(node)
            # edge_tuples = [(graph['edge_index'][0][j], graph['edge_index'][1][j]) for j in range(graph['edge_index'].shape[1])]
            # G.add_edges_from(edge_tuples)
            # orc = OllivierRicci(G, alpha=0.5, verbose="INFO") 
            # orc.compute_ricci_curvature()
            # edge_feat=graph['edge_feat'].tolist()
            # edge_curvature = []
            # node_curvature=[]
            # for j,edge in enumerate(edge_tuples):
            #     n1,n2=edge
            #     if edge in G.edges:
            #         curvature=orc.G[n1][n2]["ricciCurvature"]
            #     else:
            #         curvature=orc.G[n2][n1]["ricciCurvature"]
            #     edge_feat[j].append(curvature)
            #     edge_curvature.append(curvature)
            # node_feat=graph['node_feat'].tolist()
            # for node in range(graph['num_nodes']):
            #     node_feat[node].append(orc.G.nodes[node].get('ricciCurvature', None))
            #     if node_feat[node][-1] is None:
            #         node_feat[node][-1]=0
            #     node_curvature.append(node_feat[node][-1])
            edge_fn=f'edge_curvature/{i}_edge_curvature.txt'  
            edge_filename = osp.join(self.folder,edge_fn)
            # if osp.exists(edge_filename):
            #     print('exist')
            # else:
            #     with open(edge_filename,'w') as file:
            #         for item in edge_curvature:
            #             file.write(str(item)+'\n')
            
            node_fn=f'node_curvature/{i}_node_curvature.txt'    
            node_filename = osp.join(self.folder,node_fn)
            # if osp.exists(node_filename):
            #     print('exist')
            # else:
            #     with open(node_filename,'w') as file:
            #         for item in node_curvature:
            #             file.write(str(item)+'\n')

            # node_feat_array=np.array(node_feat)
            # edge_feat_array=np.array(edge_feat) 
            edge_cur=[]
            with open(edge_filename, 'r') as file:
                for line in file:
                    
                    edge_cur.append(float(line.strip()))
            # node_cur=[]
            # with open(node_filename, 'r') as file:
            #     for line in file:
     
            #         float_num = float(line.strip())
            #         scaled_int = int(float_num * (2 ** 32))
            #         node_cur.append(scaled_int)
            # node_feat=graph['node_feat'].tolist()
            # for node in range(graph['num_nodes']):
            #     node_feat[node].append(node_cur[node])
                
            # node_feat_array=np.array(node_feat)
            
            data.curva=torch.from_numpy(np.array(edge_cur)).to(torch.float32)
            
            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(
                torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(
                torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([eval(data_df['labels'].iloc[i])])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        """ Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(self.root,
                              "splits_random_stratified_peptide.pickle")
        with open(split_file, 'rb') as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        return split_dict


if __name__ == '__main__':
    dataset = PeptidesFunctionalDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
