

from pathlib import Path
import numpy as np
from collections import defaultdict
import enum

from ..data.lef_uspto import LEF_USPTO, DataVariants
from ..data.transforms import TransformStrToBrokenDownParts, TransformToRdKitIntermediates
from ..models.llm import ModelVariants

class TestManager:
    TEST_INDICES_PATH="data/test_indices.npy"
    ICL_INDICES_PATH="data/test_ICL_indices.npy"
    
    def __init__(self, proj_root='.', n_samples=20, icl_samples=5, gpt4=True, scaffold=True, self_consistency=False):
        
        # load data
        self.train_data = LEF_USPTO(DataVariants.TRAIN, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
        self.test_data = LEF_USPTO(DataVariants.TEST, transforms=[TransformStrToBrokenDownParts(), TransformToRdKitIntermediates()])
        
        # test
        self.n = n_samples
        self.test_indices = np.load(Path(proj_root) / self.TEST_INDICES_PATH, allow_pickle=True)[:self.n]
        
        # training
        self.k = icl_samples
        self.scaffold = scaffold
        if self.scaffold:
            self.icl_indices = np.array(np.load(Path(proj_root) / self.ICL_INDICES_PATH, allow_pickle=True))[self.test_indices,:self.k]
        else:
            self.icl_indices = np.random.randint(0, len(self.test_data), (self.n, self.k))
        
        print("Loading Samples...", end = '\t', flush = True)
        tmp = [self.train_data[self.icl_indices[i]] for i in range(self.n)]
        self.sample = list(zip(self.test_indices, self.test_data[self.test_indices], tmp))
        print("Done")
        
        # model
        self.model = ModelVariants.GPT4 if gpt4 else ModelVariants.GPT3_5
        
        # self-consistency
        self.sc = self_consistency
        self.temp=.5 if self.sc else 0
        self.num_preds=20 if self.sc else 1
        

class ResultVariants(enum.StrEnum):
    """Variants for computing results"""
    VANILLA='vanilla'
    SC='self-consistency'


class Results:
    """Tracks where the results of tests are saved"""
    def __init__(self, proj_root='.', header="baseline", model=ModelVariants.GPT4, scaffold=True, filters = [], variant=ResultVariants.VANILLA, save_dir=None ):
        
        # result dir
        self.result_dir = Path(proj_root) / "results"
        self.result_dir.mkdir(exist_ok=True)
        
        # filters
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        else:
            self.save_dir = self.result_dir / header
            for f in filters:
                self.save_dir = self.save_dir / f
            self.scaffold_label = "scaffold" if scaffold else "random"
            self.save_dir = self.save_dir / model.value / self.scaffold_label
        self.save_dir.mkdir(parents=True,exist_ok=True)

        # info
        self.test_indices = []
        self.icl_indices = []
        self.prompts = []
        self.gt = []
        self.predicted = []
        self.info = defaultdict(dict)
        self.variant=variant
    
    def store(self, test_indice, prompt, gt, predicted, icl_indices=None, **kwargs):
        self.test_indices.append(test_indice)
        self.icl_indices.append(icl_indices)
        self.prompts.append(prompt)
        self.gt.append(gt)
        self.predicted.append(predicted)
        
        for key, value in kwargs:
            self.info[key][test_indice] = value
    
    def save(self):
        np.savez(str(self.save_dir / "results.npz"), 
                 variant=str(self.variant),
                 test_indices=self.test_indices,
                 prompts=self.prompts,
                 gt=self.gt,
                 predicted=self.predicted,
                 icl_indices=self.icl_indices,
                 info=self.info)
    
    def load(self):
        x = np.load(str(self.save_dir /  "results.npz"), allow_pickle=True)
        self.test_indices = x['test_indices']
        self.gt = x['gt']
        self.predicted = x['predicted']
        self.prompts = x['prompts']
        self.icl_indices = x['icl_indices']
        self.info = x['info']
        self.variant=x['variant']
        
        
        
        
        
        