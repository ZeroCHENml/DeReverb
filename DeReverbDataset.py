from imports import *

class DeReverbDataset(Dataset):
    """
    Pytorch dataset for audio sequence to sequence tasks. 
    """
    def __init__(self, root_path, 
                 speech_path='clean-speech', 
                 ir_path='impulse-responses', 
                 noise_path='noise', 
                 rev_tfms=None, 
                 clean_tfms=None, 
                 crop_load=None,
                 use_cuda=False):
        self.rp = Path(root_path)
        self.speech_files = (self.rp/speech_path).ls()
        self.ir_files = (self.rp/ir_path).ls()
        self.noise_files = (self.rp/noise_path).ls()
        self.rev_tfms = rev_tfms
        self.clean_tfms = clean_tfms
        self.use_cuda = use_cuda
        
        if crop_load:
            self.loader = crop_load
        else:
            self.loader = ta.load
        
    def __len__(self):
        return len(self.speech_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sfn = self.speech_files[idx]
        speech, ssr = self.loader(sfn)
        
        # Transforms change the clean speech, adding noise, reverb and 
        # other effects.
        if self.clean_tfms:
            speech = self.clean_tfms(speech)
            
        if self.use_cuda:
            speech = speech.cuda(non_blocking=True)
            
        if self.rev_tfms:
            reverbed = self.rev_tfms(speech)
            
            
        
        reverbed = torch.Tensor(reverbed)
        speech = torch.Tensor(speech)
        sample = (reverbed, speech)
        return sample
