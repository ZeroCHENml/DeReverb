from imports import *

class Noise(object):
    """ Adds a random noise file to the original signal at a random normally distributed amplitude
    """
    def __init__(self, noise_path, noise_sr=None, db_mean=-12, db_sd=3):
        """
        Args:
            noise_path (str|Path): path to noise files
            noise_sr (int): rate at which to resample noise files. If None, uses native sample rate.
            db_mean (float|int): mean amplitude of noise w.r.t. speech signal in decibels (default: -12)
            db_sd (float|int): standard deviation of noise amplitude in decibels (default: 3)
        """
        self.noises, self.srs = zip(*[ta.load(x) for x in tqdm(noise_path.ls(), desc='Loading Noises...')])
        if noise_sr:
            self.noises = [ta.transforms.Resample(self.srs[i], noise_sr)(self.noises[i]) for i in 
                           tqdm(range(len(self.noises)), desc='Resampling Noises...')]
            
        self.db_dist = (db_mean, db_sd)
        
    def __call__(self, speech):
        db = np.random.normal(self.db_dist[0], self.db_dist[1])
        amp = lr.db_to_amplitude(db)
        n = random.choice(self.noises)
        if speech.shape[-1] < n.shape[-1]:
            return speech + amp * n[:,:speech.shape[-1]]
        else:
            return speech[:,:n.shape[-1]] + amp * n

class Reverb(object):
    """ Adds a convolutional reverb to the speech from a randomly chosen impulse response
    """
    def __init__(self, ir_path, ir_sr=None, ir_mono=True):
        """
        Args:
            ir_path (str|Path): path to a directory of impulse responses
            ir_sr (int): rate at which to resample impulse responses. if None (default) uses native sample rate.
            ir_mono (bool): if true, loads only first channel of impulse response, else loads all channels
        """
        self.irs, self.srs = zip(*[ta.load(x) for x in tqdm(ir_path.ls(), desc='Loading Impulse Responses...')])
        if ir_sr:
            self.irs = [ta.transforms.Resample(self.srs[i], ir_sr)(self.irs[i]) for i in 
                        tqdm(range(len(self.irs)), desc='Resampling Impulse Responses...')]
        if ir_mono:
            self.irs = [x[0,:] for x in self.irs]
    
    def __call__(self, speech):
        ir = random.choice(self.irs)
        # TODO: replace this with torch.conv1d
        return torch.Tensor(np.convolve(speech.squeeze().numpy(), ir.squeeze().numpy())[:speech.shape[-1]]).unsqueeze(0)

class RandomCrop(object):
    """ Crop sample to fixed length starting at random position. Pads with zeros if sample not long enough.
    """
    def __init__(self, length, no_rand=False):
        """
        Args:
            length (int): length of returned clips in samples
            no_rand (bool): if true will always start at beginning of clip. (default: False)
        """
        self.crop_len = length
        self.no_rand = no_rand
        
    def __call__(self, speech):
        if self.no_rand:
            start = 0
        else:
            start = random.randint(0, abs(speech.shape[-1]-self.crop_len))
            
        if speech.shape[-1] > self.crop_len:
            return speech[:,start:start+self.crop_len]
        else:
            retval = torch.zeros((1,self.crop_len))
            retval[:,0:speech.shape[-1]] = speech
            return retval
