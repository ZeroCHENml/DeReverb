from imports import *

class Noise(object):
    """ Adds a random noise file to the original signal at a random normally distributed amplitude
    """
    def __init__(self, noise_path, noise_sr=None, db_mean=-12, db_sd=3, use_cuda=False):
        """
        Args:
            noise_path (str|Path): path to noise files, or list of files
            noise_sr (int): rate at which to resample noise files. If None, uses native sample rate.
            db_mean (float|int): mean amplitude of noise w.r.t. speech signal in decibels (default: -12)
            db_sd (float|int): standard deviation of noise amplitude in decibels (default: 3)
        """
        if type(noise_path) is not list:
            noise_path = list(noise_path.glob('*.wav'))
            
        self.noises, self.srs = zip(*[ta.load(x) for x in tqdm(noise_path, desc='Loading Noises...')])
        
        if use_cuda:
            self.noises = [x.cuda() for x in self.noises]
            
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
        Crops the beginning of the impulse response to non-silent parts to maintain time-alignment of input and target
    """
    def __init__(self, ir_path, ir_sr=None, ir_mono=True, use_cuda=False):
        """
        Args:
            ir_path (str|Path): path to a directory of impulse responses or list of files
            ir_sr (int): rate at which to resample impulse responses. if None (default) uses native sample rate.
            ir_mono (bool): if true, loads only first channel of impulse response, else loads all channels
        """
        if type(ir_path) is not list:
            ir_path = list(ir_path.glob('*.wav'))
            
        self.irs, self.srs = zip(*[ta.load(x) for x in tqdm(ir_path, desc='Loading Impulse Responses...')])
        
        if ir_sr:
            self.irs = [ta.transforms.Resample(self.srs[i], ir_sr)(self.irs[i]) for i in 
                        tqdm(range(len(self.irs)), desc='Resampling Impulse Responses...')]
        if ir_mono:
            self.irs = [x[0,:] for x in self.irs]
            
        if use_cuda:
            self.irs = [x.cuda() for x in self.irs]
            
        # Crop beginning silence of IRS
        crop_idxs = [x.argmax(-1) for x in self.irs]
        self.irs = [x[crop_idxs[i]:].unsqueeze(0).unsqueeze(1) for i,x in enumerate(self.irs)]
    
    def __call__(self, speech):
        ir = random.choice(self.irs)
        if ir.shape[-1] < speech.shape[-1]:
            padding = ir.shape[-1]
            inputs, filters = speech.unsqueeze(0), ir.flip(-1)
        else:
            padding = speech.shape[-1]
            inputs, filters = ir, speech.unsqueeze(0).flip(-1)
            
        return F.conv1d(inputs, filters, padding=padding)[0,:,:speech.shape[-1]].cpu()
        
        # TODO: replace this with torch.conv1d
        #return torch.Tensor(np.convolve(speech.squeeze().numpy(), ir.squeeze().numpy(), 'same')).unsqueeze(0)

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

class LoadCrop(object):
    """ Similar to RandomCrop, but acts on loading the signal to save disk bandwidth.
        About 20x faster in testing when using a frame length of 2048.
    """
    def __init__(self, length, no_rand=False):
        """
        Args:
            length (int): length of returned clips in samples
            no_rand (bool): if true will always start at beginning of clip. (default: False)
        """
        self.crop_len = length
        self.no_rand = no_rand
        
    def __call__(self, fn):
        si,_ = ta.info(str(fn))
        
        if self.no_rand:
            start = 0
        else:
            start = random.randint(0, abs(si.length-self.crop_len-1))
        
            
        try:
            if si.length > (self.crop_len + start):
                return ta.load(fn, num_frames=self.crop_len, offset=start)
        except Exception as E:
            pass
            
        # if problem happened above
        speech,ssr = ta.load(fn)
        if speech.shape[-1] < self.crop_len:
            retval = torch.zeros((1,self.crop_len))
            retval[:,0:speech.shape[-1]] = speech
            return (retval, ssr)
        else: # some other problem occurred reading a chunk of the file
            return (speech[:,:self.crop_len], ssr)
