from imports import *

# Shorthand amplitude->db conversion
db = lambda x: lr.amplitude_to_db(x)

def plot_spec(sig,sr=22050, channel=None):
    """
    Plots spectrogram of time domain signal (numpy or tensor) with default parameters.
    channel: selects one channel if input has multiple channels
    """
    if type(sig) is torch.Tensor:
        sig = sig.squeeze().numpy()
    if channel != None:
        sig = sig[:,channel].squeeze()
    spec = db(abs(lr.stft(sig))) 
    return specshow(spec, x_axis='time', y_axis='mel', sr=sr)
    
    
def cola_window(win_size, hop_len, win_type=np.hamming):
    """
    Creates a window for which the COLA constraint holds
    """
    w = win_type(win_size)
    w = np.sqrt(w)
    K = np.sqrt(hop_len / sum(pow(w,2)))
    w = w * K
    return w

def fourier(x):
    """
    Converts a time domain tensor to frequency domain and takes angle/phase form
    """
    return taf.magphase(x.rfft(1))

def polar_to_cart(x):
    """
    Converts a tensor with last dimension in [angle, phase] to [a, jb] form
    """
    amp = x[:,0].unsqueeze(-1)
    phase = x[:,1].unsqueeze(-1)
    return torch.cat((amp*torch.cos(phase), amp*torch.sin(phase)), dim=-1)

def db_to_amp(x):
    """
    Converts a pytorch tensor from decibels to amplitude
    """
    return torch.pow(10, (0.5*x))

def amp_to_db(x):
    """
    Converts a pytorch tensor amplitude to decibels
    """
    return taf.amplitude_to_DB(x, 20, -80, 0)
