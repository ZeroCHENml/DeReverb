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
    specshow(spec, x_axis='time', y_axis='mel', sr=sr)