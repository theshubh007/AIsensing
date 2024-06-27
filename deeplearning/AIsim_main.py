#main AI simulation file
#based on deepMIMO5_sim.py, add different channel models, include CDL
#redesign from deepMIMO5 import Transmitter

import DeepMIMO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from matplotlib import colors

from deepMIMO5 import get_deepMIMOdata, DeepMIMODataset
from deepMIMO5 import StreamManagement, MyResourceGrid, Mapper, MyResourceGridMapper, MyDemapper, OFDMModulator, OFDMDemodulator, BinarySource, ebnodb2no, hard_decisions, calculate_BER
from deepMIMO5 import complex_normal, mygenerate_OFDMchannel, RemoveNulledSubcarriers, MyApplyOFDMChannel, time_lag_discrete_time_channel, cir_to_time_channel, MyApplyTimeChannel

from sionna_tf import MyLMMSEEqualizer, LMMSEEqualizer, SymbolLogits2LLRs#, OFDMDemodulator #ZFPrecoder, OFDMModulator, KroneckerPilotPattern, Demapper, RemoveNulledSubcarriers, 
from channel import MyLSChannelEstimator, LSChannelEstimator, ApplyTimeChannel#, time_lag_discrete_time_channel #, ApplyTimeChannel #cir_to_time_channel
from ldpc.encoding import LDPC5GEncoder
from ldpc.decoding import LDPC5GDecoder

import scipy

def to_numpy(input_array):
    # Check if the input is already a NumPy array
    if isinstance(input_array, np.ndarray):
        return input_array
    
    # Check if the input is a TensorFlow tensor
    try:
        import tensorflow as tf
        if isinstance(input_array, tf.Tensor):
            return input_array.numpy()
    except ImportError:
        pass
    
    # Check if the input is a PyTorch tensor
    try:
        import torch
        if isinstance(input_array, torch.Tensor):
            return input_array.numpy()
    except ImportError:
        pass
    
    raise TypeError("Input type not supported. Please provide a NumPy array, TensorFlow tensor, or PyTorch tensor.")

class Transmitter():
    def __init__(self, channeldataset='deepmimo', channeltype="ofdm", scenario='O1_60', dataset_folder='data/DeepMIMO', num_rx = 1, num_tx = 1, \
                 batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                 subcarrier_spacing=15e3, num_guard_carriers=[15,16], pilot_ofdm_symbol_indices=[2], \
                USE_LDPC = True, pilot_pattern = "kronecker", guards = True, showfig = True) -> None:
        self.channeltype = channeltype
        self.channeldataset = channeldataset
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.num_ofdm_symbols = num_ofdm_symbols
        self.num_bits_per_symbol = num_bits_per_symbol
        self.showfig = showfig
        self.pilot_pattern = pilot_pattern
        self.scenario = scenario
        self.dataset_folder = dataset_folder
        self.num_rx = num_rx
        self.num_tx = num_tx

        self.num_ut = num_rx #1
        self.num_bs = num_tx #1
        self.num_ut_ant = num_rx #2 #4
        self.num_bs_ant = 16 #8
        #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]

        # The number of transmitted streams is equal to the number of UT antennas
        # in both uplink and downlink
        #NUM_STREAMS_PER_TX = NUM_UT_ANT
        #NUM_UT_ANT = num_rx
        num_streams_per_tx = num_rx ##1
        # Create an RX-TX association matrix.
        # RX_TX_ASSOCIATION[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        # For example, considering a system with 2 RX and 4 TX, the RX-TX
        # association matrix could be
        # [ [1 , 1, 0, 0],
        #   [0 , 0, 1, 1] ]
        # which indicates that the RX 0 receives from TX 0 and 1, and RX 1 receives from
        # TX 2 and 3.
        #
        # we have only a single transmitter and receiver,
        # the RX-TX association matrix is simply:
        #RX_TX_ASSOCIATION = np.array([[1]]) #np.ones([num_rx, 1], int)
        RX_TX_ASSOCIATION = np.ones([num_rx, num_tx], int) #[[1]]
        self.STREAM_MANAGEMENT = StreamManagement(RX_TX_ASSOCIATION, num_streams_per_tx) #RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

        if guards:
            cyclic_prefix_length = 6 #0 #6 Length of the cyclic prefix
            if num_guard_carriers is None and type(num_guard_carriers) is not list:
                num_guard_carriers = [5,6] #[0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null=True #False
            if pilot_ofdm_symbol_indices is None and type(pilot_ofdm_symbol_indices) is not list:
                pilot_ofdm_symbol_indices=[2,11]
        else:
            cyclic_prefix_length = 0 #0 #6 Length of the cyclic prefix
            num_guard_carriers = [0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null=False
            pilot_ofdm_symbol_indices=[0,0]
        #pilot_pattern = "kronecker" #"kronecker", "empty"
        #fft_size = 76
        #num_ofdm_symbols=14
        RESOURCE_GRID = MyResourceGrid( num_ofdm_symbols=num_ofdm_symbols,
                                            fft_size=fft_size,
                                            subcarrier_spacing=subcarrier_spacing, #60e3, #30e3,
                                            num_tx=num_tx, #1
                                            num_streams_per_tx=num_streams_per_tx, #1
                                            cyclic_prefix_length=cyclic_prefix_length,
                                            num_guard_carriers=num_guard_carriers,
                                            dc_null=dc_null,
                                            pilot_pattern=pilot_pattern,
                                            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices)
        if showfig:
            RESOURCE_GRID.show() #14(OFDM symbol)*76(subcarrier) array=1064
            RESOURCE_GRID.pilot_pattern.show();
            #The pilot patterns are defined over the resource grid of *effective subcarriers* from which the nulled DC and guard carriers have been removed. 
            #This leaves us in our case with 76 - 1 (DC) - 5 (left guards) - 6 (right guards) = 64 effective subcarriers.

        if showfig and pilot_pattern == "kronecker":
            #actual pilot sequences for all streams which consists of random QPSK symbols.
            #By default, the pilot sequences are normalized, such that the average power per pilot symbol is
            #equal to one. As only every fourth pilot symbol in the sequence is used, their amplitude is scaled by a factor of two.
            plt.figure()
            plt.title("Real Part of the Pilot Sequences")
            for i in range(num_streams_per_tx):
                plt.stem(np.real(RESOURCE_GRID.pilot_pattern.pilots[0, i]),
                        markerfmt="C{}.".format(i), linefmt="C{}-".format(i),
                        label="Stream {}".format(i))
            plt.legend()
        print("Average energy per pilot symbol: {:1.2f}".format(np.mean(np.abs(RESOURCE_GRID.pilot_pattern.pilots[0,0])**2)))
        self.num_streams_per_tx = num_streams_per_tx
        self.RESOURCE_GRID = RESOURCE_GRID
        print("RG num_ofdm_symbols", RESOURCE_GRID.num_ofdm_symbols)
        print("RG ofdm_symbol_duration", RESOURCE_GRID.ofdm_symbol_duration)

        #num_bits_per_symbol = 4
        # Codeword length
        n = int(RESOURCE_GRID.num_data_symbols * num_bits_per_symbol) #num_data_symbols:64*14=896 896*4=3584, if empty 1064*4=4256

        #USE_LDPC = True
        if USE_LDPC:
            coderate = 0.5
            # Number of information bits per codeword
            k = int(n * coderate)  
            encoder = LDPC5GEncoder(k, n) #1824, 3648
            decoder = LDPC5GDecoder(encoder, hard_out=True)
            self.decoder = decoder
            self.encoder = encoder
        else:
            coderate = 1
            # Number of information bits per codeword
            k = int(n * coderate)  
        self.k = k # Number of information bits per codeword
        self.USE_LDPC = USE_LDPC
        self.coderate = coderate
        
        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = MyResourceGridMapper(RESOURCE_GRID) #ResourceGridMapper(RESOURCE_GRID)

        #receiver part
        self.mydemapper = MyDemapper("app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)

        #Channel part
        if self.channeldataset=='deepmimo':
            self.create_DeepMIMOchanneldataset() #get self.data_loader
        elif self.channeldataset=='cdl':
            self.create_CDLchanneldataset() #get self.cdl
        #call get_channelcir to get channelcir
        
        if self.channeltype=='ofdm':
            # Function that will apply the channel frequency response to an input signal
            #channel_freq = ApplyOFDMChannel(add_awgn=True)
            # Generate the OFDM channel
            self.applychannel = MyApplyOFDMChannel(add_awgn=True)
            #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            #(64, 1, 1, 1, 16, 1, 76)
        elif self.channeltype=='time': #time channel:
            #channel_time = ApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
            bandwidth = self.RESOURCE_GRID.bandwidth #4560000
            l_min, l_max = time_lag_discrete_time_channel(bandwidth) #-6, 20
            l_tot = l_max-l_min+1 #27
            self.l_tot = l_tot
            self.l_min = l_min
            self.l_max = l_max
            self.applychannel = MyApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True)
            # OFDM modulator and demodulator
            self.modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
            self.demodulator = OFDMDemodulator(self.RESOURCE_GRID.fft_size, l_min, self.RESOURCE_GRID.cyclic_prefix_length)


    def create_DeepMIMOchanneldataset(self):
        num_rx = self.num_rx
        #DeepMIMO provides multiple [scenarios](https://deepmimo.net/scenarios/) that one can select from. 
        #In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60). 
        #Please download the "O1_60" data files [from this page](https://deepmimo.net/scenarios/o1-scenario/).
        #The downloaded zip file should be extracted into a folder, and the parameter `'dataset_folder` should be set to point to this folder
        DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder, showfig=self.showfig)
        # The number of UE locations in the generated DeepMIMO dataset
        num_ue_locations = len(DeepMIMO_dataset[0]['user']['channel']) # 18100
        # Pick the largest possible number of user locations that is a multiple of ``num_rx``
        ue_idx = np.arange(num_rx*(num_ue_locations//num_rx)) #(18100,) 0~18099
        # Optionally shuffle the dataset to not select only users that are near each others
        np.random.shuffle(ue_idx)
        # Reshape to fit the requested number of users
        ue_idx = np.reshape(ue_idx, [-1, num_rx]) # In the shape of (floor(18100/num_rx) x num_rx) (18100,1)
        self.channeldataset = DeepMIMODataset(DeepMIMO_dataset=DeepMIMO_dataset, ue_idx=ue_idx)
        h, tau = next(iter(self.channeldataset)) #h: (1, 1, 1, 16, 10, 1), tau:(1, 1, 10)
        #complex gains `h` and delays `tau` for each path
        #print(h.shape) #[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        #print(tau.shape) #[num_rx, num_tx, num_paths]

        # torch dataloaders
        self.data_loader = DataLoader(dataset=self.channeldataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        if self.showfig:
            self.plotchimpulse()
    
    def create_CDLchanneldataset(self):
        try:
            from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
        except ImportError:
            pass
        # Define the number of UT and BS antennas.
        # For the CDL model, a single UT and BS are supported.
        #The CDL model only works for systems with a single transmitter and a single receiver. The transmitter and receiver can be equipped with multiple antennas.
        num_ut = 1
        num_bs = 1
        num_ut_ant = self.num_ut_ant #2
        num_bs_ant = self.num_bs_ant #16 #8

        carrier_frequency = 2.6e9 # Carrier frequency in Hz.
                          # This is needed here to define the antenna element spacing.

        ut_array = AntennaArray(num_rows=1,
                                num_cols=int(num_ut_ant/2),
                                polarization="dual",
                                polarization_type="cross",
                                antenna_pattern="38.901",
                                carrier_frequency=carrier_frequency)
        if self.showfig:
            ut_array.show()
            ut_array.show_element_radiation_pattern()

        bs_array = AntennaArray(num_rows=1,
                                num_cols=int(num_bs_ant/2),
                                polarization="dual",
                                polarization_type="cross",
                                antenna_pattern="38.901",
                                carrier_frequency=carrier_frequency)
        if self.showfig:
            bs_array.show()
            bs_array.show_element_radiation_pattern()
        
        #CDL channel model
        delay_spread = 300e-9 # Nominal delay spread in [s]. Please see the CDL documentation
                            # about how to choose this value. 

        direction = "uplink"  # The `direction` determines if the UT or BS is transmitting.
                            # In the `uplink`, the UT is transmitting.
        cdl_model = "B"       # Suitable values are ["A", "B", "C", "D", "E"]

        speed = 10            # UT speed [m/s]. BSs are always assumed to be fixed.
                            # The direction of travel will chosen randomly within the x-y plane.

        # Configure a channel impulse reponse (CIR) generator for the CDL model.
        # cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
        self.cdl = CDL(cdl_model, delay_spread, carrier_frequency, ut_array, bs_array, direction, min_speed=speed)
        #The cdl can be used to generate batches of random realizations of continuous-time
        #channel impulse responses, consisting of complex gains `a` and delays `tau` for each path. 
        #To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for `num_time_samples` samples.

    def get_channelcir(self,returnformat='numpy'):
        if self.channeldataset=='deepmimo':
            h_b, tau_b = next(iter(self.data_loader)) #h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
        elif self.channeldataset=='cdl':
            if self.channeltype=='ofdm':
                h_b, tau_b = self.cdl(batch_size=self.batch_size, num_time_steps=self.RESOURCE_GRID.num_ofdm_symbols, sampling_frequency=1/self.RESOURCE_GRID.ofdm_symbol_duration)
            elif self.channeltype=='time':
                h_b, tau_b = self.cdl(batch_size=self.batch_size, num_time_steps=self.RESOURCE_GRID.num_time_samples+self.l_tot-1, sampling_frequency=self.RESOURCE_GRID.bandwidth)
        # In CDL, Direction = "uplink" the UT is transmitting.
        # num_bs_ant = 16 = num_rx_ant
        # num_ut_ant = 2 = num_tx_ant
        #print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        #print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
        if returnformat=='numpy':
            h_b=to_numpy(h_b)
            tau_b=to_numpy(tau_b)
        if self.showfig:
            plt.figure()
            plt.title("Channel impulse response realization")
            plt.stem(tau_b[0,0,0,:]/1e-9, np.abs(h_b)[0,0,0,0,0,:,0])#10 different pathes
            plt.xlabel(r"$\tau$ [ns]")
            plt.ylabel(r"$|a|$")

            plt.figure()
            plt.title("Time evolution of path gain")
            plt.plot(np.arange(self.RESOURCE_GRID.num_ofdm_symbols)*self.RESOURCE_GRID.ofdm_symbol_duration/1e-6, np.real(h_b)[0,0,0,0,0,0,:])
            plt.plot(np.arange(self.RESOURCE_GRID.num_ofdm_symbols)*self.RESOURCE_GRID.ofdm_symbol_duration/1e-6, np.imag(h_b)[0,0,0,0,0,0,:])
            plt.legend(["Real part", "Imaginary part"])
            plt.xlabel(r"$t$ [us]")
            plt.ylabel(r"$a$");
        return h_b, tau_b

    # def get_htau_batch(self, returnformat='numpy'):
    #     h_b, tau_b = next(iter(self.data_loader)) #h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
    #     #print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    #     #print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
    #     if returnformat=="numpy":
    #         tau_b=tau_b.numpy()#torch tensor to numpy
    #         h_b=h_b.numpy()
    #     return h_b, tau_b

    def get_OFDMchannelresponse(self):
        h_b, tau_b = self.get_channelcir()
        from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
        from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

        frequencies = subcarrier_frequencies(self.RESOURCE_GRID.fft_size, self.RESOURCE_GRID.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, h_b, tau_b, normalize=True)
        #h_freq.shape #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]

        if self.showfig:
            plt.figure()
            plt.title("Channel frequency response at given time")
            plt.plot(np.real(h_freq[0,0,0,0,0,0,:]))
            plt.plot(np.imag(h_freq[0,0,0,0,0,0,:]))
            plt.xlabel("OFDM Symbol Index")
            plt.ylabel(r"$h$")
            plt.legend(["Real part", "Imaginary part"]);
        return h_freq
    
    def get_timechannelresponse(self):
        h_b, tau_b = self.get_channelcir()
        #h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)
        #[2, 1, 16, 1, 2, 1164, 17]
        h_time = cir_to_time_channel(self.RESOURCE_GRID.bandwidth, h_b, tau_b, l_min=self.l_min, l_max=self.l_max, normalize=True) 
        #h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        if self.showfig:
            plt.figure()
            plt.title("Discrete-time channel impulse response")
            plt.stem(np.abs(h_time[0,0,0,0,0,0]))
            plt.xlabel(r"Time step $\ell$")
            plt.ylabel(r"$|\bar{h}|$");
        return h_time
    

    def generateChannel(self, x_rg, no, channeltype='ofdm'):
        #x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        #h_b, tau_b = self.get_htau_batch()
        h_b, tau_b = self.get_channelcir()
        h_out = None
        #print(h_b.shape) #complex (64, 1, 1, 1, 16, 10, 1)[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps] 
        #print(tau_b.shape) #float (64, 1, 1, 10)[batch, num_rx, num_tx, num_paths]
        if channeltype=="ofdm": # Generate the OFDM channel response
            #computes the Fourier transform of the continuous-time channel impulse response at a set of `frequencies`, corresponding to the different subcarriers.
            ##h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76) 
            h_freq = mygenerate_OFDMchannel(h_b, tau_b, self.fft_size, subcarrier_spacing=60000.0, dtype=np.complex64, normalize_channel=True)
            #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
            #(64, 1, 1, 1, 16, 1, 76)

            remove_nulled_scs = RemoveNulledSubcarriers(self.RESOURCE_GRID)
            h_out = remove_nulled_scs(h_freq) #(64, 1, 1, 1, 16, 1, 64)
            if self.showfig:
                h_freq_plt = h_out[0,0,0,0,0,0] #get the last dimension: fft_size [76]
                #h_freq_plt = h_freq[0,0,0,0,0,0] #get the last dimension: fft_size [76]
                plt.figure()
                plt.plot(np.real(h_freq_plt))
                plt.plot(np.imag(h_freq_plt))
                plt.xlabel("Subcarrier index")
                plt.ylabel("Channel frequency response")
                plt.legend(["Ideal (real part)", "Ideal (imaginary part)"]);
                plt.title("Comparison of channel frequency responses");

            # Generate the OFDM channel
            channel_freq = MyApplyOFDMChannel(add_awgn=True)
            #h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            #(64, 1, 1, 1, 16, 1, 76)
            y = channel_freq([x_rg, h_freq, no]) #h_freq is array
            #Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex    
            #print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)

            #y = ApplyOFDMChannel(symbol_resourcegrid=x_rg, channel_frequency=h_freq, noiselevel=no, add_awgn=True)
            # y is the symbol received after the channel and noise
            #Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex    
        elif channeltype=="perfect":
            y=x_rg
        elif channeltype=="awgn":
            y=x_rg #(64, 1, 1, 14, 76)
            noise=complex_normal(y.shape, var=1.0)
            print(noise.dtype)
            noise = noise.astype(y.dtype)
            noise *= np.sqrt(no)
            y=y+noise
        elif channeltype=="time":
            bandwidth = self.RESOURCE_GRID.bandwidth #4560000
            l_min, l_max = time_lag_discrete_time_channel(bandwidth) #-6, 20
            l_tot = l_max-l_min+1 #27
            # Compute the discrete-time channel impulse reponse
            h_time = cir_to_time_channel(bandwidth, h_b, tau_b, l_min=l_min, l_max=l_max, normalize=True) 
            #h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1] complex[64, 1, 1, 1, 16, 1, 27]
            h_out = h_time
            if self.showfig:
                plt.figure()
                plt.title("Discrete-time channel impulse response")
                plt.stem(np.abs(h_time[0,0,0,0,0,0]))
                plt.xlabel(r"Time step $\ell$")
                plt.ylabel(r"$|\bar{h}|$");
            #channel_time = ApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
            channel_time = MyApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True)
            # OFDM modulator and demodulator
            modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
            demodulator = OFDMDemodulator(self.RESOURCE_GRID.fft_size, l_min, self.RESOURCE_GRID.cyclic_prefix_length)

            # OFDM modulation with cyclic prefix insertion
            #x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]
            x_time = modulator(x_rg) #output: (64, 1, 1, 1064)
            # Compute the channel output
            # This computes the full convolution between the time-varying
            # discrete-time channel impulse reponse and the discrete-time
            # transmit signal. With this technique, the effects of an
            # insufficiently long cyclic prefix will become visible. This
            # is in contrast to frequency-domain modeling which imposes
            # no inter-symbol interfernce.
            y_time = channel_time([x_time, h_time, no]) #[64, 1, 1, 1174]
            #y_time = channel_time([x_time, h_time]) #(64, 1, 1, 1090) complex

            #Do modulator and demodulator test
            y_test = demodulator(x_time)
            differences = np.abs(x_rg - y_test)
            threshold=1e-7
            num_differences = np.sum(differences > threshold)
            print("Number of differences:", num_differences)
            print(np.allclose(x_rg, y_test))
            print("Demodulation error (L2 norm):", np.linalg.norm(x_rg - y_test))
            
            # OFDM demodulation and cyclic prefix removal
            y = demodulator(y_time)
            #y = y_test
            #y: [64, 1, 1, 14, 76]
        return y, h_out

    def uplinktransmission(self, b=None, no=0, perfect_csi=False):

        if b is None:
            binary_source = BinarySource()
            # Start Transmitter self.k Number of information bits per codeword
            b = binary_source([self.batch_size, 1, self.num_streams_per_tx, self.k]) #[64,1,1,3584] if empty [64,1,1,1536] [batch_size, num_tx, num_streams_per_tx, num_databits]
        if self.USE_LDPC:
            c = self.encoder(b) #tf.tensor[64,1,1,3072] [batch_size, num_tx, num_streams_per_tx, num_codewords]
        else:
            c = b
        x = self.mapper(c) #np.array[64,1,1,896] if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x_rg = self.rg_mapper(x) ##complex array[64,1,1,14,76] 14*76=1064
        #output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        #we generate random batches of CIR, transform them in the frequency domain and apply them to the resource grid in the frequency domain.
        if self.channeltype=='ofdm':
            h_out = self.get_OFDMchannelresponse()
            print("h_freq shape:", h_out.shape)
        elif self.channeltype=='time':
            h_out = self.get_timechannelresponse()
        
        #apply channel
        y = self.applychannel([x_rg, h_out, no])
        
        return y, x_rg
    
    def channel_estimation(self, y, no, h_out=None):
        #perform channel estimation via pilots
        print("Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots)) #1
        # Receiver
        ls_est = LSChannelEstimator(self.RESOURCE_GRID, interpolation_type="lin_time_avg")
        #ls_est = MyLSChannelEstimator(self.RESOURCE_GRID, interpolation_type="lin_time_avg")

        #Observed resource grid y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], (64, 1, 1, 14, 76) complex
        #no : [batch_size, num_rx, num_rx_ant] 
        h_hat, err_var = ls_est([y, no]) #tf tensor (64, 1, 1, 1, 1, 14, 64), (1, 1, 1, 1, 1, 14, 64)
        #h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        #Channel estimates accross the entire resource grid for all transmitters and streams

        if self.showfig:
            h_est = h_hat[0,0,0,0,0,0] #(64, 1, 1, 1, 1, 14, 44)
            plt.figure()
            if h_out is not None:
                h_perfect = h_out[0,0,0,0,0,0] #(64, 1, 1, 1, 16, 1, 44)
                plt.plot(np.real(h_perfect))
                plt.plot(np.imag(h_perfect))
            plt.plot(np.real(h_est), "--")
            plt.plot(np.imag(h_est), "--")
            plt.xlabel("Subcarrier index")
            plt.ylabel("Channel frequency response")
            plt.legend(["Ideal (real part)", "Ideal (imaginary part)", "Estimated (real part)", "Estimated (imaginary part)"]);
            plt.title("Comparison of channel frequency responses");
        return h_hat, err_var

    def receiver(self, y, no, x_rg, b=None, perfect_csi= False):
        print(self.RESOURCE_GRID.pilot_pattern) #<__main__.EmptyPilotPattern object at 0x7f2659dfd9c0>
        print("Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots))
        if self.pilot_pattern == "empty" and perfect_csi == False: #"kronecker", "empty"
            #no channel estimation
            llr = self.mydemapper([y, no]) #[64, 1, 1, 14, 304]
            # Reshape the array by collapsing the last two dimensions
            llr_est = llr.reshape(llr.shape[:-2] + (-1,)) #(64, 1, 1, 4256)

            llr_perfect = self.mydemapper([x_rg, no]) #[64, 1, 1, 14, 304]
            llr_perfect = llr_perfect.reshape(llr_perfect.shape[:-2] + (-1,)) #(64, 1, 1, 4256)
            b_perfect = hard_decisions(llr_perfect, np.int32) ##(64, 1, 1, 4256) 0,1 [64, 1, 1, 14, 304] 2128
            #BER=calculate_BER(b, b_perfect)
            #print("Perfect BER:", BER)
        else: # channel estimation or perfect_csi
            #perform channel estimation via pilots
            h_hat, err_var = self.channel_estimation(y, no)
            
            #lmmse_equ = LMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            lmmse_equ = MyLMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            #input (y, h_hat, err_var, no)
            #Received OFDM resource grid after cyclic prefix removal and FFT y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            #Channel estimates for all streams from all transmitters h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
            x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no]) #(64, 1, 1, 912), (64, 1, 1, 912)
            #Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
            #Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
            x_hat=x_hat.numpy() #(64, 1, 1, 912)
            no_eff=no_eff.numpy() #(64, 1, 1, 912)
            no_eff=np.mean(no_eff)

            llr_est = self.mydemapper([x_hat, no_eff]) #(64, 1, 1, 3072)
            #output: [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]

        #llr_est #(64, 1, 1, 4256)
        if self.USE_LDPC:
            b_hat_tf = self.decoder(llr_est) #[64, 1, 1, 2128]
            b_hat = b_hat_tf.numpy()
        else:
            b_hat = hard_decisions(llr_est, np.int32) 
        if b is not None:
            BER=calculate_BER(b, b_hat)
            print("BER Value:", BER)
            return b_hat, BER
        else:
            return b_hat, None


    def __call__(self, b=None, ebno_db = 15.0, channeltype='ofdm', perfect_csi=False):
        # Transmitter
        y, x_rg = self.uplinktransmission(b=b, ebno_db=ebno_db, perfect_csi=perfect_csi)
        print("y shape:", y.shape)

        #Compute the noise power for a given Eb/No value. This takes not only the coderate but also the overheads related pilot transmissions and nulled carriers
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        # Convert it to a NumPy float
        no = np.float32(no) #0.0158

        self.channel_estimation(y, no, h_out=None)

        b_hat, BER = self.receiver(y, no, x_rg, b=None, perfect_csi= False)
        return b_hat, BER

def test_CDLchannel():
    transmit = Transmitter(channeldataset='cdl', channeltype='ofdm', num_rx = 1, num_tx = 1, \
                    batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
                    subcarrier_spacing=60e3, \
                    USE_LDPC = False, pilot_pattern = "empty", guards=False, showfig=showfigure)
    transmit.get_channelcir()
    transmit.get_OFDMchannelresponse()
    b_hat, BER = transmit(ebno_db = 15.0)

if __name__ == '__main__':

    #testOFDMModulatorDemodulator()
    scenario='O1_60'
    #dataset_folder='data' #r'D:\Dataset\CommunicationDataset\O1_60'
    dataset_folder='data/DeepMIMO'
    #dataset_folder=r'D:\Dataset\CommunicationDataset\O1_60'
    ofdmtest = True
    showfigure = True
    if ofdmtest is True:
        test_CDLchannel()
    print("Finished")



