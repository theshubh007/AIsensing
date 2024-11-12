# main AI simulation file
# based on deepMIMO5_sim.py, add different channel models, include CDL
# redesign from deepMIMO5 import Transmitter

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import tensorflow as tf
from datetime import datetime


from deepMIMO5 import (
    get_deepMIMOdata,
    DeepMIMODataset,
    flatten_last_dims,
    myexpand_to_rank,
)
from deepMIMO5 import count_errors, count_block_errors
from deepMIMO5 import (
    StreamManagement,
    MyResourceGrid,
    Mapper,
    MyResourceGridMapper,
    MyDemapper,
    BinarySource,
    ebnodb2no,
    hard_decisions,
    calculate_BER,
)
from deepMIMO5 import (
    complex_normal,
    mygenerate_OFDMchannel,
    RemoveNulledSubcarriers,
    MyApplyOFDMChannel,
    MyApplyTimeChannel,
)
from deepMIMO5 import (
    time_lag_discrete_time_channel,
    cir_to_time_channel,
    cir_to_ofdm_channel,
    subcarrier_frequencies,
)

# from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
# from sionna.channel import ApplyOFDMChannel#, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna_tf import (
    MyLMMSEEqualizer,
)  # , LMMSEEqualizer, SymbolLogits2LLRs#, OFDMDemodulator #ZFPrecoder, OFDMModulator, KroneckerPilotPattern, Demapper, RemoveNulledSubcarriers,

from deepMIMO5 import OFDMModulator, OFDMDemodulator

# from sionna.ofdm import OFDMModulator, OFDMDemodulator
# from sionna_tf import OFDMModulator, OFDMDemodulator
from channel import (
    MyLSChannelEstimator,
)  # , LSChannelEstimator, ApplyTimeChannel#, time_lag_discrete_time_channel #, ApplyTimeChannel #cir_to_time_channel
from ldpc.encoding import LDPC5GEncoder
from ldpc.decoding import LDPC5GDecoder

import os
from Neural_receiver_v1 import process_signal

IMG_FORMAT = ".pdf"  # ".png"


def ber_plot(
    ebno_dbs,
    bers,
    legend="",
    ylabel="BER",
    title="Bit Error Rate",
    ebno=True,
    xlim=None,
    ylim=None,
    is_bler=False,
    savefigpath="./data/ber.jpg",
):
    if not isinstance(legend, list):
        assert isinstance(legend, str)
        legend = [legend]

    assert isinstance(title, str), "title must be str."

    fig, ax = plt.subplots(figsize=(16, 10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(title, fontsize=25)
    # return figure handle
    if isinstance(bers, list):
        for idx, b in enumerate(bers):
            if is_bler:
                line_style = "--"
            else:
                line_style = ""
            plt.semilogy(ebno_dbs, b, line_style, linewidth=2)
    else:
        if is_bler:
            line_style = "--"
        else:
            line_style = ""
        plt.semilogy(ebno_dbs, bers, line_style, linewidth=2)

    plt.grid(which="both")
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.legend(legend, fontsize=20)
    if savefigpath is not None:
        plt.savefig(savefigpath)
        plt.close(fig)
    else:
        # plt.close(fig)
        pass
    return fig, ax


def ber_plot_single(
    ebno_dbs, bers, title="BER Simulation", savefigpath="./data/ber.jpg"
):

    fig, ax = plt.subplots(figsize=(16, 10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # A tuple of two floats defining x-axis limits.
    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    is_bler = False
    plt.title(title, fontsize=25)
    # return figure handle
    if is_bler:
        line_style = "--"
    else:
        line_style = ""
    plt.semilogy(ebno_dbs, bers, line_style, linewidth=2)

    plt.grid(which="both")
    ebno = True
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    ylabel = "BER"
    plt.ylabel(ylabel, fontsize=25)
    # legend=""
    # plt.legend(legend, fontsize=20)
    if savefigpath is not None:
        plt.savefig(savefigpath)
        plt.close(fig)


def ber_plot_single2(
    ebno_dbs, bers, is_bler=False, title="BER Simulation", savefigpath="./data/ber.jpg"
):

    fig, ax = plt.subplots(figsize=(16, 10))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # A tuple of two floats defining x-axis limits.
    # if xlim is not None:
    #     plt.xlim(xlim)
    # if ylim is not None:
    #     plt.ylim(ylim)

    plt.title(title, fontsize=25)
    # return figure handle
    if is_bler:
        line_style = "--"
    else:
        line_style = ""
    plt.semilogy(ebno_dbs, bers, line_style, linewidth=2)

    plt.grid(which="both")
    ebno = True
    if ebno:
        plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
    else:
        plt.xlabel(r"$E_s/N_0$ (dB)", fontsize=25)
    if is_bler:
        ylabel = "BLER"
    else:
        ylabel = "BER"

    plt.ylabel(ylabel, fontsize=25)
    # legend=""
    # plt.legend(legend, fontsize=20)
    if savefigpath is not None:
        plt.savefig(savefigpath)
        plt.close(fig)


def sim_ber(ebno_dbs, eval_transceiver, b, batch_size):
    # num_points = 100  # Example value, replace with the actual value
    ebno_dbs_np = np.array(ebno_dbs, dtype=np.float64)  # Cast to the desired data type
    batch_size_np = np.array(
        batch_size, dtype=np.int32
    )  # Cast to the desired data type
    num_points = ebno_dbs_np.shape[0]  # 20

    bit_n = np.size(b)  # 272384
    block_n = np.size(b[..., -1])  # 128, b: (128, 1, 1, 2128)

    bers = []
    blers = []
    BERs = []

    for ebno_db in ebno_dbs:
        # b_hat, BER = eval_transceiver(b=b, ebno_db = ebno_db, channeltype=channeltype)
        b_hat, BER = eval_transceiver(b=b, ebno_db=ebno_db)
        BERs.append(BER)
        # count errors
        bit_e = count_errors(b, b_hat)
        block_e = count_block_errors(b, b_hat)

        # Initialize NumPy arrays bit_errors, block_errors, nb_bits, and nb_blocks (if not already initialized)
        nb_bits = np.zeros_like(bit_n, dtype=np.int64)
        nb_blocks = np.zeros_like(block_n, dtype=np.int64)
        # bit_errors = 0
        # block_errors = 0
        bit_errors = np.zeros_like(bit_e, dtype=np.int64)
        block_errors = np.zeros_like(block_e, dtype=np.int64)

        # Update variables
        bit_errors = bit_errors + np.int64(bit_e)
        block_errors = block_errors + np.int64(block_e)
        nb_bits = nb_bits + np.int64(bit_n)
        nb_blocks = nb_blocks + np.int64(block_n)

        ber = np.divide(bit_errors.astype(np.float64), nb_bits.astype(np.float64))
        bler = np.divide(block_errors.astype(np.float64), nb_blocks.astype(np.float64))

        # Replace NaN values with zeros
        ber = np.where(np.isnan(ber), np.zeros_like(ber), ber)
        bler = np.where(np.isnan(bler), np.zeros_like(bler), bler)

        bers.append(ber)
        blers.append(bler)

    return bers, blers, BERs


def simulationloop(ebno_dbs, eval_transceiver, b=None, perfect_csi=False):
    bers = []
    for ebno_db in ebno_dbs:
        b_hat, BER = eval_transceiver(b=b, ebno_db=ebno_db, perfect_csi=perfect_csi)
        bers.append(BER)
    bers_np = np.array(bers)
    return bers_np


class NearestNeighborInterpolator:
    # pylint: disable=line-too-long
    r"""NearestNeighborInterpolator(pilot_pattern)

    Nearest-neighbor channel estimate interpolation on a resource grid.

    This class assigns to each element of an OFDM resource grid one of
    ``num_pilots`` provided channel estimates and error
    variances according to the nearest neighbor method. It is assumed
    that the measurements were taken at the nonzero positions of a
    :class:`~sionna.ofdm.PilotPattern`.

    The figure below shows how four channel estimates are interpolated
    accross a resource grid. Grey fields indicate measurement positions
    while the colored regions show which resource elements are assigned
    to the same measurement value.

    .. image:: ../figures/nearest_neighbor_interpolation.png

    Parameters
    ----------
    pilot_pattern : PilotPattern
        An instance of :class:`~sionna.ofdm.PilotPattern`

    Input
    -----
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimates for the pilot-carrying resource elements

    err_var : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_pilot_symbols], tf.complex
        Channel estimation error variances for the pilot-carrying resource elements

    Output
    ------
    h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, pilot_pattern):
        # super().__init__()

        assert (
            pilot_pattern.num_pilot_symbols > 0
        ), """The pilot pattern cannot be empty"""

        # Reshape mask to shape [-1,num_ofdm_symbols,num_effective_subcarriers]
        mask = np.array(pilot_pattern.mask)  # (1, 2, 14, 64)
        mask_shape = (
            mask.shape
        )  # Store to reconstruct the original shape (1, 2, 14, 64)
        mask = np.reshape(mask, [-1] + list(mask_shape[-2:]))  # (2, 14, 64)

        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = pilot_pattern.pilots  # (1, 2, 128)
        pilots = np.reshape(pilots, [-1] + [pilots.shape[-1]])  # (2, 128)

        max_num_zero_pilots = np.max(np.sum(np.abs(pilots) == 0, -1))  # 64
        assert (
            max_num_zero_pilots < pilots.shape[-1]
        ), """Each pilot sequence must have at least one nonzero entry"""

        # Compute gather indices for nearest neighbor interpolation
        gather_ind = np.zeros_like(mask, dtype=np.int32)  # (2, 14, 64)
        for a in range(gather_ind.shape[0]):  # For each pilot pattern...
            i_p, j_p = np.where(mask[a])  # ...determine the pilot indices

            for i in range(mask_shape[-2]):  # Iterate over...
                for j in range(mask_shape[-1]):  # ... all resource elements

                    # Compute Manhattan distance to all pilot positions
                    d = np.abs(i - i_p) + np.abs(j - j_p)

                    # Set the distance at all pilot positions with zero energy
                    # equal to the maximum possible distance
                    d[np.abs(pilots[a]) == 0] = np.sum(mask_shape[-2:])

                    # Find the pilot index with the shortest distance...
                    ind = np.argmin(d)

                    # ... and store it in the index tensor
                    gather_ind[a, i, j] = ind

        # Reshape to the original shape of the mask, i.e.:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,...
        #  ..., num_effective_subcarriers]
        # self._gather_ind = tf.reshape(gather_ind, mask_shape)
        self._gather_ind = np.reshape(
            gather_ind, mask_shape
        )  # _gather_ind: (1, 2, 14, 64)
        np.save("data/inter_gather_ind.npy", self._gather_ind)

    def mygather(self, inputs, method="tf"):
        # Interpolate through gather. Shape:
        # [num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  ..., num_effective_subcarriers, k, l, m]
        if method == "tf":
            outputs = tf.gather(inputs, self._gather_ind, 2, batch_dims=2)
        # batch_dims: An optional parameter that specifies the number of batch dimensions. It controls how many leading dimensions are considered as batch dimensions.
        elif method == "np":
            result = inputs.copy()
            # Gather along each dimension
            # for dim in range(batch_dims, len(inputs.shape)): #2-6
            #     result = np.take(result, indices, axis=dim)

            # gather_ind_nobatch = indices[0, 0] #ignore first two dimensions as batch (14, 64)
            # result = np.take(result, gather_ind_nobatch, axis=2) #(1, 2, 14, 64, 2, 1, 16)
            gather_ind_nobatch = self._gather_ind[
                0, 0
            ]  # ignore first two dimensions as batch (14, 64)
            result1 = np.take(
                result, gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            gather_ind_nobatch = self._gather_ind[
                0, 1
            ]  # ignore first two dimensions as batch (14, 64)
            outputs = np.take(
                result, gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            outputs[0, 0, :, :, :, :, :] = result1[0, 0, :, :, :, :, :]
        else:  # Wrong result
            # outputs = np.take(inputs, self._gather_ind, axis=2, mode='wrap') #(1, 2, 1, 2, 14, 64, 2, 1, 16), _gather_ind: (1, 2, 14, 64)
            # outputs: (1, 2, 14, 64, 2, 1, 16)
            self._gather_ind_nobatch = self._gather_ind[
                0, 0
            ]  # ignore first two dimensions as batch (14, 64)
            outputs = np.take(
                inputs, self._gather_ind_nobatch, axis=2
            )  # (1, 2, 14, 64, 2, 1, 16)
            np.save("data/outputs_inter.npy", outputs)
            # outputs = inputs[:, :, self._gather_ind, :, :, :] #(1, 2, 1, 2, 14, 64, 2, 1, 16)
            # Perform the gathe
            # axis = 2
            # batch_dims = 2
            # outputs = np.take_along_axis(inputs, self._gather_ind, axis=axis, batch_dims=batch_dims)
        return outputs

    def _interpolate(self, inputs):
        # inputs has shape: (1, 2, 128, 2, 1, 16)
        # [k, l, m, num_tx, num_streams_per_tx, num_pilots]

        # Transpose inputs to bring batch_dims for gather last. New shape:
        # [num_tx, num_streams_per_tx, num_pilots, k, l, m]
        # perm = tf.roll(tf.range(tf.rank(inputs)), -3, 0)
        # inputs = tf.transpose(inputs, perm)
        perm = np.roll(
            np.arange(np.ndim(inputs)), -3, 0
        )  # shift the dimensions. (2, 1, 16, 1, 2, 128)
        inputs = np.transpose(inputs, perm)  # (1, 2, 128, 2, 1, 16)

        # np.save('inputs_inter.npy', inputs)
        outputs = self.mygather(inputs)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx,...
        #  ..., num_ofdm_symbols, num_effective_subcarriers]
        # perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        # outputs = tf.transpose(outputs, perm)
        perm = np.roll(np.arange(np.ndim(outputs)), 3, 0)  # [4, 5, 6, 0, 1, 2, 3]
        outputs = np.transpose(outputs, perm)  # (2, 1, 16, 1, 2, 14, 64)

        return outputs

    def __call__(self, h_hat, err_var):

        h_hat = self._interpolate(h_hat)
        err_var = self._interpolate(err_var)
        return h_hat, err_var


class MyLSChannelEstimatorNP:
    # pylint: disable=line-too-long
    r"""LSChannelEstimator(resource_grid, interpolation_type="nn", interpolator=None, dtype=tf.complex64, **kwargs)

    Layer implementing least-squares (LS) channel estimation for OFDM MIMO systems.

    After LS channel estimation at the pilot positions, the channel estimates
    and error variances are interpolated accross the entire resource grid using
    a specified interpolation function.

    For simplicity, the underlying algorithm is described for a vectorized observation,
    where we have a nonzero pilot for all elements to be estimated.
    The actual implementation works on a full OFDM resource grid with sparse
    pilot patterns. The following model is assumed:

    .. math::

        \mathbf{y} = \mathbf{h}\odot\mathbf{p} + \mathbf{n}

    where :math:`\mathbf{y}\in\mathbb{C}^{M}` is the received signal vector,
    :math:`\mathbf{p}\in\mathbb{C}^M` is the vector of pilot symbols,
    :math:`\mathbf{h}\in\mathbb{C}^{M}` is the channel vector to be estimated,
    and :math:`\mathbf{n}\in\mathbb{C}^M` is a zero-mean noise vector whose
    elements have variance :math:`N_0`. The operator :math:`\odot` denotes
    element-wise multiplication.

    The channel estimate :math:`\hat{\mathbf{h}}` and error variances
    :math:`\sigma^2_i`, :math:`i=0,\dots,M-1`, are computed as

    .. math::

        \hat{\mathbf{h}} &= \mathbf{y} \odot
                           \frac{\mathbf{p}^\star}{\left|\mathbf{p}\right|^2}
                         = \mathbf{h} + \tilde{\mathbf{h}}\\
             \sigma^2_i &= \mathbb{E}\left[\tilde{h}_i \tilde{h}_i^\star \right]
                         = \frac{N_0}{\left|p_i\right|^2}.

    The channel estimates and error variances are then interpolated accross
    the entire resource grid.

    Parameters
    ----------
    resource_grid : ResourceGrid
        An instance of :class:`~sionna.ofdm.ResourceGrid`.

    interpolation_type : One of ["nn", "lin", "lin_time_avg"], string
        The interpolation method to be used.
        It is ignored if ``interpolator`` is not `None`.
        Available options are :class:`~sionna.ofdm.NearestNeighborInterpolator` (`"nn`")
        or :class:`~sionna.ofdm.LinearInterpolator` without (`"lin"`) or with
        averaging across OFDM symbols (`"lin_time_avg"`).
        Defaults to "nn".

    interpolator : BaseChannelInterpolator
        An instance of :class:`~sionna.ofdm.BaseChannelInterpolator`,
        such as :class:`~sionna.ofdm.LMMSEInterpolator`,
        or `None`. In the latter case, the interpolator specfied
        by ``interpolation_type`` is used.
        Otherwise, the ``interpolator`` is used and ``interpolation_type``
        is ignored.
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, no) :
        Tuple:

    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], tf.complex
        Observed resource grid

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
        Channel estimates accross the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_ls``, tf.float
        Channel estimation error variance accross the entire resource grid
        for all transmitters and streams
    """

    def __init__(
        self, resource_grid, interpolation_type="nn", interpolator=None, **kwargs
    ):
        # super().__init__(dtype=dtype, **kwargs)

        # assert isinstance(resource_grid, ResourceGrid),\
        #     "You must provide a valid instance of ResourceGrid."
        self._pilot_pattern = resource_grid.pilot_pattern

        # added test code
        mask = np.array(self._pilot_pattern.mask)  # (1, 2, 14, 64)
        mask_shape = mask.shape  # Store to reconstruct the original shape
        # Reshape the pilots to shape [-1, num_pilot_symbols]
        pilots = self._pilot_pattern.pilots  # (1, 2, 128)
        print(mask_shape)  # (1, 2, 14, 64)
        # print('mask:', mask[0,0,0,:]) #all 0
        # print('pilots:', pilots[0,0,:]) #(1, 2, 128) -0.99999994-0.99999994j 0.        +0.j          0.99999994+0.99999994j
        # 0.99999994-0.99999994j  0.        +0.j          0.99999994-0.99999994j
        self._removed_nulled_scs = RemoveNulledSubcarriers(resource_grid)

        assert interpolation_type in [
            "nn",
            "lin",
            "lin_time_avg",
            None,
        ], "Unsupported `interpolation_type`"
        self._interpolation_type = interpolation_type

        # if self._interpolation_type == "nn":
        self._interpol = NearestNeighborInterpolator(self._pilot_pattern)
        # elif self._interpolation_type == "lin":
        #     self._interpol = LinearInterpolator(self._pilot_pattern)
        # elif self._interpolation_type == "lin_time_avg":
        #     self._interpol = LinearInterpolator(self._pilot_pattern,
        #                                         time_avg=True)

        # Precompute indices to gather received pilot signals
        num_pilot_symbols = self._pilot_pattern.num_pilot_symbols  # 128
        mask = flatten_last_dims(self._pilot_pattern.mask)  # (1, 2, 896)
        # np.save('mask.npy', mask)
        # pilot_ind = tf.argsort(mask, axis=-1, direction="DESCENDING") #(1, 2, 896)
        ##np.argsort is small to bigger (index of 0s first, index of 1s later), add [..., ::-1] to flip the results from bigger to small (index 1s first, index 0s later)
        pilot_ind = np.argsort(mask, axis=-1)[
            ..., ::-1
        ]  # (1, 2, 896) reverses the order of the indices along the last axis
        # select num_pilot_symbols, i.e., get all index of 1s in mask, due to the np.argsort(small to bigger), the order for these 1s index is not sorted
        self._pilot_ind = pilot_ind[..., :num_pilot_symbols]  # (1, 2, 128)
        # add sort again for these 1s index (small index to bigger)
        # analysis in tfnumpy.py
        self._pilot_ind = np.sort(self._pilot_ind)
        print(self._pilot_ind)

    def estimate_at_pilot_locations(self, y_pilots, no):

        # y_pilots : [batch_size, num_rx, num_rx_ant, num_tx, num_streams,
        #               num_pilot_symbols], tf.complex
        #     The observed signals for the pilot-carrying resource elements.

        # no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        #   tf.float
        #     The variance of the AWGN.

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        # h_ls = tf.math.divide_no_nan(y_pilots, self._pilot_pattern.pilots)
        # h_ls = np.divide(y_pilots, self._pilot_pattern.pilots) #(2, 1, 16, 1, 2, 128)
        # h_ls = np.nan_to_num(h_ls) #replaces NaN (Not-a-Number) values with zeros.

        h_ls = np.divide(
            y_pilots,
            self._pilot_pattern.pilots,
            out=np.zeros_like(y_pilots),
            where=self._pilot_pattern.pilots != 0,
        )
        # h_ls: (2, 1, 16, 1, 2, 128)
        # Compute error variance and broadcast to the same shape as h_ls
        # Expand rank of no for broadcasting
        # no = expand_to_rank(no, tf.rank(h_ls), -1)
        # no = myexpand_to_rank(no, h_ls.ndim, -1) #(1, 1, 1, 1, 1, 1)

        # Expand rank of pilots for broadcasting
        # pilots = expand_to_rank(self._pilot_pattern.pilots, tf.rank(h_ls), 0)
        pilots = myexpand_to_rank(
            self._pilot_pattern.pilots, h_ls.ndim, 0
        )  # (1, 1, 1, 1, 2, 128)

        # Compute error variance, broadcastable to the shape of h_ls
        # err_var = tf.math.divide_no_nan(no, tf.abs(pilots)**2)
        pilotssquare = np.abs(pilots) ** 2
        # err_var = np.divide(no, pilotssquare)
        # err_var = np.nan_to_num(err_var) #replaces NaN (Not-a-Number) values with zeros. (1, 1, 1, 1, 2, 128)
        no_array = np.full(pilots.shape, no, dtype=np.float32)  # (1, 1, 1, 1, 2, 128)
        err_var = np.divide(
            no_array, pilotssquare, out=np.zeros_like(no_array), where=pilotssquare != 0
        )  # (1, 1, 1, 1, 2, 128)

        return h_ls, err_var

    # def call(self, inputs):
    def __call__(self, inputs):

        y, no = inputs  # y: (64, 1, 1, 14, 76) complex64
        y = to_numpy(y)  # (2, 1, 16, 14, 76)
        no = np.array(no, dtype=np.float32)

        # y has shape:
        # [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,..
        # ... fft_size]
        #
        # no can have shapes [], [batch_size], [batch_size, num_rx]
        # or [batch_size, num_rx, num_rx_ant]

        # Removed nulled subcarriers (guards, dc)
        y_eff = self._removed_nulled_scs(y)  # (2, 1, 16, 14, 64) complex64

        # Flatten the resource grid for pilot extraction
        # New shape: [...,num_ofdm_symbols*num_effective_subcarriers]
        y_eff_flat = flatten_last_dims(y_eff)  # (2, 1, 16, 896)
        # plt.figure()
        # plt.plot(np.real(y_eff_flat[0,0,0,:]))
        # plt.plot(np.imag(y_eff_flat[0,0,0,:]))
        # plt.title('y_eff_flat')

        # Gather pilots along the last dimensions
        # Resulting shape: y_eff_flat.shape[:-1] + pilot_ind.shape, i.e.:
        # [batch_size, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ..., num_pilot_symbols]
        # y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)
        # y_pilots = y_eff_flat[self._pilot_ind] #(2, 1, 16, 896)
        # y_pilots = np.take(y_eff_flat, self._pilot_ind, axis=-1) #y_eff_flat:(2, 1, 16, 896), _pilot_ind:(1, 2, 128) => y_pilots(2, 1, 16, 1, 2, 128)
        # Gather elements from y_eff_flat based on pilot_ind
        y_pilots = y_eff_flat[..., self._pilot_ind]  # (2, 1, 16, 1, 2, 128)

        # plt.figure()
        # plt.plot(np.real(y_pilots[0,0,0,0,0,:]))
        # plt.plot(np.imag(y_pilots[0,0,0,0,0,:]))
        # plt.title('y_pilots')
        # np.save('y_eff_flat.npy', y_eff_flat)
        # np.save('pilot_ind.npy', self._pilot_ind)
        # np.save('y_pilots.npy', y_pilots)

        # Compute LS channel estimates
        # Note: Some might be Inf because pilots=0, but we do not care
        # as only the valid estimates will be considered during interpolation.
        # We do a save division to replace Inf by 0.
        # Broadcasting from pilots here is automatic since pilots have shape
        # [num_tx, num_streams, num_pilot_symbols]
        h_hat, err_var = self.estimate_at_pilot_locations(
            y_pilots, no
        )  # y_pilots: (2, 1, 16, 1, 2, 128), h_hat:(2, 1, 16, 1, 2, 128)
        # np.save('h_hat_pilot.npy', h_hat) #(2, 1, 16, 1, 2, 128)
        # Interpolate channel estimates over the resource grid
        if self._interpolation_type is not None:
            h_hat, err_var = self._interpol(
                h_hat, err_var
            )  # h_hat: (2, 1, 16, 1, 2, 128)=>
            # np.save('h_hat_inter.npy', h_hat)
            # err_var = tf.maximum(err_var, tf.cast(0, err_var.dtype))
            err_var = np.maximum(err_var, 0)

        return h_hat, err_var


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

    raise TypeError(
        "Input type not supported. Please provide a NumPy array, TensorFlow tensor, or PyTorch tensor."
    )


class Transmitter:
    def __init__(
        self,
        channeldataset="deepmimo",
        channeltype="ofdm",
        scenario="O1_60",
        dataset_folder="data/DeepMIMO",
        direction="uplink",
        num_ut=1,
        num_ut_ant=2,
        num_bs=1,
        num_bs_ant=16,
        batch_size=64,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=15e3,
        num_guard_carriers=None,
        pilot_ofdm_symbol_indices=None,
        USE_LDPC=True,
        pilot_pattern="kronecker",
        guards=True,
        showfig=True,
        savedata=True,
        outputpath=None,
    ) -> None:
        # num_guard_carriers=[15,16]
        self.channeltype = channeltype
        self.channeldataset = channeldataset
        self.fft_size = fft_size
        self.batch_size = batch_size
        self.num_ofdm_symbols = num_ofdm_symbols
        self.num_bits_per_symbol = num_bits_per_symbol
        self.showfig = showfig
        self.savedata = savedata
        self.pilot_pattern = pilot_pattern
        self.scenario = scenario
        self.dataset_folder = dataset_folder
        self.direction = direction
        self.num_ut = num_ut  # num_rx #1
        self.num_bs = num_bs  # num_tx #1
        self.num_ut_ant = num_ut_ant  # num_rx #2 #4
        self.num_bs_ant = num_bs_ant  # 8
        self.num_time_steps = 1  # num_ofdm_symbols #???
        self.outputpath = outputpath
        # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        if direction == "uplink":  # the UT is transmitting.
            self.num_tx = self.num_ut
            self.num_rx = self.num_bs
            # The number of transmitted streams is equal to the number of UT antennas
            # in both uplink and downlink
            # NUM_STREAMS_PER_TX = NUM_UT_ANT
            # NUM_UT_ANT = num_rx
            num_streams_per_tx = num_ut_ant  # num_rx ##1
        else:  # downlink
            self.num_tx = self.num_bs
            self.num_rx = self.num_ut
            num_streams_per_tx = num_bs_ant  # num_rx ##1
        self.num_streams_per_tx = num_streams_per_tx

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
        # RX_TX_ASSOCIATION = np.array([[1]]) #np.ones([num_rx, 1], int)
        RX_TX_ASSOCIATION = np.ones([self.num_rx, self.num_tx], int)  # [[1]]
        self.STREAM_MANAGEMENT = StreamManagement(
            RX_TX_ASSOCIATION, num_streams_per_tx
        )  # RX_TX_ASSOCIATION, NUM_STREAMS_PER_TX

        if guards:
            cyclic_prefix_length = 6  # 0 #6 Length of the cyclic prefix
            if num_guard_carriers is None and type(num_guard_carriers) is not list:
                num_guard_carriers = [
                    5,
                    6,
                ]  # [0, 0] #List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null = True  # False
            if (
                pilot_ofdm_symbol_indices is None
                and type(pilot_ofdm_symbol_indices) is not list
            ):
                pilot_ofdm_symbol_indices = [2, 11]
        else:
            cyclic_prefix_length = 0  # 0 #6 Length of the cyclic prefix
            num_guard_carriers = [
                0,
                0,
            ]  # List of two integers defining the number of guardcarriers at the left and right side of the resource grid.
            dc_null = False
            pilot_ofdm_symbol_indices = [0, 0]
        # pilot_pattern = "kronecker" #"kronecker", "empty"
        self.cyclic_prefix_length = cyclic_prefix_length
        self.num_guard_carriers = num_guard_carriers
        self.pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices
        # fft_size = 76
        # num_ofdm_symbols=14
        RESOURCE_GRID = MyResourceGrid(
            num_ofdm_symbols=num_ofdm_symbols,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,  # 60e3, #30e3,
            num_tx=self.num_tx,  # 1
            num_streams_per_tx=num_streams_per_tx,  # 1
            cyclic_prefix_length=cyclic_prefix_length,
            num_guard_carriers=num_guard_carriers,
            dc_null=dc_null,
            pilot_pattern=pilot_pattern,
            pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
        )
        if showfig:
            fig = RESOURCE_GRID.show()  # 14(OFDM symbol)*76(subcarrier) array=1064
            if outputpath is not None:
                figurename = os.path.join(outputpath, "RESOURCE_GRID" + IMG_FORMAT)
                fig.savefig(figurename)

            figs = RESOURCE_GRID.pilot_pattern.show()
            if outputpath is not None:
                for i, fig in enumerate(figs):
                    figurename = os.path.join(
                        outputpath, f"pilot_pattern_{i}" + IMG_FORMAT
                    )
                    fig.savefig(figurename)

            # The pilot patterns are defined over the resource grid of *effective subcarriers* from which the nulled DC and guard carriers have been removed.
            # This leaves us in our case with 76 - 1 (DC) - 5 (left guards) - 6 (right guards) = 64 effective subcarriers.

        if showfig and pilot_pattern == "kronecker":
            # actual pilot sequences for all streams which consists of random QPSK symbols.
            # By default, the pilot sequences are normalized, such that the average power per pilot symbol is
            # equal to one. As only every fourth pilot symbol in the sequence is used, their amplitude is scaled by a factor of two.
            plt.figure()
            plt.title("Real Part of the Pilot Sequences")
            for i in range(num_streams_per_tx):
                plt.stem(
                    np.real(RESOURCE_GRID.pilot_pattern.pilots[0, i]),
                    markerfmt="C{}.".format(i),
                    linefmt="C{}-".format(i),
                    label="Stream {}".format(i),
                )
            plt.legend()
            if outputpath is not None:
                figurename = os.path.join(outputpath, "PilotSeq" + IMG_FORMAT)
                plt.savefig(figurename)
            print(
                "Average energy per pilot symbol: {:1.2f}".format(
                    np.mean(np.abs(RESOURCE_GRID.pilot_pattern.pilots[0, 0]) ** 2)
                )
            )
        self.RESOURCE_GRID = RESOURCE_GRID
        print("RG num_ofdm_symbols", RESOURCE_GRID.num_ofdm_symbols)  # 14
        print(
            "RG ofdm_symbol_duration", RESOURCE_GRID.ofdm_symbol_duration
        )  # 1.7982456140350878e-05
        # from sionna.channel import subcarrier_frequencies
        self.frequencies = subcarrier_frequencies(
            RESOURCE_GRID.fft_size, RESOURCE_GRID.subcarrier_spacing
        )  # corresponding to the different subcarriers
        # 76, 60k

        # print("subcarriers frequencies", self.frequencies) #-2280000. -2220000. -2160000...2100000.  2160000.  2220000

        # num_bits_per_symbol = 4
        # Codeword length
        n = int(
            RESOURCE_GRID.num_data_symbols * num_bits_per_symbol
        )  # num_data_symbols: if empty 1064*4=4256, else, 768*4=3072
        self.n = n  # 1536

        # USE_LDPC = True
        if USE_LDPC:
            coderate = 0.5
            # Number of information bits per codeword
            k = int(n * coderate)
            encoder = LDPC5GEncoder(k, n)  # 1824, 3648
            decoder = LDPC5GDecoder(encoder, hard_out=True)
            self.decoder = decoder
            self.encoder = encoder
        else:
            coderate = 1
            # Number of information bits per codeword
            k = int(n * coderate)  # 1536
        self.k = k  # Number of information bits per codeword, 3072
        self.USE_LDPC = USE_LDPC
        self.coderate = coderate

        self.mapper = Mapper("qam", num_bits_per_symbol)
        self.rg_mapper = MyResourceGridMapper(
            RESOURCE_GRID
        )  # ResourceGridMapper(RESOURCE_GRID)

        # receiver part
        self.mydemapper = MyDemapper(
            "app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol
        )

        # Channel part
        if self.channeldataset == "deepmimo":
            self.create_DeepMIMOchanneldataset()  # get self.data_loader
        elif self.channeldataset == "cdl":
            self.create_CDLchanneldataset()  # get self.cdl
        # call get_channelcir to get channelcir

        if self.channeltype == "ofdm":
            # Function that will apply the channel frequency response to an input signal
            # channel_freq = ApplyOFDMChannel(add_awgn=True)
            # Generate the OFDM channel
            self.applychannel = MyApplyOFDMChannel(add_awgn=True)
            # self.applychannel = ApplyOFDMChannel(add_awgn=True)
            # h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            # (64, 1, 1, 1, 16, 1, 76)
        elif self.channeltype == "time":  # time channel:
            # channel_time = ApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
            bandwidth = self.RESOURCE_GRID.bandwidth  # 4560000
            l_min, l_max = time_lag_discrete_time_channel(bandwidth)  # -6, 20
            l_tot = l_max - l_min + 1  # 27
            self.l_tot = l_tot
            self.l_min = l_min
            self.l_max = l_max
            # self.applychannel = MyApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True)
            self.applychannel = MyApplyTimeChannel(
                self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True
            )
            # OFDM modulator and demodulator
            self.modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
            self.demodulator = OFDMDemodulator(
                self.RESOURCE_GRID.fft_size,
                l_min,
                self.RESOURCE_GRID.cyclic_prefix_length,
            )
        if self.pilot_pattern != "empty":
            self.remove_nulled_scs = RemoveNulledSubcarriers(self.RESOURCE_GRID)
            # ls_est = LSChannelEstimator(self.RESOURCE_GRID, interpolation_type="lin_time_avg")
            self.ls_est = MyLSChannelEstimator(
                self.RESOURCE_GRID, interpolation_type="nn"
            )  # "lin_time_avg")
            # self.ls_est = MyLSChannelEstimatorNP(self.RESOURCE_GRID, interpolation_type="nn")#"lin_time_avg")
            # lmmse_equ = LMMSEEqualizer(self.RESOURCE_GRID, self.STREAM_MANAGEMENT)
            self.lmmse_equ = MyLMMSEEqualizer(
                self.RESOURCE_GRID, self.STREAM_MANAGEMENT
            )

    def create_DeepMIMOchanneldataset(self):
        num_rx = self.num_rx
        # DeepMIMO provides multiple [scenarios](https://deepmimo.net/scenarios/) that one can select from.
        # In this example, we use the O1 scenario with the carrier frequency set to 60 GHz (O1_60).
        # Please download the "O1_60" data files [from this page](https://deepmimo.net/scenarios/o1-scenario/).
        # The downloaded zip file should be extracted into a folder, and the parameter `'dataset_folder` should be set to point to this folder
        if self.direction == "uplink":
            DeepMIMO_dataset = get_deepMIMOdata(
                scenario=self.scenario,
                dataset_folder=self.dataset_folder,
                num_ue_antenna=self.num_ut_ant,
                num_bs_antenna=self.num_bs_ant,
                showfig=self.showfig,
            )
            # DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder, num_ue_antenna=self.num_bs_ant, num_bs_antenna=self.num_ut_ant, showfig=self.showfig)
        else:
            DeepMIMO_dataset = get_deepMIMOdata(
                scenario=self.scenario,
                dataset_folder=dataset_folder,
                num_ue_antenna=self.num_bs_ant,
                num_bs_antenna=self.num_ut_ant,
                showfig=self.showfig,
            )
            # DeepMIMO_dataset = get_deepMIMOdata(scenario=scenario, dataset_folder=dataset_folder, num_ue_antenna=self.num_ut_ant, num_bs_antenna=self.num_bs_ant, showfig=self.showfig)
        DeepMIMO_dataset = get_deepMIMOdata(
            scenario=self.scenario,
            dataset_folder=self.dataset_folder,
            showfig=self.showfig,
        )
        # The number of UE locations in the generated DeepMIMO dataset
        num_ue_locations = len(DeepMIMO_dataset[0]["user"]["channel"])  # 18100
        # Pick the largest possible number of user locations that is a multiple of ``num_rx``
        ue_idx = np.arange(num_rx * (num_ue_locations // num_rx))  # (18100,) 0~18099
        # Optionally shuffle the dataset to not select only users that are near each others
        np.random.shuffle(ue_idx)
        # Reshape to fit the requested number of users
        ue_idx = np.reshape(
            ue_idx, [-1, num_rx]
        )  # In the shape of (floor(18100/num_rx) x num_rx) (18100,1)
        self.deepmimodataset = DeepMIMODataset(
            DeepMIMO_dataset=DeepMIMO_dataset,
            ue_idx=ue_idx,
            num_time_steps=self.num_time_steps,
        )
        h, tau = next(
            iter(self.deepmimodataset)
        )  # h: (1, 1, 1, 16, 10, 1), tau:(1, 1, 10)
        # complex gains `h` and delays `tau` for each path
        # print(h.shape) #[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(tau.shape) #[num_rx, num_tx, num_paths]

        # torch dataloaders
        self.data_loader = DataLoader(
            dataset=self.deepmimodataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        if self.showfig:
            # self.plotchimpulse()
            h_b, tau_b = next(
                iter(self.data_loader)
            )  # h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
            # print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            # print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
            tau_b = tau_b.numpy()  # torch tensor to numpy
            h_b = h_b.numpy()
            plt.figure()
            plt.title("Channel impulse response realization")
            plt.stem(
                tau_b[0, 0, 0, :] / 1e-9, np.abs(h_b)[0, 0, 0, 0, 0, :, 0]
            )  # 10 different pathes
            plt.xlabel(r"$\tau$ [ns]")
            plt.ylabel(r"$|a|$")
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "Channel_impulse" + IMG_FORMAT
                )
                plt.savefig(figurename)

    def create_CDLchanneldataset(self):
        # try:
        #     #from sionna.channel.tr38901 import AntennaArray, CDL
        #     from sionna_tf_cdl import AntennaArray, CDL
        # except ImportError:
        #     pass
        from sionna_tf_cdl import AntennaArray, CDL

        # Define the number of UT and BS antennas.
        # For the CDL model, a single UT and BS are supported.
        # The CDL model only works for systems with a single transmitter and a single receiver. The transmitter and receiver can be equipped with multiple antennas.
        num_ut_ant = self.num_ut_ant  # 2
        num_bs_ant = self.num_bs_ant  # 16 #8

        carrier_frequency = 2.6e9  # Carrier frequency in Hz.
        # This is needed here to define the antenna element spacing.

        ut_array = AntennaArray(
            num_rows=1,
            num_cols=int(num_ut_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        if self.showfig:
            ut_array.show()
            ut_array.show_element_radiation_pattern()

        bs_array = AntennaArray(
            num_rows=1,
            num_cols=int(num_bs_ant / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=carrier_frequency,
        )
        if self.showfig:
            bs_array.show()
            bs_array.show_element_radiation_pattern()

        # CDL channel model
        delay_spread = (
            300e-9  # Nominal delay spread in [s]. Please see the CDL documentation
        )
        # about how to choose this value.

        direction = (
            "uplink"  # The `direction` determines if the UT or BS is transmitting.
        )
        # In the `uplink`, the UT is transmitting.
        cdl_model = "B"  # Suitable values are ["A", "B", "C", "D", "E"]

        speed = 10  # UT speed [m/s]. BSs are always assumed to be fixed.
        # The direction of travel will chosen randomly within the x-y plane.

        # Configure a channel impulse reponse (CIR) generator for the CDL model.
        # cdl() will generate CIRs that can be converted to discrete time or discrete frequency.
        self.cdl = CDL(
            cdl_model,
            delay_spread,
            carrier_frequency,
            ut_array,
            bs_array,
            direction,
            min_speed=speed,
        )
        # The cdl can be used to generate batches of random realizations of continuous-time
        # channel impulse responses, consisting of complex gains `a` and delays `tau` for each path.
        # To account for time-varying channels, a channel impulse responses is sampled at the `sampling_frequency` for `num_time_samples` samples.

    def get_channelcir(self, returnformat="numpy"):
        if self.channeldataset == "deepmimo":
            # https://github.com/DeepMIMO/DeepMIMO-python/blob/master/src/DeepMIMOv3/sionna_adapter.py
            h_b, tau_b = next(
                iter(self.data_loader)
            )  # h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
            num_time_steps = self.deepmimodataset.num_time_steps  # 1
            sampling_frequency = 1 / self.RESOURCE_GRID.ofdm_symbol_duration
        elif self.channeldataset == "cdl":
            if self.channeltype == "ofdm":
                num_time_steps = self.RESOURCE_GRID.num_ofdm_symbols
                sampling_frequency = 1 / self.RESOURCE_GRID.ofdm_symbol_duration
            elif self.channeltype == "time":
                num_time_steps = self.RESOURCE_GRID.num_time_samples + self.l_tot - 1
                sampling_frequency = self.RESOURCE_GRID.bandwidth
            h_b, tau_b = self.cdl(
                batch_size=self.batch_size,
                num_time_steps=num_time_steps,
                sampling_frequency=sampling_frequency,
            )
        self.num_time_steps = num_time_steps
        self.sampling_frequency = sampling_frequency
        # In CDL, Direction = "uplink" the UT is transmitting.
        # num_bs_ant = 16 = num_rx_ant
        # num_ut_ant = 2 = num_tx_ant
        # print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps] (64, 1, 16, 1, 2, 23, 14)
        # print(tau_b.shape) #[batch, num_rx, num_tx, num_paths] (64, 1, 1, 23)
        if returnformat == "numpy":
            h_b = to_numpy(h_b)
            tau_b = to_numpy(tau_b)

        if self.showfig:
            plt.figure()
            plt.title("Channel impulse response realization")
            plt.stem(
                tau_b[0, 0, 0, :] / 1e-9, np.abs(h_b)[0, 0, 0, 0, 0, :, 0]
            )  # 10 different pathes
            plt.xlabel(r"$\tau$ [ns]")
            plt.ylabel(r"$|a|$")
            if self.outputpath is not None:
                figurename = os.path.join(self.outputpath, "Channel_cir" + IMG_FORMAT)
                plt.savefig(figurename)

            plt.figure()
            plt.title("Time evolution of path gain")
            # x_timesteps = np.arange(num_time_steps)*self.RESOURCE_GRID.ofdm_symbol_duration/1e-6
            x_timesteps = np.arange(num_time_steps) / sampling_frequency / 1e-6
            plt.plot(x_timesteps, np.real(h_b)[0, 0, 0, 0, 0, 0, :])
            plt.plot(x_timesteps, np.imag(h_b)[0, 0, 0, 0, 0, 0, :])
            plt.legend(["Real part", "Imaginary part"])
            plt.xlabel(r"$t$ [us]")
            plt.ylabel(r"$a$")
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "Time_evolution_path" + IMG_FORMAT
                )
                plt.savefig(figurename)

        return h_b, tau_b

    # def get_htau_batch(self, returnformat='numpy'):
    #     h_b, tau_b = next(iter(self.data_loader)) #h_b: [64, 1, 1, 1, 16, 10, 1], tau_b=[64, 1, 1, 10]
    #     #print(h_b.shape) #[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    #     #print(tau_b.shape) #[batch, num_rx, num_tx, num_paths]
    #     if returnformat=="numpy":
    #         tau_b=tau_b.numpy()#torch tensor to numpy
    #         h_b=h_b.numpy()
    #     return h_b, tau_b

    def get_OFDMchannelresponse(self, h_b, tau_b):
        # h_b, tau_b = self.get_channelcir()
        # from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
        # from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
        # from sionna.channel import cir_to_ofdm_channel
        # frequencies = subcarrier_frequencies(self.RESOURCE_GRID.fft_size, self.RESOURCE_GRID.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(self.frequencies, h_b, tau_b, normalize=True)
        # h_freq.shape #[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers] (64, 1, 16, 1, 2, 14, 76)
        # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size](2, 1, 1, 1, 16, 1, 76)
        if self.showfig:
            plt.figure()
            plt.title("Channel frequency response at given time")
            plt.plot(np.real(h_freq[0, 0, 0, 0, 0, 0, :]))
            plt.plot(np.imag(h_freq[0, 0, 0, 0, 0, 0, :]))
            plt.xlabel("OFDM Symbol Index")
            plt.ylabel(r"$h$")
            plt.legend(["Real part", "Imaginary part"])
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "OFDMchannelresponse" + IMG_FORMAT
                )
                plt.savefig(figurename)
        return h_freq

    def get_timechannelresponse(self, h_b, tau_b):
        # h_b, tau_b = self.get_channelcir()
        # h_time = cir_to_time_channel(rg.bandwidth, a, tau, l_min=l_min, l_max=l_max, normalize=True)
        # [2, 1, 16, 1, 2, 1164, 17]
        # from sionna.channel import cir_to_time_channel
        h_time = cir_to_time_channel(
            self.RESOURCE_GRID.bandwidth,
            h_b,
            tau_b,
            l_min=self.l_min,
            l_max=self.l_max,
            normalize=True,
        )
        # h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        if self.showfig:
            plt.figure()
            plt.title("Discrete-time channel impulse response")
            plt.stem(np.abs(h_time[0, 0, 0, 0, 0, 0]))
            plt.xlabel(r"Time step $\ell$")
            plt.ylabel(r"$|\bar{h}|$")
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "Discrete_time_channel" + IMG_FORMAT
                )
                plt.savefig(figurename)
        return h_time

    def generateChannel(self, x_rg, no, channeltype="ofdm"):
        # x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]

        # h_b, tau_b = self.get_htau_batch()
        h_b, tau_b = self.get_channelcir()
        h_out = None
        # print(h_b.shape) #complex (64, 1, 1, 1, 16, 10, 1)[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # print(tau_b.shape) #float (64, 1, 1, 10)[batch, num_rx, num_tx, num_paths]
        if channeltype == "ofdm":  # Generate the OFDM channel response
            # computes the Fourier transform of the continuous-time channel impulse response at a set of `frequencies`, corresponding to the different subcarriers.
            ##h: [64, 1, 1, 1, 16, 10, 1], tau: [64, 1, 1, 10] => (64, 1, 1, 1, 16, 1, 76)
            h_freq = mygenerate_OFDMchannel(
                h_b,
                tau_b,
                self.fft_size,
                subcarrier_spacing=60000.0,
                dtype=np.complex64,
                normalize_channel=True,
            )
            # h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            # [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
            # (64, 1, 1, 1, 16, 1, 76)

            remove_nulled_scs = RemoveNulledSubcarriers(self.RESOURCE_GRID)
            h_out = remove_nulled_scs(h_freq)  # (64, 1, 1, 1, 16, 1, 64)
            if self.showfig:
                h_freq_plt = h_out[
                    0, 0, 0, 0, 0, 0
                ]  # get the last dimension: fft_size [76]
                # h_freq_plt = h_freq[0,0,0,0,0,0] #get the last dimension: fft_size [76]
                plt.figure()
                plt.plot(np.real(h_freq_plt))
                plt.plot(np.imag(h_freq_plt))
                plt.xlabel("Subcarrier index")
                plt.ylabel("Channel frequency response")
                plt.legend(["Ideal (real part)", "Ideal (imaginary part)"])
                plt.title("Comparison of channel frequency responses")
                if self.outputpath is not None:
                    figurename = os.path.join(
                        self.outputpath, "channel_compare" + IMG_FORMAT
                    )
                    plt.savefig(figurename)

            # Generate the OFDM channel
            channel_freq = MyApplyOFDMChannel(add_awgn=True)
            # h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
            # (64, 1, 1, 1, 16, 1, 76)
            y = channel_freq([x_rg, h_freq, no])  # h_freq is array
            # Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
            # print(y.shape) #[64, 1, 1, 14, 76] dim (3,4 removed)

            # y = ApplyOFDMChannel(symbol_resourcegrid=x_rg, channel_frequency=h_freq, noiselevel=no, add_awgn=True)
            # y is the symbol received after the channel and noise
            # Channel outputs y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], complex
        elif channeltype == "perfect":
            y = x_rg
        elif channeltype == "awgn":
            y = x_rg  # (64, 1, 1, 14, 76)
            noise = complex_normal(y.shape, var=1.0)
            print(noise.dtype)
            noise = noise.astype(y.dtype)
            noise *= np.sqrt(no)
            y = y + noise
        elif channeltype == "time":
            bandwidth = self.RESOURCE_GRID.bandwidth  # 4560000
            l_min, l_max = time_lag_discrete_time_channel(bandwidth)  # -6, 20
            l_tot = l_max - l_min + 1  # 27
            # Compute the discrete-time channel impulse reponse
            h_time = cir_to_time_channel(
                bandwidth, h_b, tau_b, l_min=l_min, l_max=l_max, normalize=True
            )
            # h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1] complex[64, 1, 1, 1, 16, 1, 27]
            h_out = h_time
            if self.showfig:
                plt.figure()
                plt.title("Discrete-time channel impulse response")
                plt.stem(np.abs(h_time[0, 0, 0, 0, 0, 0]))
                plt.xlabel(r"Time step $\ell$")
                plt.ylabel(r"$|\bar{h}|$")
                if self.outputpath is not None:
                    figurename = os.path.join(
                        self.outputpath, "Discrete_time_channel2" + IMG_FORMAT
                    )
                    plt.savefig(figurename)
            # channel_time = ApplyTimeChannel(self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=False)
            channel_time = MyApplyTimeChannel(
                self.RESOURCE_GRID.num_time_samples, l_tot=l_tot, add_awgn=True
            )
            # OFDM modulator and demodulator
            modulator = OFDMModulator(self.RESOURCE_GRID.cyclic_prefix_length)
            demodulator = OFDMDemodulator(
                self.RESOURCE_GRID.fft_size,
                l_min,
                self.RESOURCE_GRID.cyclic_prefix_length,
            )

            # OFDM modulation with cyclic prefix insertion
            # x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]
            x_time = modulator(x_rg)  # output: (64, 1, 1, 1064)
            # Compute the channel output
            # This computes the full convolution between the time-varying
            # discrete-time channel impulse reponse and the discrete-time
            # transmit signal. With this technique, the effects of an
            # insufficiently long cyclic prefix will become visible. This
            # is in contrast to frequency-domain modeling which imposes
            # no inter-symbol interfernce.
            y_time = channel_time([x_time, h_time, no])  # [64, 1, 1, 1174]
            # y_time = channel_time([x_time, h_time]) #(64, 1, 1, 1090) complex

            # Do modulator and demodulator test
            y_test = demodulator(x_time)
            differences = np.abs(x_rg - y_test)
            threshold = 1e-7
            num_differences = np.sum(differences > threshold)
            print("Number of differences:", num_differences)
            print(np.allclose(x_rg, y_test))
            print("Demodulation error (L2 norm):", np.linalg.norm(x_rg - y_test))

            # OFDM demodulation and cyclic prefix removal
            y = demodulator(y_time)
            # y = y_test
            # y: [64, 1, 1, 14, 76]
        return y, h_out

    def uplinktransmission(self, b=None, h_out=None, no=0):

        if b is None:
            binary_source = BinarySource()
            # Start Transmitter self.k Number of information bits per codeword
            b = binary_source(
                [self.batch_size, 1, self.num_streams_per_tx, self.k]
            )  # [64,1,2,4256] if empty [64,1,1,1536] [batch_size, num_tx, num_streams_per_tx, num_databits]
        if self.USE_LDPC:
            c = self.encoder(
                b
            )  # tf.tensor[64,1,1,3072] [batch_size, num_tx, num_streams_per_tx, num_codewords]
        else:
            c = b
        x = self.mapper(
            c
        )  # np.array[64,1,1,896] if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x_rg = self.rg_mapper(x)  ##complex array[64,1,1,14,76] 14*76=1064
        # output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76] (64, 1, 2, 14, 76)

        # apply channel
        if self.channeltype == "ofdm":
            y = self.applychannel([x_rg, h_out, no])
        else:  # time channel
            # OFDM modulation with cyclic prefix insertion
            # x_rg:[batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size][64,1,1,14,76]
            x_time = self.modulator(x_rg)  # output: (64, 1, 1, 1064)
            # Compute the channel output
            # This computes the full convolution between the time-varying
            # discrete-time channel impulse reponse and the discrete-time
            # transmit signal. With this technique, the effects of an
            # insufficiently long cyclic prefix will become visible. This
            # is in contrast to frequency-domain modeling which imposes
            # no inter-symbol interfernce.

            # x_time = tf.convert_to_tensor(x_time, dtype=tf.complex64) #(2, 1, 2, 1148)
            # h_out = to_numpy(h_out) h_out is tf tensor
            y_time = self.applychannel([x_time, h_out, no])  # (2, 1, 16, 1174)
            # #Do modulator and demodulator test
            # y_test = self.demodulator(x_time)
            # differences = np.abs(x_rg - y_test)
            # threshold=1e-7
            # num_differences = np.sum(differences > threshold)
            # print("Number of differences:", num_differences)
            # print(np.allclose(x_rg, y_test))
            # print("Demodulation error (L2 norm):", np.linalg.norm(x_rg - y_test))

            print("y_time shape:", y_time.shape)  # (64, 1, 16, 1174)
            # y_time = tf.convert_to_tensor(y_time, dtype=tf.complex64)
            # OFDM demodulation and cyclic prefix removal
            y = self.demodulator(y_time)  # y: (2, 1, 16, 14, 76)
        # x :  Channel inputs [batch size, num_tx, num_tx_ant, num_time_samples], tf.complex
        # h_time : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], tf.complex

        return y, x_rg, x, b

    def ofdmchannel_estimation(self, y, no, h_out=None, perfect_csi=False):
        # perform channel estimation via pilots
        print("Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots))  # 1
        h_perfect = None

        if perfect_csi or self.savedata:
            # For perfect CSI, the receiver gets the channel frequency response as input
            # However, the channel estimator only computes estimates on the non-nulled
            # subcarriers. Therefore, we need to remove them here from `h_freq`.
            # This step can be skipped if no subcarriers are nulled.
            h_perfect, err_var_perfect = self.remove_nulled_scs(h_out), 0.0
            print("h_out after remove nulled shape:", h_perfect.shape)
        else:
            h_perfect = None
            err_var_perfect = None

        if (not perfect_csi) or self.showfig or self.savedata:
            # Observed resource grid y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols,fft_size], (64, 1, 1, 14, 76) complex
            # no : [batch_size, num_rx, num_rx_ant]
            h_hat, err_var = self.ls_est(
                [y, no]
            )  # tf tensor (64, 1, 16, 1, 2, 14, 64), (1, 1, 1, 1, 2, 14, 64)
            # h_ls : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,fft_size], tf.complex
            # Channel estimates accross the entire resource grid for all transmitters and streams

            if self.showfig:
                h_est = h_hat[0, 0, 0, 0, 0, 0]  # (64, 1, 1, 1, 1, 14, 44)
                plt.figure()
                if h_out is not None:
                    h_perfect, err_var = self.remove_nulled_scs(h_out), 0.0
                    h_perfect = h_perfect[0, 0, 0, 0, 0, 0]  # (64, 1, 1, 1, 16, 1, 44)
                    plt.plot(np.real(h_perfect))
                    plt.plot(np.imag(h_perfect))
                plt.plot(np.real(h_est), "--")
                plt.plot(np.imag(h_est), "--")
                plt.xlabel("Subcarrier index")
                plt.ylabel("Channel frequency response")
                plt.legend(
                    [
                        "Ideal (real part)",
                        "Ideal (imaginary part)",
                        "Estimated (real part)",
                        "Estimated (imaginary part)",
                    ]
                )
                plt.title("Comparison of channel frequency responses")
                if self.outputpath is not None:
                    figurename = os.path.join(
                        self.outputpath, "OFDMChannel_compare" + IMG_FORMAT
                    )
                    plt.savefig(figurename)

        # if perfect_csi:
        #     return h_perfect, err_var_perfect
        # else:
        #     return h_hat, err_var
        return h_hat, err_var, h_perfect, err_var_perfect

    def test_timechannel_estimation(self, b=None, no=0):
        # ebno_db = 30
        # from sionna.utils import ebnodb2no #BinarySource, sim_ber
        # no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)

        # The CIR needs to be sampled every 1/bandwith [s].
        # In contrast to frequency-domain modeling, this implies
        # that the channel can change over the duration of a single
        # OFDM symbol. We now also need to simulate more
        # time steps.
        # from sionna.channel import cir_to_ofdm_channel, cir_to_time_channel, ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
        num_streams_per_tx = 2
        # from sionna.ofdm import ResourceGrid#, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
        # rg = MyResourceGrid(num_ofdm_symbols=14,
        #           fft_size=76,
        #           subcarrier_spacing=60e3, #15e3,
        #           num_tx=1,
        #           num_streams_per_tx=num_streams_per_tx,
        #           cyclic_prefix_length=6,
        #           num_guard_carriers=[5,6],
        #           dc_null=True,
        #           pilot_pattern="kronecker",
        #           pilot_ofdm_symbol_indices=[2,11])
        rg = self.RESOURCE_GRID

        option1 = True
        if option1:
            a, tau = self.cdl(
                batch_size=2,
                num_time_steps=rg.num_time_samples + self.l_tot - 1,
                sampling_frequency=rg.bandwidth,
            )
            # a, tau = cdl(batch_size=32, num_time_steps=self.RESOURCE_GRID.num_ofdm_symbols, sampling_frequency=1/self.RESOURCE_GRID.ofdm_symbol_duration)
            a = to_numpy(a)  # (2, 1, 16, 1, 2, 23, 1174)
            tau = to_numpy(tau)  # (2, 1, 1, 23)
            h_time = cir_to_time_channel(
                rg.bandwidth, a, tau, l_min=self.l_min, l_max=self.l_max, normalize=True
            )  # (2, 1, 16, 1, 2, 1174, 27)
        else:
            cir = self.cdl(
                self.batch_size, rg.num_time_samples + self.l_tot - 1, rg.bandwidth
            )
            a, tau = cir
            a = to_numpy(a)
            tau = to_numpy(tau)
            print("a shape:", a.shape)  # (64, 1, 16, 1, 2, 23, 1174)
            print("tau shape:", tau.shape)  # (64, 1, 1, 23)
            # Compute the discrete-time channel impulse reponse
            h_time = cir_to_time_channel(
                rg.bandwidth, *cir, self.l_min, self.l_max, normalize=True
            )
            print("h_time shape:", h_time.shape)  # (64, 1, 16, 1, 2, 1174, 27)

        # Function that will apply the discrete-time channel impulse response to an input signal
        channel_time = MyApplyTimeChannel(
            rg.num_time_samples, l_tot=self.l_tot, add_awgn=True
        )

        # we generate random batches of CIR, transform them in the frequency domain and apply them to the resource grid in the frequency domain.
        # h_b, tau_b = self.get_channelcir()
        # if self.channeltype=='ofdm':
        #     h_out = self.get_OFDMchannelresponse(h_b, tau_b)
        #     print("h_freq shape:", h_out.shape) #(64, 1, 16, 1, 2, 14, 76)
        # elif self.channeltype=='time':
        #     h_out = self.get_timechannelresponse(h_b, tau_b) #(64, 1, 16, 1, 2, 1174, 27)

        # h_time = cir_to_time_channel(self.RESOURCE_GRID.bandwidth, *cir, self.l_min, self.l_max, normalize=True)
        # h_out = cir_to_time_channel(self.RESOURCE_GRID.bandwidth, h_b, tau_b, l_min=self.l_min, l_max=self.l_max, normalize=True)
        # h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1]
        if self.showfig:
            plt.figure()
            plt.title("Discrete-time channel impulse response")
            plt.stem(np.abs(h_time[0, 0, 0, 0, 0, 0]))
            plt.xlabel(r"Time step $\ell$")
            plt.ylabel(r"$|\bar{h}|$")
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "Discrete_time_channel2" + IMG_FORMAT
                )
                plt.savefig(figurename)

                # OFDM modulator and demodulator
        # from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
        # from deepMIMO5 import OFDMModulator, OFDMDemodulator
        # from sionna.mapping import Mapper, Demapper
        # from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
        modulator = OFDMModulator(rg.cyclic_prefix_length)
        demodulator = OFDMDemodulator(rg.fft_size, self.l_min, rg.cyclic_prefix_length)

        # The mapper maps blocks of information bits to constellation symbols
        mapper = self.mapper  # Mapper("qam", self.num_bits_per_symbol)

        # The resource grid mapper maps symbols onto an OFDM resource grid
        rg_mapper = self.rg_mapper  # ResourceGridMapper(rg)

        if b is None:
            binary_source = BinarySource()
            # Start Transmitter self.k Number of information bits per codeword
            b = binary_source(
                [self.batch_size, 1, self.num_streams_per_tx, self.k]
            )  # [64,1,2,4256] if empty [64,1,1,1536] [batch_size, num_tx, num_streams_per_tx, num_databits]
        if self.USE_LDPC:
            c = self.encoder(
                b
            )  # tf.tensor[64,1,1,3072] [batch_size, num_tx, num_streams_per_tx, num_codewords]
        else:
            c = b
        x = mapper(
            c
        )  # np.array[64,1,1,896] if empty np.array[64,1,1,1064] 1064*4=4256 [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x_rg = rg_mapper(x)  ##complex array(2, 1, 2, 14, 76) 14*76=1064
        # output: [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size](2, 1, 2, 14, 76)

        # OFDM modulation with cyclic prefix insertion
        x_time = modulator(x_rg)
        print("x_time shape:", x_time.shape)  # (2, 1, 2, 1148)

        # time_channel = TimeChannel(self.cdl, self.RESOURCE_GRID.bandwidth, self.RESOURCE_GRID.num_time_samples,
        #                    l_min=self.l_min, l_max=self.l_max, normalize_channel=True,
        #                    add_awgn=True, return_channel=True)

        # y_time1, h_time = time_channel([x_time, no])
        # print("y_time1 shape:", y_time1.shape)
        # no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, rg)
        # x_time = tf.convert_to_tensor(x_time, dtype=tf.complex64) #(2, 1, 2, 1148)
        # y_time = self.applychannel([x_time, h_time, no])
        # h_time = tf.convert_to_tensor(h_time, dtype=tf.complex64)
        y_time = channel_time([x_time, h_time, no])
        # y_time = to_numpy(y_time)
        print("y_time shape:", y_time.shape)  # (2, 1, 16, 1174)

        # time_channel = TimeChannel(cdl, rg.bandwidth, rg.num_time_samples,
        #                    l_min=l_min, l_max=l_max, normalize_channel=True,
        #                    add_awgn=True, return_channel=True)

        # y_time, h_time = time_channel([x_time, no])

        # y_test = demodulator(x_time)
        # differences = np.abs(x_rg - y_test.numpy()) #(2, 1, 2, 14, 76)
        # threshold=1e-7
        # num_differences = np.sum(differences > threshold)
        # print("Number of differences:", num_differences)
        # print(np.allclose(x_rg, y_test))
        # print("Demodulation error (L2 norm):", np.linalg.norm(x_rg - y_test))

        # OFDM demodulation and cyclic prefix removal
        y = demodulator(y_time)
        print("y shape:", y.shape)  # (2, 1, 16, 14, 76)

        # We need to sub-sample the channel impulse reponse to compute perfect CSI
        # for the receiver as it only needs one channel realization per OFDM symbol
        a_freq = a[
            ..., rg.cyclic_prefix_length : -1 : (rg.fft_size + rg.cyclic_prefix_length)
        ]
        a_freq = a_freq[..., : rg.num_ofdm_symbols]
        print("a_freq shape:", a_freq.shape)  # (64, 1, 16, 1, 2, 23, 14)

        # Compute the channel frequency response
        h_freq = cir_to_ofdm_channel(self.frequencies, a_freq, tau, normalize=True)
        print("h_freq shape:", h_freq.shape)  # (2, 1, 16, 1, 2, 14, 76)
        h_hat, err_var = self.remove_nulled_scs(h_freq), 0.0
        print("h_hat shape:", h_hat.shape)  # (2, 1, 16, 1, 2, 14, 64)

        h_perf = h_hat[0, 0, 0, 0, 0, 0]

        # from sionna.ofdm import LSChannelEstimator #ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
        # The LS channel estimator will provide channel estimates and error variances
        ls_est = MyLSChannelEstimator(
            rg, interpolation_type="nn"
        )  # LSChannelEstimator(rg, interpolation_type="nn")
        ls_est2 = MyLSChannelEstimatorNP(
            self.RESOURCE_GRID, interpolation_type="nn"
        )  # "lin_time_avg")
        # ls_est2 = MyLSChannelEstimator(self.RESOURCE_GRID, interpolation_type="lin_time_avg")

        # We now compute the LS channel estimate from the pilots.
        # print(y[0,0,0,0,:])
        y = tf.convert_to_tensor(y, dtype=tf.complex64)

        # shape_a = (2, 1, 16, 1, 2, 128)
        # # Create an array 'no' with the same shape as 'a'
        # v = 1+1j  # Replace with your desired float value
        # y_pilots = np.full(shape_a, v, dtype=np.complex64)
        # h_hat, err_var = ls_est.estimate_at_pilot_locations(y_pilots, no) #y_pilots: (2, 1, 16, 1, 2, 128), h_hat:(2, 1, 16, 1, 2, 128)
        # h_hat2, err_var = ls_est2.estimate_at_pilot_locations(y_pilots, no) #y_pilots: (2, 1, 16, 1, 2, 128), h_hat:(2, 1, 16, 1, 2, 128)
        # plt.figure()
        # plt.plot(np.real(h_hat[0,0,0,0,0]))
        # plt.plot(np.imag(h_hat[0,0,0,0,0]))
        # plt.plot(np.real(h_hat2[0,0,0,0,0]), "--")
        # plt.plot(np.imag(h_hat2[0,0,0,0,0]), "--")

        h_est, _ = ls_est(
            [y, no]
        )  # (64, 1, 16, 1, 2, 14, 64) [batch_size, num_rx, num_rx_ant, num_ofdm_symbols=14, sub-carriers=64]
        h_est2, _ = ls_est2([y, no])  # (64, 1, 16, 1, 2, 14, 64)
        # h_hat:(2, 1, 16, 1, 2, 128)
        print(np.allclose(h_est.numpy(), h_est2))

        # h_est = h_est[0,0,0,0,0,:]
        # h_est2 = h_est2[0,0,0,0,0,:]

        h_est = h_est[0, 0, 0, 0, 0, 0]
        h_est2 = h_est2[0, 0, 0, 0, 0, 0]
        # print(h_est)
        # print(h_est2)

        plt.figure()
        plt.plot(np.real(h_perf))
        plt.plot(np.imag(h_perf))
        plt.plot(np.real(h_est), "--")
        plt.plot(np.imag(h_est), "--")
        plt.plot(np.real(h_est2), "r")
        plt.plot(np.imag(h_est2), "g")
        plt.xlabel("Subcarrier index")
        plt.ylabel("Channel frequency response")
        plt.legend(
            [
                "Ideal (real part)",
                "Ideal (imaginary part)",
                "Estimated (real part)",
                "Estimated (imaginary part)",
            ]
        )
        plt.title("Comparison of channel frequency responses")

    def timechannel_estimation(self, y, no, a=None, tau=None, perfect_csi=False):

        if perfect_csi or self.showfig or self.savedata:
            print("a shape:", a.shape)  # (64, 1, 16, 1, 2, 23, 1174)
            print("tau shape:", tau.shape)  # (64, 1, 1, 23)

            # We need to sub-sample the channel impulse reponse to compute perfect CSI
            # for the receiver as it only needs one channel realization per OFDM symbol
            a_freq = a[
                ...,
                self.RESOURCE_GRID.cyclic_prefix_length : -1 : (
                    self.RESOURCE_GRID.fft_size
                    + self.RESOURCE_GRID.cyclic_prefix_length
                ),
            ]  # (64, 1, 16, 1, 2, 23, 15)
            a_freq = a_freq[
                ..., : self.RESOURCE_GRID.num_ofdm_symbols
            ]  # (64, 1, 16, 1, 2, 23, 14)
            print("a_freq shape:", a_freq.shape)

            # Compute the channel frequency response
            h_freq = cir_to_ofdm_channel(self.frequencies, a_freq, tau, normalize=True)
            print("h_freq shape:", h_freq.shape)  # (64, 1, 16, 1, 2, 14, 76)
            h_perfect, err_var_perfect = self.remove_nulled_scs(h_freq), 0.0
            print("h_perfect shape:", h_perfect.shape)  # (64, 1, 16, 1, 2, 14, 64)
        else:
            h_perfect = None
            err_var_perfect = None

        if (not perfect_csi) or self.showfig or self.savedata:
            # ls_est = LSChannelEstimator(self.RESOURCE_GRID, interpolation_type="nn")
            # ls_est = MyLSChannelEstimatorNP(self.RESOURCE_GRID, interpolation_type="nn")
            # ls_est2 = MyLSChannelEstimator(self.RESOURCE_GRID, interpolation_type="nn") #"lin_time_avg"

            h_hat, err_var = self.ls_est([y, no])  # (2, 1, 16, 1, 2, 14, 64)
            # h_hat2, err_var2 = ls_est2([y, no])#tf.tensor
            # np.save('h_hat2.npy', h_hat)
            # np.save('h_hat_tf.npy', h_hat2.numpy())
            # print("h_hat shape:", h_hat.shape) #(64, 1, 16, 1, 2, 14, 64)
            # print("err_var shape:", err_var.shape) #(1, 1, 1, 1, 2, 14, 64)

        if self.showfig:
            # we assumed perfect CSI, i.e., h_perfect correpsond to the exact ideal channel frequency response.
            h_perf = h_perfect[0, 0, 0, 0, 0, 0]
            h_est = h_hat[0, 0, 0, 0, 0, 0]
            # h_est2 = h_hat2[0,0,0,0,0,0]
            plt.figure()
            plt.plot(np.real(h_perf))
            plt.plot(np.imag(h_perf))
            plt.plot(np.real(h_est), "--")
            plt.plot(np.imag(h_est), "--")
            # plt.plot(np.real(h_est2), "r")
            # plt.plot(np.imag(h_est2), "b")
            plt.xlabel("Subcarrier index")
            plt.ylabel("Channel frequency response")
            plt.legend(
                [
                    "Ideal (real part)",
                    "Ideal (imaginary part)",
                    "Estimated (real part)",
                    "Estimated (imaginary part)",
                ]
            )
            plt.title("Comparison of channel frequency responses")
            if self.outputpath is not None:
                figurename = os.path.join(
                    self.outputpath, "TimeChannel_compare" + IMG_FORMAT
                )
                plt.savefig(figurename)

        # if perfect_csi:
        #     return h_perfect, err_var_perfect
        # else:
        #     return h_hat, err_var
        return h_hat, err_var, h_perfect, err_var_perfect

    def channelest_equ(
        self, y, no, h_b=None, tau_b=None, h_out=None, perfect_csi=False
    ):
        print(
            self.RESOURCE_GRID.pilot_pattern
        )  # <__main__.EmptyPilotPattern object at 0x7f2659dfd9c0>
        print("Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots))  # 1
        h_hat = None
        err_var = None
        h_perfect = None
        err_var_perfect = None
        if (
            self.pilot_pattern == "empty"
        ):  # and perfect_csi == False: #"kronecker", "empty"
            # no channel estimation
            x_hat = y[:, :, 0 : self.num_streams_per_tx, :, :]  # (64, 1, 2, 14, 76)
            no_eff = no
        else:  # channel estimation or perfect_csi
            # perform channel estimation via pilots
            y = tf.convert_to_tensor(y, dtype=tf.complex64)
            if self.channeltype == "ofdm":
                h_hat, err_var, h_perfect, err_var_perfect = (
                    self.ofdmchannel_estimation(
                        y=y, no=no, h_out=h_out, perfect_csi=perfect_csi
                    )
                )
            else:  # time channel
                h_hat, err_var, h_perfect, err_var_perfect = (
                    self.timechannel_estimation(
                        y=y, no=no, a=h_b, tau=tau_b, perfect_csi=perfect_csi
                    )
                )
                h_hat = tf.convert_to_tensor(
                    h_hat, dtype=tf.complex64
                )  # (2, 1, 16, 1, 2, 14, 64)
                # err_var=tf.convert_to_tensor(err_var, dtype=tf.complex64) #(1, 1, 1, 1, 2, 14, 64)

            # h_hat shape: (64, 1, 16, 1, 2, 14, 64), err_var: (1, 1, 1, 1, 2, 14, 64)
            # input (y, h_hat, err_var, no)
            # Received OFDM resource grid after cyclic prefix removal and FFT y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            # Channel estimates for all streams from all transmitters h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
            x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
            # Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
            # Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
            x_hat = x_hat.numpy()  # x_hat: (2, 1, 2, 768), no_eff: (2, 1, 2, 768)
            no_eff = no_eff.numpy()
            no_eff = np.mean(no_eff)

        return x_hat, no_eff, h_hat, err_var, h_perfect, err_var_perfect

    def demapper_decision(self, x_hat, no_eff):
        if (
            self.pilot_pattern == "empty"
        ):  # and perfect_csi == False: #"kronecker", "empty"
            # no channel estimation
            llr = self.mydemapper(
                [x_hat, no_eff]
            )  # [64, 1, 16, 14, 76]=>(64, 1, 16, 14, 304)
            # Reshape the array by collapsing the last two dimensions
            llr_est = llr.reshape(llr.shape[:-2] + (-1,))  # (64, 1, 16, 4256)
        else:
            llr_est = self.mydemapper(
                [x_hat, no_eff]
            )  # (128, 1, 2, 1536) #(2, 1, 2, 3072)
            # output: [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]

        # llr_est #(64, 1, 1, 4256)
        if self.USE_LDPC:
            b_hat_tf = self.decoder(llr_est)  # [64, 1, 1, 2128]
            b_hat = b_hat_tf.numpy()
        else:
            b_hat = hard_decisions(
                llr_est, np.int32
            )  # (128, 1, 2, 1536) #(2, 1, 2, 3072)
        return b_hat, llr_est

    def receiver(
        self, y, no, x_rg, b=None, h_b=None, tau_b=None, h_out=None, perfect_csi=False
    ):
        print(
            self.RESOURCE_GRID.pilot_pattern
        )  # <__main__.EmptyPilotPattern object at 0x7f2659dfd9c0>
        print("Num of Pilots:", len(self.RESOURCE_GRID.pilot_pattern.pilots))
        if (
            self.pilot_pattern == "empty"
        ):  # and perfect_csi == False: #"kronecker", "empty"
            # no channel estimation
            x_hat = y[:, :, 0 : self.num_streams_per_tx, :, :]  # (64, 1, 2, 14, 76)
            llr = self.mydemapper(
                [x_hat, no]
            )  # [64, 1, 16, 14, 76]=>(64, 1, 16, 14, 304)
            # Reshape the array by collapsing the last two dimensions
            llr_est = llr.reshape(llr.shape[:-2] + (-1,))  # (64, 1, 16, 4256)
            # elif self.pilot_pattern == "empty" and perfect_csi == True:
            llr_perfect = self.mydemapper(
                [x_rg, no]
            )  # (64, 1, 2, 14, 76)=>(64, 1, 2, 14, 304)
            llr_perfect = llr_perfect.reshape(
                llr_perfect.shape[:-2] + (-1,)
            )  # (64, 1, 2, 4256)
            # llr_est = llr_perfect
            b_perfect = hard_decisions(
                llr_perfect, np.int32
            )  ##(64, 1, 1, 4256) 0,1 [64, 1, 1, 14, 304] 2128
            if b is not None:
                BER = calculate_BER(b, b_perfect)
                print("Perfect BER:", BER)
        else:  # channel estimation or perfect_csi
            # perform channel estimation via pilots
            y = tf.convert_to_tensor(y, dtype=tf.complex64)
            if self.channeltype == "ofdm":
                h_hat, err_var, h_perfect, err_var_perfect = (
                    self.ofdmchannel_estimation(
                        y=y, no=no, h_out=h_out, perfect_csi=perfect_csi
                    )
                )
            else:  # time channel
                h_hat, err_var, h_perfect, err_var_perfect = (
                    self.timechannel_estimation(
                        y=y, no=no, a=h_b, tau=tau_b, perfect_csi=perfect_csi
                    )
                )
                h_hat = tf.convert_to_tensor(
                    h_hat, dtype=tf.complex64
                )  # (2, 1, 16, 1, 2, 14, 64)
                # err_var=tf.convert_to_tensor(err_var, dtype=tf.complex64) #(1, 1, 1, 1, 2, 14, 64)

            # h_hat shape: (64, 1, 16, 1, 2, 14, 64), err_var: (1, 1, 1, 1, 2, 14, 64)
            # input (y, h_hat, err_var, no)
            # Received OFDM resource grid after cyclic prefix removal and FFT y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            # Channel estimates for all streams from all transmitters h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], tf.complex
            x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
            # Estimated symbols x_hat : [batch_size, num_tx, num_streams, num_data_symbols], tf.complex
            # Effective noise variance for each estimated symbol no_eff : [batch_size, num_tx, num_streams, num_data_symbols], tf.float
            x_hat = x_hat.numpy()  # x_hat: (2, 1, 2, 768), no_eff: (2, 1, 2, 768)
            no_eff = no_eff.numpy()
            no_eff = np.mean(no_eff)

            llr_est = self.mydemapper([x_hat, no_eff])  # (2, 1, 2, 3072)
            # output: [batch size, num_rx, num_rx_ant, n * num_bits_per_symbol]

        # llr_est #(64, 1, 1, 4256)
        if self.USE_LDPC:
            b_hat_tf = self.decoder(llr_est)  # [64, 1, 1, 2128]
            b_hat = b_hat_tf.numpy()
        else:
            b_hat = hard_decisions(llr_est, np.int32)  # (2, 1, 2, 3072)
        if b is not None:
            BER = calculate_BER(b, b_hat)
            print("BER Value:", BER)
            return b_hat, BER
        else:
            return b_hat, None

    def gettransmitSignal(self, b, ebno_db=15.0, perfect_csi=False):
        # Part 1: Transmission
        # Compute the noise power for a given Eb/No value. This takes not only the coderate but also the overheads related pilot transmissions and nulled carriers
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        # Convert it to a NumPy float
        no = np.float32(no)  # 0.0158

        # we generate random batches of CIR, transform them in the frequency domain and apply them to the resource grid in the frequency domain.
        h_b, tau_b = (
            self.get_channelcir()
        )  # h_b: (128, 1, 16, 1, 2, 23, 14), tau_b: (128, 1, 1, 23)
        if self.channeltype == "ofdm":
            h_out = self.get_OFDMchannelresponse(h_b, tau_b)  # cir_to_ofdm_channel
            print("h_freq shape:", h_out.shape)  # (128, 1, 16, 1, 2, 14, 76)
        elif self.channeltype == "time":
            h_out = self.get_timechannelresponse(
                h_b, tau_b
            )  # (64, 1, 16, 1, 2, 1174, 27)

        # Transmitter
        # This calls the uplinktransmission method, which simulates the transmission of data. It takes the input bits b, the noise power no, and the channel response h_out. The method returns:
        # y: The received signal after transmission.
        # x_rg: The resource grid used for transmission.
        # x: The transmitted signal.
        # b: The original bits.
        y, x_rg, x, b = self.uplinktransmission(
            b=b, no=no, h_out=h_out
        )  # y = self.applychannel([x_rg, h_out, no])
        print(
            "y shape:", y.shape
        )  # (64, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        return y, x_rg, x, b, no, h_b, tau_b, h_out

    def __call__(
        self, b=None, ebno_db=15.0, perfect_csi=False, datapath="data/saved_data.npy"
    ):

        # Part 1: Transmission
        # Compute the noise power for a given Eb/No value. This takes not only the coderate but also the overheads related pilot transmissions and nulled carriers
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate)
        # Convert it to a NumPy float
        no = np.float32(no)  # 0.0158

        # we generate random batches of CIR, transform them in the frequency domain and apply them to the resource grid in the frequency domain.
        h_b, tau_b = (
            self.get_channelcir()
        )  # h_b: (128, 1, 16, 1, 2, 23, 14), tau_b: (128, 1, 1, 23)
        if self.channeltype == "ofdm":
            h_out = self.get_OFDMchannelresponse(h_b, tau_b)  # cir_to_ofdm_channel
            print("h_freq shape:", h_out.shape)  # (128, 1, 16, 1, 2, 14, 76)
        elif self.channeltype == "time":
            h_out = self.get_timechannelresponse(
                h_b, tau_b
            )  # (64, 1, 16, 1, 2, 1174, 27)

        # Transmitter
        # This calls the uplinktransmission method, which simulates the transmission of data. It takes the input bits b, the noise power no, and the channel response h_out. The method returns:
        # y: The received signal after transmission.
        # x_rg: The resource grid used for transmission.
        # x: The transmitted signal.
        # b: The original bits.
        y, x_rg, x, b = self.uplinktransmission(
            b=b, no=no, h_out=h_out
        )  # y = self.applychannel([x_rg, h_out, no])
        print(
            "y shape:", y.shape
        )  # (64, 1, 16, 14, 76) [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

        # Transmission part is done,
        # Receiver part starts
        # Option1:
        # b_hat, BER = self.receiver(y, no, x_rg, b=b, h_b=h_b, tau_b=tau_b, h_out=h_out, perfect_csi= perfect_csi)
        # Option2:
        # x_hat, no_eff, h_hat, err_var, h_perfect, err_var_perfect = self.channelest_equ(
        #     y, no, h_b=h_b, tau_b=tau_b, h_out=h_out, perfect_csi=perfect_csi
        # )

        # adding nrx
        # Load the model
        x_hat, no_eff = process_signal(y, x, no, h_b, tau_b, h_out)

        # Outputs:
        # x_hat: The estimated transmitted signal.
        # no_eff: The effective noise after processing.
        # h_hat: The estimated channel response.
        # err_var: The error variance.
        # h_perfect: The perfect channel response (if applicable).
        # err_var_perfect: The error variance for the perfect channel.

        # demapper_Decision takes the estimated transmitted signal x_hat and the effective noise no_eff. It returns:
        # b_hat: The estimated bits after demapping.
        # llr_est: The log-likelihood ratios for the estimated bits.
        b_hat, llr_est = self.demapper_decision(x_hat=x_hat, no_eff=no_eff)
        BER = calculate_BER(b, b_hat)
        print("BER Value:", BER)

        # if self.savedata:
        #     saved_data = self.save_parameters()
        #     saved_data["no"] = no
        #     saved_data["h_b"] = h_b
        #     saved_data["tau_b"] = tau_b
        #     saved_data["h_out"] = h_out
        #     saved_data["y"] = y
        #     saved_data["x_rg"] = x_rg
        #     saved_data["x"] = x
        #     saved_data["b"] = b
        #     saved_data["x_hat"] = x_hat
        #     saved_data["no_eff"] = no_eff
        #     saved_data["h_hat"] = to_numpy(h_hat)
        #     # saved_data['err_var']=to_numpy(err_var)
        #     saved_data["err_var"] = err_var
        #     saved_data["h_perfect"] = h_perfect
        #     saved_data["err_var_perfect"] = err_var_perfect
        #     saved_data["b_hat"] = b_hat
        #     saved_data["llr_est"] = llr_est
        #     saved_data["BER"] = BER
        #     np.save(datapath, saved_data)
        # np.load d2.item() to retrieve the actual dict object first:
        return b_hat, BER

    def save_parameters(self):
        saved_data = {}
        saved_data["currenttime"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        saved_data["channeltype"] = self.channeltype
        saved_data["channeldataset"] = self.channeldataset
        saved_data["fft_size"] = self.fft_size
        saved_data["batch_size"] = self.batch_size
        saved_data["num_ofdm_symbols"] = self.num_ofdm_symbols
        saved_data["num_bits_per_symbol"] = self.num_bits_per_symbol
        saved_data["pilot_pattern"] = self.pilot_pattern
        saved_data["pilots"] = self.RESOURCE_GRID.pilot_pattern.pilots  # new added
        saved_data["num_data_symbols"] = self.RESOURCE_GRID.num_data_symbols
        saved_data["cyclic_prefix_length"] = self.RESOURCE_GRID.cyclic_prefix_length
        saved_data["ofdm_symbol_duration"] = self.RESOURCE_GRID.ofdm_symbol_duration
        saved_data["num_time_samples"] = self.RESOURCE_GRID.num_time_samples
        saved_data["bandwidth"] = self.RESOURCE_GRID.bandwidth
        saved_data["scenario"] = self.scenario
        saved_data["dataset_folder"] = self.dataset_folder
        saved_data["num_ut"] = self.num_ut
        saved_data["num_bs"] = self.num_bs
        saved_data["num_ut_ant"] = self.num_ut_ant
        saved_data["num_bs_ant"] = self.num_bs_ant
        saved_data["direction"] = self.direction
        saved_data["num_streams_per_tx"] = self.num_streams_per_tx
        saved_data["cyclic_prefix_length"] = self.cyclic_prefix_length
        saved_data["num_guard_carriers"] = self.num_guard_carriers
        saved_data["pilot_ofdm_symbol_indices"] = self.pilot_ofdm_symbol_indices
        saved_data["frequencies"] = self.frequencies
        saved_data["coderate"] = self.coderate
        saved_data["k"] = self.k  # num of information per codeword
        saved_data["n"] = (
            self.n
        )  # Codeword length n = int(RESOURCE_GRID.num_data_symbols * num_bits_per_symbol) #num_data_symbols: if empty 1064*4=4256, else, 768*4=3072
        saved_data["num_time_steps"] = self.num_time_steps
        saved_data["sampling_frequency"] = self.sampling_frequency

        return saved_data

        # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]


def test_DeepMIMOchannel(scenario="O1_60", dataset_folder="data/DeepMIMO"):
    transmit = Transmitter(
        channeldataset="deepmimo",
        channeltype="ofdm",
        scenario=scenario,
        dataset_folder=dataset_folder,
        direction="uplink",
        num_ut=1,
        num_ut_ant=1,
        num_bs=1,
        num_bs_ant=16,
        batch_size=2,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
    )
    b_hat, BER = transmit(ebno_db=5.0, perfect_csi=False)
    b_hat, BER = transmit(ebno_db=5.0, perfect_csi=True)


def test_CDLchannel():
    # transmit = Transmitter(channeldataset='cdl', channeltype='ofdm', direction='uplink', \
    #                 batch_size =64, fft_size = 76, num_ofdm_symbols=14, num_bits_per_symbol = 4,  \
    #                 subcarrier_spacing=60e3, \
    #                 USE_LDPC = False, pilot_pattern = "empty", guards=False, showfig=showfigure)
    # b_hat, BER = transmit(ebno_db = 15.0, channeltype='ofdm', perfect_csi=False)
    # b_hat, BER = transmit(ebno_db = 15.0, channeltype='ofdm', perfect_csi=True)
    # not work for MIMO

    transmit = Transmitter(
        channeldataset="cdl",
        channeltype="time",
        direction="uplink",
        batch_size=2,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
        savedata=True,
    )
    # transmit.test_timechannel_estimation(b=None, no=0.025)
    b_hat, BER = transmit(
        ebno_db=5.0, perfect_csi=False, datapath="data/cdl_time_saved_ebno5.npy"
    )
    b_hat, BER = transmit(
        ebno_db=5.0,
        perfect_csi=True,
        datapath="data/cdl_time_saved_ebno5perfectcsi.npy",
    )

    transmit = Transmitter(
        channeldataset="cdl",
        channeltype="ofdm",
        direction="uplink",
        batch_size=64,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=4,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
    )  # "kronecker" "empty"
    # transmit.get_channelcir()
    # h_out=transmit.get_OFDMchannelresponse()
    b_hat, BER = transmit(
        ebno_db=5.0, perfect_csi=False, datapath="data/cdl_ofdm_saved_ebno5.npy"
    )
    b_hat, BER = transmit(
        ebno_db=5.0,
        perfect_csi=True,
        datapath="data/cdl_ofdm_saved_ebno5perfectcsi.npy",
    )


def sim_bersingle(channeldataset="cdl", channeltype="time"):
    # Bit per channel use
    NUM_BITS_PER_SYMBOL = 2  # QPSK

    # Minimum value of Eb/N0 [dB] for simulations
    EBN0_DB_MIN = -3.0

    # Maximum value of Eb/N0 [dB] for simulations
    EBN0_DB_MAX = 25.0  # 5.0

    # How many examples are processed by Sionna in parallel
    BATCH_SIZE = 2

    # Define the number of UT and BS antennas
    NUM_UT = 1
    NUM_BS = 1
    NUM_UT_ANT = 2
    NUM_BS_ANT = 16

    ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

    showfigure = False
    eval_transceiver = Transmitter(
        channeldataset=channeldataset,
        channeltype=channeltype,
        direction="uplink",
        batch_size=BATCH_SIZE,
        fft_size=76,
        num_ut=NUM_UT,
        num_ut_ant=NUM_UT_ANT,
        num_bs=NUM_BS,
        num_bs_ant=NUM_BS_ANT,
        num_ofdm_symbols=14,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
    )
    # transmit.test_timechannel_estimation(b=None, no=0.0)
    # b_hat, BER = eval_transceiver(ebno_db = 25.0, perfect_csi=False)
    # b_hat, BER = eval_transceiver(ebno_db = 25.0, perfect_csi=True)

    # channeltype="perfect", "awgn", "ofdm", "time"
    # Number of information bits per codeword
    k = eval_transceiver.k
    binary_source = BinarySource()
    NUM_STREAMS_PER_TX = eval_transceiver.num_streams_per_tx
    # Start Transmitter self.k Number of information bits per codeword
    b = binary_source(
        [BATCH_SIZE, 1, NUM_STREAMS_PER_TX, k]
    )  # [batch_size, num_tx, num_streams_per_tx, num_databits]

    b_hat, BER = eval_transceiver(ebno_db=25.0, perfect_csi=False)

    bers, blers, BERs = sim_ber(ebno_dbs, eval_transceiver, b, BATCH_SIZE)
    # ber_plot_single(ebno_dbs, bers, title = "BER Simulation", savefigpath='./data/bernew.jpg')
    figpath = "./data/" + channeldataset + "_" + channeltype
    ber_plot_single2(
        ebno_dbs=ebno_dbs,
        bers=bers,
        is_bler=False,
        title="BER Simulation",
        savefigpath=figpath + "_ber.pdf",
    )
    ber_plot_single2(
        ebno_dbs=ebno_dbs,
        bers=blers,
        is_bler=True,
        title="BER Simulation",
        savefigpath=figpath + "_ber.pdf",
    )


def sim_bermulti():
    # Bit per channel use
    NUM_BITS_PER_SYMBOL = 2  # QPSK

    # Minimum value of Eb/N0 [dB] for simulations
    EBN0_DB_MIN = -3.0

    # Maximum value of Eb/N0 [dB] for simulations
    EBN0_DB_MAX = 25.0  # 5.0

    # How many examples are processed by Sionna in parallel
    BATCH_SIZE = 2

    # Define the number of UT and BS antennas
    NUM_UT = 1
    NUM_BS = 1
    NUM_UT_ANT = 2
    NUM_BS_ANT = 16

    ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)

    showfigure = False
    channeldataset = "cdl"
    channeltype = "time"
    eval_transceiver = Transmitter(
        channeldataset=channeldataset,
        channeltype=channeltype,
        direction="uplink",
        batch_size=BATCH_SIZE,
        fft_size=76,
        num_ut=NUM_UT,
        num_ut_ant=NUM_UT_ANT,
        num_bs=NUM_BS,
        num_bs_ant=NUM_BS_ANT,
        num_ofdm_symbols=14,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
    )
    # transmit.test_timechannel_estimation(b=None, no=0.0)
    # b_hat, BER = eval_transceiver(ebno_db = 25.0, perfect_csi=False)
    # b_hat, BER = eval_transceiver(ebno_db = 25.0, perfect_csi=True)

    # channeltype="perfect", "awgn", "ofdm", "time"
    # Number of information bits per codeword
    k = eval_transceiver.k
    binary_source = BinarySource()
    NUM_STREAMS_PER_TX = eval_transceiver.num_streams_per_tx
    # Start Transmitter self.k Number of information bits per codeword
    b = binary_source(
        [BATCH_SIZE, 1, NUM_STREAMS_PER_TX, k]
    )  # [batch_size, num_tx, num_streams_per_tx, num_databits]

    b_hat, BER = eval_transceiver(ebno_db=25.0, perfect_csi=False)

    # Multi-scenario testing
    BER_list = []
    legend = []

    bers = simulationloop(ebno_dbs, eval_transceiver, b)
    legend.append(f"Channeltype: {channeltype}")
    BER_list.append(bers)

    # Case2
    # channeldataset='cdl'
    # channeltype='time' #'ofdm'
    # eval_transceiver = Transmitter(channeldataset=channeldataset, channeltype=channeltype, direction='uplink', \
    #                 batch_size =BATCH_SIZE, fft_size = 76, num_ut = NUM_UT, num_ut_ant=NUM_UT_ANT, num_bs = NUM_BS, num_bs_ant=NUM_BS_ANT, \
    #                     num_ofdm_symbols=14, num_bits_per_symbol = NUM_BITS_PER_SYMBOL,  \
    #                 subcarrier_spacing=60e3, \
    #                 USE_LDPC = True, pilot_pattern = "kronecker", guards=True, showfig=showfigure)
    # bers=simulationloop(ebno_dbs, eval_transceiver, b)
    # legend.append(f'Channeltype: {channeltype}, use LDPC')
    # BER_list.append(bers)

    # Case3
    channeldataset = "cdl"
    channeltype = "time"  #'ofdm'
    eval_transceiver = Transmitter(
        channeldataset=channeldataset,
        channeltype=channeltype,
        direction="uplink",
        batch_size=BATCH_SIZE,
        fft_size=76,
        num_ut=NUM_UT,
        num_ut_ant=NUM_UT_ANT,
        num_bs=NUM_BS,
        num_bs_ant=NUM_BS_ANT,
        num_ofdm_symbols=14,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        # showfig=showfigure,
        showfig=True,
    )
    bers = simulationloop(ebno_dbs, eval_transceiver, b, perfect_csi=True)
    legend.append(f"Channeltype: {channeltype}, perfect CSI")
    BER_list.append(bers)

    ber_plot(
        ebno_dbs,
        BER_list,
        legend=legend,
        ylabel="BER",
        title="Bit Error Rate",
        ebno=True,
        xlim=None,
        ylim=None,
        is_bler=None,
        savefigpath="./data/bermultilistnew.pdf",
    )


def sim_bersingle2(
    channeldataset="deepmimo",
    channeltype="ofdm",
    NUM_BITS_PER_SYMBOL=2,
    EBN0_DB_MIN=-5.0,
    EBN0_DB_MAX=25.0,
    BATCH_SIZE=128,
    NUM_UT=1,
    NUM_BS=1,
    NUM_UT_ANT=1,
    NUM_BS_ANT=16,
    showfigure=False,
    datapathbase="data/",
    dataset_folder="",
):
    # Bit per channel use
    # NUM_BITS_PER_SYMBOL = 2 # QPSK

    # # Minimum value of Eb/N0 [dB] for simulations
    # EBN0_DB_MIN = -5.0 #-3.0

    # # Maximum value of Eb/N0 [dB] for simulations
    # EBN0_DB_MAX = 25.0 #5.0

    # # How many examples are processed by Sionna in parallel
    # BATCH_SIZE = 128 #64

    # # Define the number of UT and BS antennas
    # NUM_UT = 1
    # NUM_BS = 1
    # NUM_UT_ANT = 1 #2 is not working
    # NUM_BS_ANT = 16

    if not os.path.exists(datapathbase):
        os.makedirs(datapathbase)

    # - Eb/N0 Values: This line generates 20 evenly spaced values between EBN0_DB_MIN and EBN0_DB_MAX. These values will be used in the simulation to evaluate the performance of the communication system at different noise levels.
    ebno_dbs = np.linspace(EBN0_DB_MIN, EBN0_DB_MAX, 20)
    scenario = "O1_60"
    datapath = datapathbase + channeldataset + "_" + channeltype
    eval_transceiver = Transmitter(
        channeldataset=channeldataset,
        channeltype=channeltype,
        scenario=scenario,
        dataset_folder=dataset_folder,
        direction="uplink",
        num_ut=NUM_UT,
        num_ut_ant=NUM_UT_ANT,
        num_bs=NUM_BS,
        num_bs_ant=NUM_BS_ANT,
        batch_size=BATCH_SIZE,
        fft_size=76,
        num_ofdm_symbols=14,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        subcarrier_spacing=60e3,
        USE_LDPC=False,
        pilot_pattern="kronecker",
        guards=True,
        showfig=showfigure,
        savedata=True,
        outputpath=datapathbase,  # deepmimo_ofdm_
    )

    # b_hat, BER = eval_transceiver(
    #     ebno_db=5.0,
    #     perfect_csi=False,
    #     datapath=datapath + "_ebno5.npy",  # deepmimo_ofdm_
    # )

    # channeltype="perfect", "awgn", "ofdm", "time"
    # Number of information bits per codeword
    # Here, k retrieves the number of information bits per codeword from the eval_transceiver
    k = eval_transceiver.k
    binary_source = BinarySource()
    # Number of streams per transmitter
    NUM_STREAMS_PER_TX = eval_transceiver.num_streams_per_tx
    # Start Transmitter self.k Number of information bits per codeword

    # Generate Random Bits: This generates random binary data for the specified batch size, number of transmitters, number of streams, and number of bits per codeword.
    b = binary_source(
        [BATCH_SIZE, 1, NUM_STREAMS_PER_TX, k]
    )  # [batch_size, num_tx, num_streams_per_tx, num_databits]

    # #  - Second Transmission: Similar to the previous transmission, this simulates the transmission with an Eb/N0 value of 25 dB and saves the results.
    #     b_hat, BER = eval_transceiver(
    #         ebno_db=25.0, perfect_csi=False, datapath=datapath + "_ebno25.npy"
    #     )

    bers, blers, BERs = sim_ber(ebno_dbs, eval_transceiver, b, BATCH_SIZE)
    # ber_plot_single(ebno_dbs, bers, title = "BER Simulation", savefigpath='./data/bernew.jpg')
    ber_plot_single2(
        ebno_dbs,
        bers,
        is_bler=False,
        title="BER Simulation",
        savefigpath=datapath + "_ber.pdf",
    )
    ber_plot_single2(
        ebno_dbs,
        blers,
        is_bler=True,
        title="BLER Simulation",
        savefigpath=datapath + "_blers.pdf",
    )
    return bers, blers, BERs


if __name__ == "__main__":

    # testOFDMModulatorDemodulator()
    scenario = "O1_60"
    # dataset_folder='data' #r'D:\Dataset\CommunicationDataset\O1_60'
    dataset_folder = r"D:\Research\AIsensing\deeplearning\Database"
    # dataset_folder=r'D:\Dataset\CommunicationDataset\O1_60'
    cdltest = False
    bertest = True
    showfigure = True

    # test_DeepMIMOchannel()
    # bers, blers, BERs = sim_bersingle2(
    #     channeldataset="cdl",
    #     channeltype="ofdm",
    #     NUM_BITS_PER_SYMBOL=2,
    #     EBN0_DB_MIN=-5.0,
    #     EBN0_DB_MAX=25.0,
    #     BATCH_SIZE=128,
    #     NUM_UT=1,
    #     NUM_BS=1,
    #     NUM_UT_ANT=2,
    #     NUM_BS_ANT=16,
    #     showfigure=showfigure,
    #     datapathbase="data/cdl/",
    # )
    bers, blers, BERs = sim_bersingle2(
        channeldataset="deepmimo",
        channeltype="ofdm",
        NUM_BITS_PER_SYMBOL=2,
        EBN0_DB_MIN=-5.0,
        EBN0_DB_MAX=25.0,
        BATCH_SIZE=128,
        NUM_UT=1,
        NUM_BS=1,
        NUM_UT_ANT=1,
        NUM_BS_ANT=16,
        showfigure=showfigure,
        datapathbase="data/",
    )

    # bers, blers, BERs = sim_bersingle2(
    #     channeldataset="cdl",
    #     channeltype="time",
    #     NUM_BITS_PER_SYMBOL=2,
    #     EBN0_DB_MIN=-5.0,
    #     EBN0_DB_MAX=25.0,
    #     BATCH_SIZE=32,
    #     NUM_UT=1,
    #     NUM_BS=1,
    #     NUM_UT_ANT=2,
    #     NUM_BS_ANT=16,
    #     showfigure=showfigure,
    #     datapathbase="data/",
    # )
    # bers, blers, BERs = sim_bersingle2(
    #     channeldataset="deepmimo",
    #     channeltype="time",
    #     NUM_BITS_PER_SYMBOL=2,
    #     EBN0_DB_MIN=-5.0,
    #     EBN0_DB_MAX=25.0,
    #     BATCH_SIZE=32,
    #     NUM_UT=1,
    #     NUM_BS=1,
    #     NUM_UT_ANT=1,
    #     NUM_BS_ANT=16,
    #     showfigure=showfigure,
    #     datapathbase="data/",
    # )

    if cdltest is True:
        test_CDLchannel()
    if bertest is True:
        # sim_bersingle(channeldataset='cdl', channeltype='ofdm') #channeltype='time'
        sim_bermulti()
    print("Finished")
