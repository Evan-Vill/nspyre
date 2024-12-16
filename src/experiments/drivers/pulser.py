"""
Control class for pulse generation using Swabian pulse streamer

Reference:
    [1] examples in 'pulsestreamer' python package
    [2] 'pi3diamond' control software in Sen Yang group

Author: ChunTung Cheung
Email: ctcheung1123@gmail.com
Created:  2023-01-11
Modified: 2024-09-24
"""

import logging
import time

import numpy as np

# import class Sequence and OutputState for advanced sequence building
# import enum types
from pulsestreamer import (
    OutputState,
    PulseStreamer,
    Sequence,
    TriggerRearm,
    TriggerStart,
    findPulseStreamers,
)

logger = logging.getLogger(__name__)
# from pulsestreamer import OutputState

CHNUM_DO = 8
CHNUM_AO = 2
HIGH = 1
LOW = 0
INF = np.iinfo(np.int64).max
REPEAT_INFINITELY = -1

# use 0 to 7 for digital channels
# use 8 to 9 for analog channels
CHANNEL_MAP = {
    "ch0": 0,
    "ch1": 1,
    "ch2": 2,
    "ch3": 3,
    "ch4": 4,
    "ch5": 5,
    "ch6": 6,
    "ch7": 7,
    "ch8": 8,
    "ch9": 9,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
}

CHANNEL_OFFSET = {
    "ch0": 0,
    "ch1": 0,
    "ch2": 0,
    "ch3": 0,
    "ch4": 0,
    "ch5": 0,
    "ch6": 0,
    "ch7": 0,
    "ch8": 0,
    "ch9": 0,
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0,
    "7": 0,
    "8": 0,
    "9": 0,
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
}


def invert_chmap(my_map):
    inv_map = {}
    invertedkey = []
    for k, v in my_map.items():
        if v not in invertedkey:
            inv_map[v] = k
            invertedkey.append(v)
    return inv_map


class PulseGenerator(PulseStreamer):
    def __init__(self, ip="", chmap=CHANNEL_MAP, choffs=CHANNEL_OFFSET):
        if ip == "":
            devices = findPulseStreamers()
            # DHCP is activated in factory settings
            if devices != []:
                logger.debug(f"Detected Pulse Streamer 8/2: {devices}")

                # Connect to the first discovered Pulse Streamer
                ip = devices[0][0]
            else:
                # if discovery failed try to connect by the default hostname
                # IP address of the pulse streamer (default hostname is 'pulsestreamer')
                logger.exception("No Pulse Streamer found")
                ip = "pulsestreamer"
        super().__init__(ip)
        self.setChMap(chmap)
        self.choffs = CHANNEL_OFFSET.copy()
        self.setChOffset(choffs.copy())
        self.seq = Sequence()

    def setChMap(self, chmap):
        self.chmap = chmap.copy()
        self._chmap_inv = invert_chmap(self.chmap)
        self.chmap.update(CHANNEL_MAP.copy())

    def resetChOffset(self):
        self.choffs = CHANNEL_OFFSET.copy()

    def setChOffset(self, choffs):
        # the offsets only apply when using time-based pulse sequence and the transaltor
        # users need to include the offsets manually when using channel-based sequence

        # make sure the offset values are non-negative
        base = min(list(choffs.values()) + [0])

        # assign the offsets specified
        for key, value in choffs.items():
            # make sure the offset values are non-negative
            offset = value - base
            self.choffs[key] = offset
            self.choffs[self.chmap[key]] = offset
            self.choffs[f"ch{self.chmap[key]}"] = offset
            self.choffs[f"{self.chmap[key]}"] = offset

    def setTrigger(self, start=TriggerStart.IMMEDIATE, rearm=TriggerRearm.MANUAL):
        # Default: Start the sequence after the upload and disable the retrigger-function
        super().setTrigger(start=start, rearm=rearm)

    def stream(
        self,
        seq="AUTO",
        n_runs=1,
        state_i=OutputState.ZERO(),
        state_f=OutputState.ZERO(),
    ):
        # run the sequence
        # n_runs = 'INFIITE' # repeat the sequence all the time

        # #reset the device - all outputs 0V
        # super().reset()

        # set constant state of the device
        super().constant(state_i)  # all outputs 0V
        if seq == "AUTO":
            super().stream(self.seq, n_runs, state_f)
        elif type(seq) is Sequence:
            super().stream(seq, n_runs, state_f)
        logger.info(f"Stream Sequence with {n_runs} runs")

    def resetSeq(self):
        self.seq = Sequence()
        self.stream()

    def reset(self):
        # reset system and start next sequence
        self.resetSeq()
        # input("\nPress ENTER to reset system and start delay-compensated sequence")
        super().reset()
        logger.info("Reset Pulse Streamer")

    def setDigital(self, ch, pulse_patt, offset=False):
        if offset:
            pulse_patt = [(self.choffs[ch], 0)] + list(pulse_patt)
        self.seq.setDigital(self.chmap[ch], pulse_patt)

    def setAnalog(self, ch, pulse_patt, offset=False):
        if offset:
            pulse_patt = [(self.choffs[ch], 0)] + pulse_patt
        self.seq.setAnalog(self.chmap[ch] % CHNUM_DO, pulse_patt)

    def plotSeq(self, plot_all=True):
        """
        modify from Swabian Instrument package
        plots sequence data using plotly
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.exception(
                "Module plotly not found.\n"
                + "For visualizing the sequence data via Sequence().plot(), \n"
                + "please manually install the package by typing: \n"
                + "   > pip install plotly, nbformat\n"
                + "in your terminal."
            )
            return

        # assuming self.seq.__pad_seq is a dictionary with the key as the sequence number and pattern_data as an array
        # where pattern_data[1] is channel data and pattern_data[2] is time data
        self.seq._Sequence__pad()

        if plot_all:
            # Create a subplot grid with 10 rows (1 for each channel)
            fig = make_subplots(
                rows=10, cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            # Loop through the sequence dictionary
            for key, pattern_data in self.seq._Sequence__pad_seq.items():
                # Create the time and channel data for plotting
                t = np.concatenate((np.array([0], dtype=np.int64), pattern_data[2]))
                plot_ch_data = np.append(pattern_data[1], pattern_data[1][-1])

                # Determine the row for subplot
                row = 10 - key

                if key > (Sequence.digital_channel - 1):
                    # Analog channel plotting
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=plot_ch_data,
                            mode="lines",
                            name=f"A{key-Sequence.digital_channel}",
                            line_shape="hv",
                            line=dict(color="black"),
                        ),
                        row=row,
                        col=1,
                    )
                    fig.update_yaxes(
                        title_text=f"A{key-Sequence.digital_channel}",
                        range=[-1.5, 1.5],
                        tickfont=dict(size=6),
                        row=row,
                        col=1,
                    )
                else:
                    # Digital channel plotting
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=plot_ch_data,
                            mode="lines",
                            name=f"D{key}",
                            line_shape="hv",
                        ),
                        row=row,
                        col=1,
                    )
                    if key in self._chmap_inv.keys():
                        chanel_name = f"D{key}<br>{self._chmap_inv[key]}"
                    else:
                        chanel_name = f"D{key}"
                    fig.update_yaxes(
                        title_text=chanel_name,
                        range=[-0.4, 1.4],
                        showticklabels=False,
                        row=row,
                        col=1,
                    )

                # Disable the x-tick labels for all subplots except the last
                if key > 0:
                    fig.update_xaxes(showticklabels=False, row=row, col=1)
                else:
                    fig.update_xaxes(title_text="time/ns", row=row, col=1)

            # Layout adjustments
            fig.update_layout(
                height=600,
                width=600,
                title_text="Sequence",
                showlegend=False,
                margin=dict(l=50, r=50, t=40, b=40),
            )
        else:
            num_ch = len(self._chmap_inv)
            # Create a subplot grid with 10 rows (1 for each channel)
            fig = make_subplots(
                rows=num_ch, cols=1, shared_xaxes=True, vertical_spacing=0.02
            )
            row = 1
            for key, name in self._chmap_inv.items():
                # Create the time and channel data for plotting
                pattern_data = self.seq._Sequence__pad_seq[key]
                t = np.concatenate((np.array([0], dtype=np.int64), pattern_data[2]))
                plot_ch_data = np.append(pattern_data[1], pattern_data[1][-1])

                if key > (Sequence.digital_channel - 1):
                    # Analog channel plotting
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=plot_ch_data,
                            mode="lines",
                            name=f"A{key-Sequence.digital_channel}",
                            line_shape="hv",
                            line=dict(color="black"),
                        ),
                        row=row,
                        col=1,
                    )
                    fig.update_yaxes(
                        title_text=f"A{key-Sequence.digital_channel}<br>{self._chmap_inv[key]}",
                        range=[-1.5, 1.5],
                        tickfont=dict(size=6),
                        row=row,
                        col=1,
                    )
                else:
                    # Digital channel plotting
                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=plot_ch_data,
                            mode="lines",
                            name=f"D{key}",
                            line_shape="hv",
                        ),
                        row=row,
                        col=1,
                    )
                    chanel_name = f"D{key}<br>{name}"

                    fig.update_yaxes(
                        title_text=chanel_name,
                        range=[-0.4, 1.4],
                        showticklabels=False,
                        row=row,
                        col=1,
                    )

                # Disable the x-tick labels for all subplots except the last
                if row < num_ch:
                    fig.update_xaxes(showticklabels=False, row=row, col=1)
                else:
                    fig.update_xaxes(title_text="time/ns", row=row, col=1)
                row += 1
            # Layout adjustments
            fig.update_layout(
                height=100 + 50 * num_ch,
                width=600,
                title_text="Sequence",
                showlegend=False,
                margin=dict(l=50, r=50, t=40, b=40),
            )

        return fig

    def seqTranslator(self, seq_tbased):
        """
        WARNING!! currently this translator only works for digital channels
        TODO: translate both digital and analog channels
        TODO: improve the performance, currently it is too slow, probably due to the nested loop
            e.g.
            INFO the seq translator started
            Time taken for seqTranslator: 0.2845 seconds
            Time taken for processing: 2.3355 seconds
            Time taken for setting digital channels: 3.3116 seconds
        translate time-based sequence to channel-based sequence
        for example we translate
            seq_tbased = [
                            (["laser"], 300),
                            ([], 300),
                            ([], 300),
                            (["mw_A"], 300),
                            (["mw_B", "laser"], 300),
                            (["laser"], 300)
                         ]
        into
            seq_chbased = {
                        'laser': [(300, 1), (300, 0), (300, 0), (300, 0), (300, 1), (300, 1)],
                        'mw_B': [(300, 0), (300, 0), (300, 0), (300, 0), (300, 1), (300, 0)],
                        'mw_A': [(300, 0), (300, 0), (300, 0), (300, 1), (300, 0), (300, 0)]
                        }
        """
        total_time = 0
        ch_all = set()

        start = time.time()
        for channels, duration in seq_tbased:
            total_time += duration
            for ch in channels:
                ch_all.add(ch)
        assert (
            total_time // 1 == total_time
        ), "Sequence Duration must be Int since base unit is 1ns"
        end = time.time()
        logger.debug(f"Time taken for summarizing channels: {end-start:.4f} seconds")
        start_time = time.time()

        seq_chbased = {ch: [] for ch in ch_all}
        for channels, duration in seq_tbased:
            for ch, sequence in seq_chbased.items():
                sequence.append((duration, HIGH if ch in channels else LOW))

        end_time = time.time()
        logger.debug(f"Time taken for translation: {end_time - start_time:.4f} seconds")

        return total_time, seq_chbased

    def setSequence(self, seq_tbased):
        """
        set sequence directly using time-based sequence
        for example
            seq_tbased = [
                            (["laser"], 300),
                            ([], 300),
                            ([], 300),
                            (["mw_A"], 300),
                            (["mw_B", "laser"], 300),
                            (["laser"], 300)
                         ]
        """
        start = time.time()
        all_timestamps = [value for _, value in seq_tbased]
        total_time = sum(all_timestamps)
        assert (
            total_time // 1 == total_time
        ), "Sequence Duration must be Int since base unit is 1ns"

        ch_all = set()
        for channels, duration in seq_tbased:
            ch_all.update(channels)
        end = time.time()
        logger.debug(f"Time taken for summarizing channels: {end-start:.4f} seconds")

        start_time = time.time()
        self.resetSeq()
        self.seq._Sequence__sequence_up_to_date = False
        for ch in ch_all:
            if self.choffs[ch] >= 0:
                ch_state = [LOW]
                timeline = np.array([self.choffs[ch]] + all_timestamps, dtype=np.int64)
            else:
                ch_state = []
                timeline = np.array(all_timestamps, dtype=np.int64)
            for channels, duration in seq_tbased:
                chstate = HIGH if ch in channels else LOW
                ch_state.append(chstate)
            self.seq._Sequence__channel_digital[self.chmap[ch]] = (
                timeline,
                np.array(ch_state, dtype=np.int64),
                np.cumsum(timeline),
            )
        end_time = time.time()
        logger.debug(
            f"Time taken for setting digital channels: {end_time - start_time:.4f} seconds"
        )
        logger.info(f"Set Sequence total time {total_time} ns")
        return total_time
