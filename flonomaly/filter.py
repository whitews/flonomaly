import os
import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy import stats

import flowio

fig_size = (16, 4)


def get_false_bounds(bool_array):
    diff = np.diff(np.hstack((0, bool_array, 0)))

    start = np.where(diff == 1)
    end = np.where(diff == -1)

    return start[0], end[0]


def plot_channel(chan_events, label, xform=False, bad_events=None):
    if xform:
        chan_events = np.arcsinh(chan_events * 0.003)

    my_cmap = pyplot.cm.get_cmap('jet')
    my_cmap.set_under('w', alpha=0)

    bins = int(np.sqrt(chan_events.shape[0]))

    fig = pyplot.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(label, fontsize=16)

    ax.set_xlabel("Events", fontsize=14)

    event_range = range(0, chan_events.shape[0])

    ax.hist2d(
        event_range,
        chan_events,
        bins=[bins, bins],
        cmap=my_cmap,
        vmin=0.9
    )

    if bad_events is not None:
        starts, ends = get_false_bounds(bad_events)

        for i, s in enumerate(starts):
            ax.axvspan(
                event_range[s],
                event_range[ends[i] - 1],
                facecolor='pink',
                alpha=0.3,
                edgecolor='deeppink'
            )

    fig.tight_layout()

    pyplot.show()


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def filter_events(fcs_file, roll=1000, xform='auto', plot=False, ref_set_count=3):
    fd = flowio.FlowData(fcs_file)
    events = np.reshape(fd.events, (-1, fd.channel_count))

    channels = sorted([int(channel) - 1 for channel in fd.channels])

    event_count = events.shape[0]

    # start with boolean array where True is a bad event, initially all set to False,
    # we will OR them for each channel
    bad_events = np.zeros(event_count, dtype=bool)

    for channel in channels:
        channel_name = fd.channels[str(channel + 1)]['PnN']

        if channel_name == 'Time':
            continue
        elif channel_name[0:3] in ['FSC', 'SSC'] or xform is None:
            chan_events = events[:, channel]
        else:
            chan_events = events[:, channel]
            chan_events = np.arcsinh(chan_events * 0.003)

        rolling_mean = pd.rolling_mean(
            chan_events,
            roll,
            min_periods=1,
            center=True
        )

        median = np.median(rolling_mean)

        # find absolute difference from the median of the moving average
        median_diff = np.abs(rolling_mean - median)

        # sort the differences and take a random sample of size=roll from the top 20%
        # TODO: add check for whether there are ~ 2x the events of roll size
        reference_indices = np.argsort(median_diff)

        # create reference sets, we'll average the p-values from these
        ref_sets = []
        for i in range(0, ref_set_count):
            ref_subsample_idx = np.random.choice(int(event_count * 0.2), roll, replace=False)
            ref_sets.append(chan_events[reference_indices[ref_subsample_idx]])

        # calculate piece-wise KS test, we'll test every roll / 2 interval, cause
        # doing a true rolling window takes way too long
        strides = rolling_window(chan_events, roll)

        ks_x = []
        ks_y = []

        test_idx = list(range(0, len(strides), int(roll / 2)))
        test_idx.append(len(strides) - 1)  # cover last stride, to get any remainder

        for i in test_idx:
            kss = []

            for ref in ref_sets:
                kss.append(stats.ks_2samp(ref, strides[i]).pvalue)

            ks_x.append(i)

            ks_y.append(np.mean(kss))

        ks_y_roll = pd.rolling_mean(
            pd.Series(ks_y),
            7,
            min_periods=1,
            center=True
        )

        # interpolate our piecewise tests back to number of actual events
        interp_y = np.interp(range(0, event_count), ks_x, ks_y_roll)

        cutoff = 0.025
        bad_events = np.logical_or(bad_events, interp_y < cutoff)

        if plot:
            plot_channel(chan_events, " - ".join([str(channel + 1), channel_name]), xform=False)

            fig = pyplot.figure(figsize=(2 * ref_set_count, 4))

            for i, reference_events in enumerate(ref_sets):
                ax = fig.add_subplot(1, ref_set_count, i + 1)
                ax.set_xlim([0, reference_events.shape[0]])
                pyplot.scatter(np.arange(0, reference_events.shape[0]), reference_events, s=2,
                               edgecolors='none')

            fig.tight_layout()
            pyplot.show()

            fig = pyplot.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(" - ".join([str(channel + 1), channel_name, "Median Diff"]),
                         fontsize=16)
            ax.set_xlim([0, event_count])
            pyplot.plot(
                np.arange(0, event_count),
                median_diff,
                c='cornflowerblue',
                alpha=1.0,
                linewidth=1
            )

            fig.tight_layout()
            pyplot.show()

            fig = pyplot.figure(figsize=fig_size)
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlim([0, ks_x[-1]])
            ax.set_ylim([0, 1])
            pyplot.plot(
                ks_x,
                ks_y,
                c='cornflowerblue',
                alpha=0.6,
                linewidth=1
            )
            pyplot.plot(
                ks_x,
                ks_y_roll,
                c='darkorange',
                alpha=1.0,
                linewidth=2
            )

            ax.axhline(cutoff, linestyle='-', linewidth=1, c='coral')

            fig.tight_layout()
            pyplot.show()

    return bad_events


def plot_bad_events(fcs_file, bad_events, xform='auto'):
    fd = flowio.FlowData(fcs_file)
    events = np.reshape(fd.events, (-1, fd.channel_count))

    channels = sorted([int(channel) - 1 for channel in fd.channels])

    for channel in channels:
        channel_name = fd.channels[str(channel + 1)]['PnN']

        if channel_name == 'Time':
            continue
        elif channel_name[0:3] in ['FSC', 'SSC'] or xform is None:
            chan_events = events[:, channel]
        else:
            chan_events = events[:, channel]
            chan_events = np.arcsinh(chan_events * 0.003)

        plot_channel(
            chan_events,
            " - ".join([str(channel + 1), channel_name]),
            xform=False,
            bad_events=bad_events
        )


def create_filtered_fcs(fcs_file, results_dir, bad_events):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    flow_data = flowio.FlowData(fcs_file)
    events = np.reshape(flow_data.events, (-1, flow_data.channel_count))

    good_events = events[np.logical_not(bad_events)]
    bad_events = events[bad_events]

    base_name = os.path.basename(fcs_file)
    good_file_path = os.path.join(results_dir, base_name.replace('.fcs', '_good.fcs'))
    bad_file_path = os.path.join(results_dir, base_name.replace('.fcs', '_bad.fcs'))

    # build channel names
    channel_names = []
    opt_channel_names = []
    for channel in sorted([int(k) for k in flow_data.channels.keys()]):
        channel_names.append(flow_data.channels[str(channel)]['PnN'])

        if 'PnS' in flow_data.channels[str(channel)]:
            opt_channel_names.append(flow_data.channels[str(channel)]['PnS'])
        else:
            opt_channel_names.append(None)

    # build some extra metadata fields
    extra = {}
    acq_date = None
    if 'date' in flow_data.text:
        acq_date = flow_data.text['date']

    if 'timestep' in flow_data.text:
        extra['TIMESTEP'] = flow_data.text['timestep']

    if 'btim' in flow_data.text:
        extra['BTIM'] = flow_data.text['btim']

    if 'etim' in flow_data.text:
        extra['ETIM'] = flow_data.text['etim']

    good_fh = open(good_file_path, 'wb')
    bad_fh = open(bad_file_path, 'wb')

    flowio.create_fcs(
        good_events.flatten().tolist(),
        channel_names,
        good_fh,
        date=acq_date,
        extra=extra,
        opt_channel_names=opt_channel_names
    )
    good_fh.close()

    flowio.create_fcs(
        bad_events.flatten().tolist(),
        channel_names,
        bad_fh,
        date=acq_date,
        extra=extra,
        opt_channel_names=opt_channel_names
    )
    bad_fh.close()
