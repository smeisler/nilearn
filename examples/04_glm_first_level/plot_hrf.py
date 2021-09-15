"""
Example of MRI response functions
=================================

Within this example we are going to plot the hemodynamic response function
(:term:`HRF`) model in :term:`SPM` together with the :term:`HRF` shape proposed by G.Glover, as well as
their time and dispersion derivatives.
We also illustrate how users can input a custom response function,
which can for instance be useful when dealing with non human primate data
acquired using a contrast agent. In our case, we input a custom response function
for MION, a common agent used to enhance contrast on MRI images of monkeys.

The :term:`HRF` is the filter which couples neural responses to the metabolic-related
changes in the MRI signal. :term:`HRF` models are simply phenomenological.

In current analysis frameworks, the choice of :term:`HRF` model is essentially left to
the user. Fortunately, using the :term:`SPM` or Glover model does not make a huge
difference. Adding derivatives should be considered whenever timing
information has some degree of uncertainty, and is actually useful to detect
timing issues.

This example requires matplotlib and scipy.
"""

#########################################################################
# Define stimulus parameters and response models
# ----------------------------------------------
#
# To get an impulse response, we simulate a single event occurring at time t=0,
# with duration 1s.

import numpy as np

frame_times = np.linspace(0, 30, 61)
onset, amplitude, duration = 0.0, 1.0, 1.0
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)

#########################################################################
# Make a time array of this condition for display:
stim = np.zeros_like(frame_times)
stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude

#########################################################################
# Define custom response function for MION:
from scipy.stats import gamma


def mion_response_function(tr, oversampling=50, time_length=32.0, onset=0.0):
    dt = tr / oversampling
    time_stamps = np.linspace(
        0, time_length, np.rint(float(time_length) / dt).astype(int)
    )
    time_stamps -= onset

    delay = 5
    
    response_function = gamma.pdf(time_stamps, delay, scale=4)
    response_function /= response_function.sum()

    return response_function


#########################################################################
# Define response function models to be displayed:

rf_models = [
    ("spm + derivative + dispersion", "SPM HRF"),
    ("glover + derivative + dispersion", "Glover HRF"),
    (mion_response_function, "Mion RF"),
]

#########################################################################
# Sample and plot response functions
# ----------------------------------

import matplotlib.pyplot as plt
from nilearn.glm.first_level import compute_regressor

fig = plt.figure(figsize=(9, 4))
for i, (rf_model, model_title) in enumerate(rf_models):
    # compute signal of interest by convolution
    signal, labels = compute_regressor(
        exp_condition, rf_model, frame_times, con_id="main", oversampling=16
    )

    # plot signal
    plt.subplot(1, len(rf_models), i + 1)
    plt.fill(frame_times, stim, "k", alpha=0.5, label="stimulus")
    for j in range(signal.shape[1]):
        plt.plot(
            frame_times,
            signal.T[j],
            label=labels[j] if labels is not None else None,
        )
    plt.xlabel("time (s)")
    plt.legend(loc=1)
    plt.title(model_title)

# adjust plot
plt.subplots_adjust(bottom=0.12)
plt.show()
