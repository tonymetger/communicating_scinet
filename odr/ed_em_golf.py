#   Copyright 2020 communicating_scinet (https://github.com/tonymetger/communicating_scinet)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
from scipy.integrate import solve_ivp
from .data_handler import Dataset
from itertools import product


def sample_in_range(r, shape):
    return r[0] + (r[1] - r[0]) * np.random.rand(*shape)


def corr_sample(mass_range, charge_range):
    m = sample_in_range(mass_range, shape=())
    while True:
        m_prop = (m - mass_range[0]) / (mass_range[1] - mass_range[0])
        loc = (charge_range[0] + charge_range[1]) / 2. + m_prop * (charge_range[1] - charge_range[0]) / 2.
        scale = (charge_range[1] - charge_range[0]) / 3.5
        q = np.random.normal(loc=loc, scale=scale)
        q = q if np.random.rand() > 0.5 else -q
        if q <= charge_range[1] and q >= charge_range[0]:
            break
    return (m, q)


def corr_set(num, mass_range, charge_range):
    mm1 = []
    mm2 = []
    qq1 = []
    qq2 = []
    for _ in range(num):
        m, q = corr_sample(mass_range, charge_range)
        mm1.append(m)
        qq1.append(q)

        m, q = corr_sample(mass_range, charge_range)
        mm2.append(m)
        qq2.append(q)
    return np.vstack([np.array(mm1), np.array(mm2)]).T, np.vstack([np.array(qq1), np.array(qq2)]).T,


def charge_test_rhs(q1, q2, m1, m2, d0):
    mu = m1 * m2 / (m1 + m2)
    f = lambda t, y: np.array([y[1], q1 * q2 / (mu * y[0]**2)])
    return f


def get_v(m, m_fix, q, Q, phi, d0):
    k = - q * Q / m
    v0_norm = (k**2 / d0**2 * (1 - np.sqrt(2))**2 / (np.cos(phi)**2 - np.cos(phi)**4))**(1 / 4.)
    v_bullet = (m + m_fix) / (2 * m_fix) * v0_norm
    return v_bullet


def operational_sample(mm, qq, d0, m_fix, q_fix, v_fix, obs_times):
    """
    mm: [mass_left, mass_right]
    qq: analagous
    """

    # dat[i, j, k]: i-th agent, j-th expriment (j=0: mass, j=1: charge), k-th time step
    dat = np.zeros((2, 2, len(obs_times)), dtype=float)

    # elastic impact experiment
    for i, m in enumerate(mm):
        v = 2 * m_fix / (m + m_fix) * v_fix
        dat[i, 0] = v * obs_times

    # charges experiment
    # the golf ball is placed at 0, the reference charge at d0, and both are free to move (no friction etc.)
    # the position of the golf ball is observed as a function of time
    for i, (m, q) in enumerate(zip(mm, qq)):
        y0 = [d0, 0]
        yy = solve_ivp(charge_test_rhs(q, q_fix, m, m_fix, d0), [min(obs_times), max(obs_times)],
                       y0, t_eval=obs_times)
        center_of_mass = m_fix / (m_fix + m) * d0
        r_golf = center_of_mass - m_fix / (m + m_fix) * yy.y[0]
        dat[i, 1] = r_golf

    return dat


def create_dataset(sample_num, multi_enc, correlated=False, mass_range=[1, 10],
                   charge_range=[-1, 1], set_params=None, noise_level=0):
    """
    Params:
    =======
    sample_num: 
        number of training examples in Dataset
    multi_enc:  
        if True, the Dataset is structured for two encoders
        if False, the Dataset is structured for one encoder
    correlated: 
        if True, hidden parameters are sampled according to a correlated distribution (restricted to the mass and charge ranges)
        if False, hidden parameters are sampled independently and uniformly
    mass_range: 
        interval from which masses will be sampled
    charge_range: 
        interval from which charges will be sampled
    set_params (dict): 
        custom settings for the global parameters, must use same names as global_params dictitonary below
    noise_level: 
        std.dev. of Gaussian noise added to all data points


    Returns:
    ========
    Dataset object. The attributes of this object have the following structure:

    input_data: [
                    [input_enc_1, input_enc_2, ...],
                    [input_enc_1, input_enc_2, ...],
                    ...
                ]
                where input_enc_i is a list (time series)

    questions:  [
                    [question_dec_1, question_dec_2, ...],
                    [question_dec_1, question_dec_2, ...],
                    ...
                ]
                where question_dec_i is a list (usually with just one element).
                The ordering of decoders is s.t. local decoders come first.

    answers:    [
                    [answer_dec_1, answer_dec_2, ...],
                    [answer_dec_1, answer_dec_2, ...],
                    ...
                ]
                where answer_dec_i is a list (usually with just one element).
                The ordering of decoders is s.t. local decoders come first.

    hidden_states: [
                        [m_left, m_right, q_left, q_right],
                        [m_left, m_right, q_left, q_right],
                        ...,
                    ]
    """

    global_params = {'d0': 1.,
                     'm_fix': 1.,
                     'q_fix': .5,
                     'v_fix': 1.,
                     'obs_times': np.linspace(0, 0.9, num=10)}

    if set_params is not None:
        for k, v in set_params.items():
            global_params[k] = v

    if correlated:
        m_samples, q_samples = corr_set(sample_num, mass_range, charge_range)
    else:
        m_samples = sample_in_range(mass_range, (sample_num, 2))
        q_samples = sample_in_range(charge_range, (sample_num, 2))
    hidden_states = np.hstack([m_samples, q_samples])

    # LOCAL EXPERIMENTS
    alphas = np.pi / 4. * np.random.rand(sample_num, 2)
    v_throw = np.sqrt(global_params['d0'] / np.sin(2 * alphas))
    v_bullet_throw = (m_samples + global_params['m_fix']) / (2 * global_params['m_fix']) * v_throw
    questions_loc = np.reshape(v_bullet_throw, (sample_num, 2, 1))
    answers_loc = np.reshape(alphas, (sample_num, 2, 1))

    # INTERACTION EXPERIMENTS
    # make phi point in the right direction
    phis = (q_samples[:, 0] * q_samples[:, 1] >= 0) * sample_in_range([-np.pi / 5., -0.05], (sample_num, 2)).T + \
        (q_samples[:, 0] * q_samples[:, 1] < 0) * sample_in_range([0.05, np.pi / 5.], (sample_num, 2)).T
    phis = phis.T

    # Questions and answers
    answers_int = np.reshape(phis, (sample_num, 2, 1))
    vs = np.vstack([get_v(m_samples[:, 0], global_params['m_fix'], q_samples[:, 0], q_samples[:, 1], phis[:, 0], global_params['d0']),
                    get_v(m_samples[:, 1], global_params['m_fix'], q_samples[:, 1], q_samples[:, 0], phis[:, 1], global_params['d0'])]).T
    questions_int = np.reshape(vs, (sample_num, 2, 1))

    questions = np.hstack([questions_loc, questions_int])
    answers = np.hstack([answers_loc, answers_int])

    # Input data
    input_data = []
    for mm, qq in zip(m_samples, q_samples):
        sample = operational_sample(mm, qq, **global_params)
        input_data.append(sample)

    input_data = np.array(input_data)

    for i, j in product([0, 1], repeat=2):
        data_series = input_data[:, i, j]
        mean_abs = np.mean(np.abs(data_series))
        input_data[:, i, j] = input_data[:, i, j] + mean_abs * np.random.normal(scale=noise_level,
                                                                                size=np.shape(input_data[:, i, j]))

    input_formatted = []
    for sample in input_data:
        if multi_enc:
            input_formatted.append(
                [
                    np.ravel([sample[0, 0], sample[0, 1]]),
                    np.ravel([sample[1, 0], sample[1, 1]])
                ]
            )

        else:
            input_formatted.append(
                [
                    np.ravel([sample[0, 0], sample[0, 1], sample[1, 0], sample[1, 1]])
                ]
            )

    return Dataset(input_formatted, questions, answers, hidden_states, global_params)
