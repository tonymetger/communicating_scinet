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
from scipy.stats import unitary_group
from .data_handler import Dataset


def random_state(qubit_num):
    return unitary_group.rvs(2**qubit_num)[:, 0]


def random_mixed_state(qubit_num):
    pure_state = unitary_group.rvs(2 * 2**qubit_num)[:, 0]
    d = np.outer(pure_state.conj(), pure_state)
    return np.trace(d.reshape(2**qubit_num, 2, 2**qubit_num, 2), axis1=1, axis2=3)


def projection(a, b):
    return np.abs(np.dot(np.conj(a), b))**2


def local_measurement(joint_state, local_proj, site):
    if site == 0:
        meas_op = np.kron(np.outer(local_proj.conj(), local_proj), np.eye(2))
    elif site == 1:
        meas_op = np.kron(np.eye(2), np.outer(local_proj.conj(), local_proj))
    return np.real(np.trace(np.dot(meas_op, joint_state)))


def mixed_projection(pure_state, mixed_state):
    res = np.trace(np.dot(np.outer(pure_state.conj(), pure_state), mixed_state))
    assert(np.imag(res) < 1e-3)
    return np.real(res)


def create_dataset_mixed(sample_num, ref_meas_num=75, manual_ref_states=None):
    """
    Params:
    =======
    sample_num:
        number of training examples to be generated
    ref_meas_num:
        number of random measurements used to specify input and question state
    manual_ref_states: 
        used only as convenience function for plotting;
        needs to have the format [joint_states, joint_ref_states] (questions are irrelevant because we plot the latent layer)

    """
    if sample_num is None:
        sample_num = len(manual_ref_states[0])
    joint_qubit_num = 2
    local_qubit_num = 1
    dec_num = 3
    joint_ref_states = [random_state(joint_qubit_num) for _ in range(ref_meas_num)] if manual_ref_states is None else manual_ref_states[1]
    question_ref_states = [[random_state(local_qubit_num) for _ in range(ref_meas_num)],
                           [random_state(local_qubit_num) for _ in range(ref_meas_num)],
                           [random_state(joint_qubit_num) for _ in range(ref_meas_num)]]

    joint_states = np.empty([sample_num, 2**joint_qubit_num, 2**joint_qubit_num], dtype=np.complex_)
    joint_tomography = np.empty([sample_num, 1, ref_meas_num])

    # for local and global questions
    local_projection_states = np.empty([sample_num, 2, 2**local_qubit_num], dtype=np.complex_)
    global_projection_states = np.empty([sample_num, 1, 2**joint_qubit_num], dtype=np.complex_)
    question_tomography = np.empty([sample_num, dec_num, ref_meas_num])

    meas_results = np.empty([sample_num, dec_num, 1])

    for i in range(sample_num):
        joint_states[i] = random_mixed_state(joint_qubit_num) if manual_ref_states is None else manual_ref_states[0][i]
        joint_tomography[i, 0] = np.array([mixed_projection(s, joint_states[i]) for s in joint_ref_states])

        for k in range(dec_num):

            if k < 2:
                local_projection_states[i, k] = random_state(local_qubit_num)
                question_tomography[i, k] = np.array([projection(s, local_projection_states[i, k]) for s in question_ref_states[k]])
                meas_results[i, k, 0] = local_measurement(joint_state=joint_states[i],
                                                          local_proj=local_projection_states[i, k],
                                                          site=k)
            else:
                global_projection_states[i, 0] = random_state(joint_qubit_num)
                question_tomography[i, k] = np.array([projection(s, global_projection_states[i, 0]) for s in question_ref_states[k]])
                meas_results[i, k, 0] = mixed_projection(global_projection_states[i, 0], joint_states[i])

    hidden_states = [joint_states, local_projection_states, global_projection_states]

    return Dataset(joint_tomography, question_tomography, meas_results, hidden_states, [joint_ref_states, question_ref_states])
