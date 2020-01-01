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


import pickle as pickle
from . import io


class Dataset(object):

    """
    Data format:
        input_data:
            [
                [[1, 2], [1]],
                [[3, 2], [4]],
                [[3, 1], [5]]
            ]
            would be the correct format for 3 training examples,
            2 encoders, with the first taking input length 2 and the second taking input length 1
        questions:
            same format as input_data
        answers:
            same format as input_data
        hidden_state:
            [
                [3, 2],
                [2, 1],
                [1, 2]
            ]
            would be the correct format for three training examples of a system with 2 "hidden" paramters that are varied during training.
            Note: These hidden parameters needn't correspond to the ideal latent representation.
        global_params:
            Optional additional parameters specifying the setup (kept constant between training examples)
    """

    def __init__(self, input_data=None, questions=None, answers=None, hidden_states=None, global_params={}):
        self.input_data = input_data
        self.questions = questions
        self.answers = answers
        self.hidden_states = hidden_states
        self.global_params = global_params

    @classmethod
    def load(cls, file_name):
        with open(io.data_path + file_name + '.pkl', 'rb') as f:
            dat = pickle.load(f)
        return cls(**dat)

    def save(self, file_name):
        dat = {'input_data': self.input_data,
               'questions': self.questions,
               'answers': self.answers,
               'hidden_states': self.hidden_states,
               'global_params': self.global_params}
        for e in list(dat.values()):
            assert(e is not None)
        with open(io.data_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(dat, f, protocol=pickle.HIGHEST_PROTOCOL)

    def train_val_separation(self, p):
        """
        p: proportion of data used for validation
        """
        sep = int(len(self.input_data) * (1. - p))
        dat_train = {'global_params': self.global_params,
                     'input_data': self.input_data[:sep],
                     'questions': self.questions[:sep],
                     'answers': self.answers[:sep],
                     'hidden_states': self.hidden_states[:sep]}
        dat_val = {'global_params': self.global_params,
                   'input_data': self.input_data[sep:],
                   'questions': self.questions[sep:],
                   'answers': self.answers[sep:],
                   'hidden_states': self.hidden_states[sep:]}

        return Dataset(**dat_train), Dataset(**dat_val)
