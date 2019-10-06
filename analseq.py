'''
This module gives some simple tools for analyzing
simple sequences.
'''

from numpy.random import choice
from collections import Counter
from itertools import groupby
import numpy as np
from scipy import special

class Seq:
    '''
    An obect for handling simple sequences.
    '''

    def __init__(self, seq='', alphabet=None):
        '''
        DESCRIPTION
            Method for initializing object that by
            default will asumme an empty string but
            any arbitrary string can be given.

            An alphabet may be given for the string,
            otherwise an alphabet for be inferred as
            the set of symbols in the sequence.

        PARAMETERS
            seq (default: '')
            alphabet (default: None)
        '''
        self.seq = seq
        if not alphabet:
            self.alphabet = set(seq)
        else:
            self.alphabet = set(alphabet)

    def encode_num_to_str(self, x):
        '''
        DESCRIPTION
            Converts an iterable sequence of numbers
            into a binary str sequence where 'A'
            represents 'above the mean' and 'B'
            represents 'below the mean'. When a number
            is equal to the mean, it is not included
            in the resulting sequence. Rather, a count
            of the number of numbers equal to the mean
            is printed to notify that values were lost.

        PARAMETERS
            x (array-like)

        RETURNS
            None

        SOURCES
            Mathematical Statistics with Applications, 6th Ed.
                Dennis D. Wackerly, William Mendenhall III,
                and Richard L. Scheaffer. Duxbury Advanced Series.
        '''
        mean = np.mean(x)
        loss_count = 0
        self.seq = ''
        for x_i in x:
            if x_i > mean:
                self.seq += 'A'
            elif x_i < mean:
                self.seq += 'B'
            else:
                loss_count += 1
        print('NOTE: {} values were equal to the mean, and were dropped.'.format(loss_count))

    def gen_seq(self, n, alphabet=['H', 'T'], replace=True, asstr=True):
        alphabet = list(alphabet)
        if asstr:
            self.seq = ''.join(choice(alphabet, size=n, replace=replace))
        else:
            self.seq = choice(alphabet, size=n)
        self.alphabet = set(alphabet)
            
    def subseqs(self):
        '''
        DESCRIPTION
            Generator that yields all subsequences of a sequence.
            The sequences will not come in a prescribed order.

        PARAMETERS
            None

        YIELDS
            substr (str)

        SOURCES
            https://www.geeksforgeeks.org/python-get-all-substrings-of-given-string/
        '''
        substrs = set([self.seq[i: j] for i in range(len(self.seq)) 
              for j in range(i + 1, len(self.seq) + 1)])
        for substr in substrs:
            yield substr

    def count_runs(self):
        '''
        DESCRIPTION
            Calculate the number of runs associated with each
            symbol in the sequence.

        PARAMETERS
            None

        RETURNS
            None

        SOURCES
            https://stackoverflow.com/questions/18665563/counting-runs-in-a-string
        '''
        self.counted_runs = Counter(k for k, g in groupby(self.seq))
        self.n_runs = sum(self.counted_runs.values())

    def run_entropy(self):
        '''
        DESCRIPTION
            Calculates the Shannon entropy of the distribution of runs
            where the domain is the symbol set within the sequence. The
            probability of each run is estimated by relative frequency
            of runs associated with that run.

        PARAMETERS
            None

        RETURNS
            None
            
        SOURCE
            https://en.wikipedia.org/wiki/Entropy_(information_theory)
            https://en.wiktionary.org/wiki/Shannon_entropy
        '''
        self.run_dist = [self.counted_runs[i] / self.n_runs for i in self.counted_runs.keys()]
        self.entropy = -np.sum(np.log2(self.run_dist) * np.array(self.run_dist))
        return self.entropy

    def prob_of_runs(self):
        '''
        DESCRIPTION
            Calculates the probability of the number of
            runs among symbols from a given binary sequence.

        PARAMETERS
            None

        RETURNS
            p (float)
        
        SOURCES
            Mathematical Statistics with Applications, 6th Ed.
                Dennis D. Wackerly, William Mendenhall III,
                and Richard L. Scheaffer. Duxbury Advanced Series.
            https://en.wikipedia.org/wiki/Combination
        '''
        assert len(self.alphabet) == 2
        symbol_counts = Counter(self.seq)
        p = 1
        for symbol in symbol_counts.keys():
            p *= special.comb(symbol_counts[symbol]-1,
                              self.counted_runs[symbol]-1)
        counts = list(symbol_counts.values())
        p /= special.comb(sum(counts),
                          counts[0])
        return p
        
        
    def runs_test(self):
        '''
        DESCRIPTION
            A test for whether a binary sequence appears
            "random".

        PARAMETERS
            None

        RETURNS
            Z: test score
            p_value: significance score
            
        SOURCES
            https://en.wikipedia.org/wiki/Wald%E2%80%93Wolfowitz_runs_test
            https://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
        '''
        assert len(self.alphabet) == 2
        R = self.n_runs
        P = np.prod(list(self.counted_runs.values()))
        S = np.sum(list(self.counted_runs.values()))
        E_R = 2 * P / S + 1
        V_R = 2 * P * (2 * P - S) / (S**2 * (S - 1))
        Z = (R - E_R) / V_R
        return Z, 1 - special.ndtr(Z)

if __name__ == '__main__':
    
    print('EXAMPLE OF BINARY SEQUENCE')
    s = Seq()
    s.gen_seq(50)
    s.count_runs()
    s.run_entropy()
    print('Runs_Test: {}\nRuns_Entropy: {}\nRuns_Probability: {}\nSequence: {}'.format(s.runs_test(),
                                                                                    s.entropy,
                                                                                    s.prob_of_runs(),
                                                                                    s.seq))
    print('\nEXAMPLE OF A DNA (4-ARY) SEQUENCE')
    s = Seq()
    s.gen_seq(50, alphabet=['A', 'T', 'C', 'G'])
    s.count_runs()
    s.run_entropy()
    print('Runs_Entropy:{}\nSequence: {}'.format(
                                  s.entropy,
                                  s.seq))
