import inspect, sys, hashlib
from math import log
# Hack around a warning message deep inside scikit learn, loaded by nltk :-(
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist, LidstoneProbDist, FreqDist

assert map_tag('brown', 'universal', 'NR-TL') == 'NOUN', '''
Brown-to-Universal POS tag map is out of date.'''


class MyProbDist(LidstoneProbDist):
    def __init__(self, freqdist, bins=None):
        # print('N:', freqdist.N())
        # print('bins:',bins)
        # print('B:',freqdist.B())
        LidstoneProbDist.__init__(self, freqdist, 0.01, bins)

    def __repr__(self):
        return '<MyProbDist based on %d samples>' % self._freqdist.N()

def myProbDist1(freqdist, gamma, bins=None):
    #print('gamma:',gamma)
    #print('bins:',bins)
    return LidstoneProbDist(freqdist, gamma, bins)

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        # raise NotImplementedError('HMM.emission_model')
        # TODO prepare data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences
        data = []
        for sent in train_data:
            sent_parsed = list(map(lambda x: (x[1], x[0].lower()),sent))
            data.extend(sent_parsed)



        # TODO compute the emission model

        #print('pair num:', len(data))
        cfdist = ConditionalFreqDist(data)
        #print(cfdist.conditions())
        #print(len(dict(cfdist['ADP'])))
        cpdist = ConditionalProbDist(cfdist, myProbDist1, 0.01)
        emission_FD = cpdist
        self.emission_PD = emission_FD
        self.states = list(cfdist.conditions())
        #print(self.elprob('VERB','is'))
        #exit()
        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4

    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state
        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        #raise NotImplementedError(('HMM.elprob',"'...'"))
        return log(self.emission_PD[state].prob(word), 2)

    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """
        #raise NotImplementedError('HMM.transition_model')
        # TODO: prepare the data
        data = []

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL <s> and the END SYMBOL </s>

        for s in train_data:
            assert(len(s)>0)
            data.append(('s',s[0][1]))
            for i in range(len(s)-1):
                data.append((s[i][1],s[i+1][1]))
            data.append((s[len(s)-1][1],'/s'))


        # TODO compute the transition model
        cfdist = ConditionalFreqDist(data)
        cpdist = ConditionalProbDist(cfdist, MyProbDist, 13)
        transition_FD = cpdist
        self.transition_PD = transition_FD
        #print(self.tlprob('VERB','VERB'))
        #exit()
        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        #raise NotImplementedError(('HMM.tlprob',"'...'"))
        return log(self.transition_PD[state1].prob(state2), 2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag

    def get_cost(self, state1, state2, observation):
        return -(self.tlprob(state1, state2) + self.elprob(state2, observation))

    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        :return: the initialized viterbi matrix and backpointer matrix of step 0
        :rtype: list(list(float)),list(list(int))
        """
        #raise NotImplementedError('HMM.initialise')
        # Initialise step 0 of viterbi, including
        #  transition from <s> to observation
        # use costs (-log-base-2 probabilities)
        # TODO
        self.viterbi = [[self.get_cost('s',state, observation) for state in self.states]]

        # Initialise step 0 of backpointer
        # TODO
        self.backpointer = [[-1 for i in range(len(self.states))]]

        return self.viterbi, self.backpointer

    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer datastructures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """
        #raise NotImplementedError(('HMM.tag',"'...'"))
        tags = []

        #assert(len(observations)>1), '''length(%d) of observations must be greater than 1.'''%(len(observations))

        # initialise
        self.initialise(observations[0])

        # viterbi decode
        for t in range(1,len(observations)): # fixme to iterate over steps
            cur_obs = observations[t]
            cur_cost = []
            cur_backpointer = []

            for s in self.states: # fixme to iterate over states
                # fixme to update the viterbi and backpointer data structures
                #  Use costs, not probabilities
                min_idx = -1
                min_cost = float("inf")

                for idx_pre_s, pre_s in enumerate(self.states):
                    c = self.get_cost(pre_s, s, cur_obs)+self.viterbi[-1][idx_pre_s]
                    assert(c>0),'''cost must be greater than 0'''
                    if  min_cost > c:
                        min_cost = c
                        min_idx = idx_pre_s
                cur_cost.append(min_cost)
                cur_backpointer.append(min_idx)

            self.viterbi.append(cur_cost)
            self.backpointer.append(cur_backpointer)

        # TODO
        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.

        termination_cost = []
        for idx, s in enumerate(self.states):
            #self.viterbi[-1][idx] -= self.tlprob(s,'/s')

            cur_cost = self.viterbi[-1][idx] - self.tlprob(s,'/s')
            termination_cost.append(cur_cost)

        self.viterbi.append(termination_cost)
        
        # TODO
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        tag_idx = [i for i, val in enumerate(self.viterbi[-1]) if (isclose(val, min(self.viterbi[-1])))][0]
        print(tag_idx)
#        print(i)
        print(self.backpointer[7])
        for i in range(len(self.viterbi)-1,0,-1):
            tags.append(self.states[tag_idx])
            print(tag_idx)
            print(i)
            tag_idx = self.backpointer[i][tag_idx]

        tags.reverse()
        return tags

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42 
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        #raise NotImplementedError(('HMM.get_viterbi_value',"'...'"))

        idx = [i for i, val in enumerate(self.states) if val==state][0]

        return self.viterbi[step][idx]

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: str
        :return: The state name to go back to at step-1
        :rtype: str
        """
        if step == 0:
            return 's'

        #raise NotImplementedError(('HMM.get_backpointer_value',"'...'"))
        idx = [i for i, val in enumerate(self.states) if val == state][0]

        return self.states[self.backpointer[step][idx]]

def answer_question4b():
    """
    Report a tagged sequence that is incorrect
    :rtype: str
    :return: your answer [max 280 chars]
    """
    '''
      [(("I'm", 'PRT'), 'PRT'), (('useless', 'ADJ'), 'ADJ'), (('for', 'ADP'), 'ADP'),
   (('anything', 'NOUN'), 'NOUN'), (('but', 'ADP'), 'CONJ'), (('racing', 'VERB'), 'ADJ'),
    (('cars', 'NOUN'), 'NOUN'), (('.', '.'), '.')]
    '''
       # raise NotImplementedError(('answer_question4b','([],[],"")'))

    tagged_sequence =  [("I'm", 'PRT'),('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'ADP'), ('racing', 'VERB'), ('cars', 'NOUN'), ('.', '.')]
    correct_sequence =  [("I'm", 'PRT'),('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'CONJ'), ('racing', 'ADJ'), ('cars', 'NOUN'), ('.', '.')]

    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
   Because some words are ambiguous, like 'but' and 'racing' in this case, and the joint probability, P(ADP|NOUN)P(but|ADP)P(VERB|ADP)P(racing|VERB)P(NOUN|VERB), is the maximum during Viterbi decoding.
""")[0:280]

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
        constructions but less than 100% lexical coverage.
        How could you use a POS tagger to ensure that the grammar
        produces a parse for any well-formed sentence,
        even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    #raise NotImplementedError(('answer_question5',"''"))

    return inspect.cleandoc("""\
If word x_i(the i-th word) is not in the lexicon, we need to estimate its emission probability P(x_i|c_i)(c_i denotes the i-th word tag).
let P(c_i|x_i) = sum(P(c_i|cc)P(cc|w_i-1) for cc in the tag set T), w_i-1 denotes the (i-1)-th word which is in the lexicon.thus
P(x_i|c_i) = P(x_i)P(c_i|x_i)/P(c_i)
           ≈ (1/P(c_i)) * sum(P(c_i|cc)P(cc|w_i-1) for cc in the tag set T )
This approach takes advantage of the existing lexicon knowledge, it will do better than the original parser.""")[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    #raise NotImplementedError(('answer_question6',"''"))

    return inspect.cleandoc("""\
Because the 'news' part of the Brown Corpus just has 4623 sentences, the original Brown Corpus tagset is too large(218 vs 12) to train a good model. The emission and transition matrix may be sparse which will lead to poor decoding results.
""")[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # prepare dataset
    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')




    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal)-test_size # fixme

    test_data_universal = tagged_sentences_universal[-test_size:]
    train_data_universal = tagged_sentences_universal[:-test_size]

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    #  until you've filled in the tag method
    ######
    s='the cat in the hat came back'.split()

    model.initialise(s[0])

    ttags =  model.tag(s) # fixme
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)

    if not (type(b_sample)== str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)


    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0
    #count = 0
    for sentence in test_data_universal:
        #print(sentence)
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)
        #print("Tagged a trial sentence:\n  %s"%list(zip(sentence,tags)))
        #exit()
        #flag = 0
        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct+=1
            else:
                incorrect+=1

                '''
                if count<10 :
                    if flag==0:
                        print("%d:\nTagged a trial sentence:\n  %s"%(count,list(zip(sentence,tags))))
                    flag = 1
                else:
                    exit()
                '''
        '''        
        if flag:
            count+=1
        '''

    #exit()
    accuracy = correct/(correct+incorrect) # fix me
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b and 5
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nAn incorrect tagged sequence is:')
    print(bad_tags)
    print('The correct tagging of this sentence would be:')
    print(good_tags)
    print('\nA possible reason why this error may have occurred is:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
