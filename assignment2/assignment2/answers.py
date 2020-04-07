a1a=['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X']
a1b=2649
a1c=12.059884527547286
a1d='function'
a2a=13
a2b=2.4630442451849275
a4a3=0.8689827219809665
a4b1=[("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'ADP'), ('racing', 'VERB'), ('cars', 'NOUN'), ('.', '.')]
a4b2=[("I'm", 'PRT'), ('useless', 'ADJ'), ('for', 'ADP'), ('anything', 'NOUN'), ('but', 'CONJ'), ('racing', 'ADJ'), ('cars', 'NOUN'), ('.', '.')]
a4b3="Because some words are ambiguous, like 'but' and 'racing' in this case, and the joint probability, P(ADP|NOUN)P(but|ADP)P(VERB|ADP)P(racing|VERB)P(NOUN|VERB), is the maximum during Viterbi decoding."
a4c=56.63055628507827
a4d=310.16310743555863
a4e=['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'ADV']
a5='If word x_i(the i-th word) is not in the lexicon, we need to estimate its emission probability P(x_i|c_i)(c_i denotes the i-th word tag).\nlet P(c_i|x_i) = sum(P(c_i|cc)P(cc|w_i-1) for cc in the tag set T), w_i-1 denotes the (i-1)-th word which is in the lexicon.thus\nP(x_i|c_i) equal P(x_i)P(c_i|x_i)/P(c_i)\n           ≈ (1/P(c_i)) * sum(P(c_i|cc)P(cc|w_i-1) for cc in the tag set T )\nThis approach takes advantage of the existing lexicon knowledge, it will do better than the original parser.'
a6="Because the 'news' part of the Brown Corpus just has 4623 sentences, the original Brown Corpus tagset is too large(218 vs 12) to train a good model. The emission and transition matrix may be sparse which will lead to poor decoding results."
a3c=16.793191282154915
a3d='s'
