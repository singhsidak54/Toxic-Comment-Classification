import numpy as np
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class MEModelNB:
  # We're gonna use Multinomial	Event	Model	Naive	Bayes for text classification.

  def __init__(self):
    self.N = 0 # Total Documents
    self.Nc = [0,1] # List of all the classes
    self.count_Nc = [0,0] # contains the count of all the documents belonging to a particular class
    self.vocab = {}  # contains the vocab 
    self.V = 0 # total vocab size
    self.count_w_c = {} # contains frequency of word in a particular class
    self.count_c = [0,0] # contains total number of words occuring in the class by index
    self.prior_prob = [] # contains the prior probability of all the classes

  def commentPreProcessing(self, comment):

    """This function performs text preprocessing i.e. it removes stopwords and does

        lemmatization and return all the useful words/tokens in the comment."""

    comment = comment.lower()

    # Performing Tokenization
    sentences = sent_tokenize(comment)

    words = []

    for sentence in sentences:
      c_words = word_tokenize(sentence)

      for w in c_words:
        if w not in words:
          words.append(w)

    # Performing Lemmatization and stopword removal.
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return words

  def train(self, comments, c_class):

    """ Accepts training data and trains the model. """

    m = len(comments)
    for i in range(m):

      comment = comments[i]
      c = c_class[i]

      # Step 1 - increment	the	count	of	total	documents	we	have	learned	from	N.
      self.N = self.N + 1

      # Step 2 - increment	the	count	of	documents	that	have	been	mapped	to	this	category	Nc.
      self.count_Nc[c] = self.count_Nc[c] + 1

      # Step 3 - Perform comment pre processing.
      words = self.commentPreProcessing(comment)

      # Step 4 -
      #          4.1 if	we	encounter	new	words	in	this	document,	add	them	to	our	vocabulary,	and	update our	vocabulary	size	|V|.
      #          4.2 update	count(	w,	c	)	=>	the	frequency	with	which	each	word	in	the	document	has	been mapped	to	this	category.
      #          4.3 update	count	(	c	)	=>	the	total	count	of	all	words	that	have	been	mapped	to	this	class.
      for w in words:
        if w in self.vocab:
          self.vocab[w] = self.vocab[w] + 1
          if self.count_w_c[w][c] == 0:
            self.count_c[c] += 1

          self.count_w_c[w][c] += 1
        else:
          self.vocab[w] = 1
          self.count_w_c.setdefault(w, [0, 0])
          self.count_w_c[w][c] = 1
          self.count_c[c] += 1
          self.V = self.V + 1

    # Step 5 - Calculate prior probability of both the classes.
    for i in range(2):
      prob = self.count_Nc[i] / float(self.N)
      self.prior_prob.append(prob)

  def predict(self, comment):

    """ This function accepts a comment and predicts whether it belongs to Toxic class or Non Toxic class."""

    # Step 1 - Peform comment pre processing.
    words = self.commentPreProcessing(comment)

    likelihood = [1, 1]

    # We	need	to	iterate	through	each	word	in	the	document
    # and	calculate: P(	w	|	c	)	=	[	count(	w,	c	)	+	1	]	/	[	count(	c	)	+	|V|	]

    # x	=	<w1,w2,w3....wn>  , where x = comment whose class is to predicted
    # P(x|c)	=	Product(P(wi|c)) , c = class which is either toxic or non toxic.

    # We	multiply	each	P(w|c) for	each	word	w	in	the	new	document,	then	multiply	by P(c) (Prior probability of class c)
    # and	the	result	is	the	probability	that	this	document	belongs	to	this	class.

    for c in self.Nc:

      for w in words:
        if w in self.vocab:
          cond = (self.count_w_c[w][c] + 1) / (self.count_c[c] + self.V)
        else:
          cond = 1 / (self.count_c[c] + self.V)

        likelihood[c] = likelihood[c] * cond

    for i in range(2):
      likelihood[i] = likelihood[i] * self.prior_prob[i]

    # Predict	the	class	which	has	highest	P(Y=C|x)	posterior	probabiliy.
    pred = np.argmax(likelihood)

    if pred == 1:
      return "Toxic"
    else:
      return "Non Toxic"

