from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from torchnlp.encoders.text import SpacyEncoder
from utils.common import TimeProfile
import pandas as pd
import numpy as np
from torchnlp.utils import datasets_iterator
from datasets.reuters21578 import reuters_dataset
import torch


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

LEARNING_METHOD = 'online'
STOP_WORDS = 'english'
MIN_N_GRAMS = 1
TOKEN_PATTERN = '[a-zA-Z]{4,}'

logger = logging.getLogger(__name__)

class LDA(object):

    def __init__(self, root: str, n_topics, n_top_words, n_grams=1, clean_txt=False):
        self.root = root
        self.n_topics = n_topics
        self.n_grams = n_grams
        self.n_top_words = n_top_words
        self.clean_txt=clean_txt
        self._vectorizer = None
        self._lda_model = None

    @property
    def vectorizer(self):
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(stop_words=STOP_WORDS,
                                               ngram_range=(MIN_N_GRAMS, self.n_grams),
                                               token_pattern=TOKEN_PATTERN)
        return self._vectorizer

    @property
    def lda_model(self):
        if self._lda_model is None:
            logger.info('Create Latent Dirichlet Allocation model')
            self._lda_model = LatentDirichletAllocation(n_components=self.n_topics,
                                                        learning_method=LEARNING_METHOD,
                                                        random_state=0)
        return self._lda_model

    def get_document_topic_distribution(self, vectorized_sentences):
        """Generate the Document-Topic probability distribution"""
        topic_names = [f'Topic {i+1}' for i in range(self.n_topics)]
        document_topic_prob_matrix = self.lda_model.transform(vectorized_sentences)
        document_topic_df = pd.DataFrame(np.round(document_topic_prob_matrix, 6), columns=topic_names)

        return document_topic_df

    def get_topic_words(self):
        """
                Topic 1   Topic 2 ...
        word 0
        word 1
        ...
        [
            [topic1 word1, topic1 word2, ...]
            [topic2 word1, topic2 word2, ...]
        ]
        """
        keywords = np.array(self.vectorizer.get_feature_names())
        topic_keywords = []

        # select the n_top_words to represent one topic
        for topic_weights in self.lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:self.n_top_words]
            topic_keywords.append(keywords.take(top_keyword_locs))

        topic_keywords = np.array(topic_keywords)

        # BElOW is dataframe topic_keywords
        # topic_names = [f'Topic {i + 1}' for i in range(self.n_topics)]
        # topic_keywords_df = pd.DataFrame(topic_keywords)
        # topic_keywords_df.index = topic_names
        # topic_keywords_df.columns = ['Word ' + str(i) for i in range(self.n_top_words)]

        return topic_keywords

    def lda(self):
        self.train_set, self.test_set = reuters_dataset(directory=self.root, train=True, test=True,
                                                        clean_txt=self.clean_txt)
        vectorized_sentences = self.vectorizer.fit_transform(self.train_set['text'])
        with TimeProfile("Fit Latent Dirichlet allocation model"):
            self.lda_model.fit(vectorized_sentences)
            document_topic = self.get_document_topic_distribution(vectorized_sentences)
            topic_keywords = self.get_topic_words()

        return document_topic, topic_keywords

    def lda_initialize_context_vectors(self, pretrained_model):
        _, topic_keywords = self.lda()
        text_corpus = [row['text'] for row in datasets_iterator(self.train_set, self.test_set)]
        encoder = SpacyEncoder(text_corpus, min_occurrences=3)
        centers = []
        for each_topic_keywords in topic_keywords.tolist():
            joined_key_words = str(' '.join(word) for word in each_topic_keywords)
            encoded_words = encoder.encode(joined_key_words)
            word_embeddings = pretrained_model(encoded_words)
            centers.append(torch.mean(word_embeddings, dim=0).cpu().data.numpy())
        centers = np.array(centers)
        return centers


def main():
    LDA_model = LDA(root='../data/corpora', n_topics=3, n_top_words=20)
    document_topic, topic_keywords = LDA_model.lda()


if __name__ == '__main__':
    main()