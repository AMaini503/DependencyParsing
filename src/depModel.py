import os,sys
from decoder import *
from aux import getIndexFromFile 
from keras.models import load_model


class DepModel:
    def __init__(self, use_model = 1):
        '''
            You can add more arguments for examples actions and model paths.
            You need to load your model here.
            actions: provides indices for actions.
            it has the same order as the data/vocabs.actions file.
        '''

        if(use_model == 1):
            print("Using Model1 (200 hidden units)")
        elif(use_model == 2):
            print("Using Model2 (400 hidden units)")
    
        # Load the indices from vocab files
        self.index_of_words = getIndexFromFile(filename = 'data/vocabs.word')
        self.index_of_pos = getIndexFromFile(filename = 'data/vocabs.pos')
        self.index_of_labels = getIndexFromFile(filename = 'data/vocabs.labels')
        self.index_of_actions = getIndexFromFile(filename = 'data/vocabs.actions')
        
        # Need to sort this to keep the same mapping between integer to label
        index_of_actions_items = self.index_of_actions.items()
        sorted_index_of_actions =  sorted(index_of_actions_items, key = lambda x: x[1])
        sorted_actions = [x[0] for x in sorted_index_of_actions] 
        self.actions = sorted_actions
        
        # Load trained model here 
        model_num = str(use_model)
        self.model = load_model('saved_models/model' + str(model_num) + '.h5')

    def score(self, str_features):
        '''
        :param str_features: String features
        20 first: words, next 20: pos, next 12: dependency labels.
        DO NOT ADD ANY ARGUMENTS TO THIS FUNCTION.
        :return: list of scores
        '''
        # transform this feature vector to vector of indices
        words = str_features[0:20]
        poss = str_features[20: 20 + 20]
        labels = str_features[40: 40 + 12]
        
        new_feature_vector = []
        for word in words:
            if(word in self.index_of_words):
                new_feature_vector.append(self.index_of_words[word])
            else:
                new_feature_vector.append(self.index_of_words['<unk>'])

        for tag in poss:
            if(tag in self.index_of_pos):
                new_feature_vector.append(self.index_of_pos[tag])
            else:
                new_feature_vector.append(self.index_of_pos['<null>'])
        
        for label in labels:
            new_feature_vector.append(self.index_of_labels[label])
        
        
        return self.model.predict(
            x = np.array([new_feature_vector]),
            batch_size = 1
        )[0]

if __name__=='__main__':
    m = None
    if(len(sys.argv) == 4):
        # Use model2
        m = DepModel(use_model_2 = True)
    else:
        m = DepModel()

    input_p = os.path.abspath(sys.argv[1])
    output_p = os.path.abspath(sys.argv[2])
    Decoder(m.score, m.actions).parse(input_p, output_p)
