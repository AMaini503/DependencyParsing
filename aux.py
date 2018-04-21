import numpy as np
def getIndexFromFile(filename = None):
    """Builds token -> integer map from the given vocab file"""
    index = dict()
    with open(filename, 'r') as f:
        for line in f:
            token, ind = line.strip().split()
            index[token] = int(ind)
    
    return index

def replaceUsingIndex(oldfilename = None, newfilename = None, indices = None):
    """Creates a new data file from an old data file using indices

    Parameters
    ----------
    oldfilename - string
        Name of the old .data file. This should have string tokens
    newfilename - string
        Name of the new .data file. This will have string tokens replaced by corresponding indices
    indices - list of dictionaries. 
        Each dictionary is a map between token and corresponding integer. This is generated by replaceUsingIndex

    """
    index_of_words, index_of_pos, index_of_labels, index_of_actions = indices
    with open(oldfilename, 'r') as f_old, open(newfilename, 'w+') as f_new:
        for line in f_old:
            tokens = line.strip().split()
            
            words = tokens[0:20]
            poss = tokens[20: 20 + 20]
            labels = tokens[40: 40 + 12]
            action = tokens[52]
            newline = ''
            
            for word in words:
                if(word in index_of_words):
                    newline += str(index_of_words[word]) + ' '
                else:
                    newline += str(index_of_words['<unk>']) + ' '
                
            for pos in poss:
                if(pos in index_of_pos):
                    newline += str(index_of_pos[pos]) + ' '
                else:
                    newline += str(index_of_pos['<null>'])
                
            for label in labels:
                newline += str(index_of_labels[label]) + ' '
            
            newline += str(index_of_actions[action]) + '\n'
            f_new.write(newline)

def LoadNewTrainFile(filename = 'data/train_with_indices.data'):
    """Returns the data in the new train file and returns them as feature matrix and target label"""
    train_data = []
    train_labels = []
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split()
            tokens = [int(x) for x in tokens]
            feature_vector = tokens[0:52]
            action = tokens[52]
            
            train_data.append(feature_vector)
            train_labels.append(action)
    
    return np.asarray(train_data), np.asarray(train_labels)

def LoadTestFile(filename = 'data/dev_with_indices.data'):
    """Returns the data in the new test file and returns them as feature matrix and target label"""
    test_data = []
    test_labels = []
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split()
            tokens = [int(x) for x in tokens]
            feature_vector = tokens[0:52]
            action = tokens[52]
            
            test_data.append(feature_vector)
            test_labels.append(action)
    
    return np.asarray(test_data), np.asarray(test_labels)
