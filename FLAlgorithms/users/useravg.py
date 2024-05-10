from fcntl import F_SETLEASE
import torch
from FLAlgorithms.users.userbase import User
import copy

class UserAVG(User):
    def __init__(self,  args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False):
        super().__init__(args, id, model, train_iterator, val_iterator, test_iterator, len_train, len_test, len_public, n_classes, use_adam=False)
        #def update_label_counts(self, labels, counts):
            #for label, count in zip(labels, counts):
                #self.label_counts[int(label)] += count

    #def clean_up_counts(self):
        #del self.label_counts
        #self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        #self.clean_up_counts()
        
        if self.E != 0: 
            self.fit_epochs(glob_iter, lr_decay=True)
        else: 
            self.fit_batches(glob_iter, count_labels=True, lr_decay=True)
                
