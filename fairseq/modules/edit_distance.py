from fairseq.scoring.tokenizer import EvaluationTokenizer
import numpy as np
import argparse

class EditDistance(object):
    def __init__(self, args):

        self.tokenizer = EvaluationTokenizer(
        tokenizer_type=args.ed_tokenizer,
        lowercase=args.ed_lowercase,
        punctuation_removal=args.ed_remove_punct,
        character_tokenization=args.ed_char_level,
    )

    def iterative_levenshtein(self, s, t, costs=(1, 1, 2)):
        """ 
            iterative_levenshtein(s, t) -> ldist
            ldist is the Levenshtein distance between the strings 
            s and t.
            For all i and j, dist[i,j] will contain the Levenshtein 
            distance between the first i characters of s and the 
            first j characters of t
            
            costs: a tuple or a list with three integers (d, i, s)
                where d defines the costs for a deletion
                        i defines the costs for an insertion and
                        s defines the costs for a substitution
        """

        rows = len(s)+1
        cols = len(t)+1
        deletes, inserts, substitutes = costs

        operation = list()
        dist = np.zeros((rows, cols))

        # source prefixes can be transformed into empty strings 
        # by deletions:

        dist[1:, 0] = np.arange(1, rows)

        # target prefixes can be created from an empty source string
        # by inserting the characters
        dist[0, 1:] = np.arange(1, cols)

        for col in range(1, cols):
            for row in range(1, rows):
                if s[row-1] == t[col-1]:
                    cost = 0
                else:
                    cost = substitutes
                dist[row][col] = min(dist[row-1][col] + deletes,
                                    dist[row][col-1] + inserts,
                                    dist[row-1][col-1] + cost) # substitution
        return dist

    def eval(self, pred, ref, costs = (1,1,2)):
        self.ref = self.tokenizer.tokenize(ref).split()
        self.pred = self.tokenizer.tokenize(pred).split()

        self.dist = self.iterative_levenshtein(self.pred, self.ref, costs)
        return self.dist[-1,-1]

    
    # def recur_path(self, s, t, costs = (1,1,2)):
    #     rows, cols = self.dist.shape
    #     row, col = rows - 1, cols - 1

    #     deletes, inserts, substitutes = costs

    #     operation = list()
    #     while(row > 0 and col > 0):
    #         if(self.dist[row, col] == self.dist[row-1, col] + deletes):
    #             operation.insert(0, 'delete ' + s[row-1])
    #             row = row -1
    #         elif(self.dist[row, col] == self.dist[row][col-1] + inserts):
    #             operation.insert(0, 'insert ' + t[col -1])
    #             col = col -1
    #         else:
    #             if(self.dist[row, col] == self.dist[row -1][col-1] + substitutes):
    #                 operation.insert(0, 'substitute ' + s[row-1]+ ' with ' + t[col-1])      
    #             row, col = row - 1, col - 1  
                            
    #     if(not row):
    #         for i in range(col, 0, -1):
    #             operation.insert(0, 'insert ' + t[i -1])
    #     else:
    #         for i in range(row, 0, -1):
    #             operation.insert(0, 'delete ' + s[i-1])

    #     return operation
    
    def get_first_modification(self, s, t, costs = (1,1,2)):

        
        if isinstance(s, str):
            s = self.tokenizer.tokenize(s).split()
        if isinstance(t, str):
            t = self.tokenizer.tokenize(t).split()
        if s and t:
            #we must ensure s has no identical prefix to t
            assert s[0] != t[0]

        self.dist = self.iterative_levenshtein(s, t, costs)

        rows, cols = self.dist.shape
        row, col = rows - 1, cols - 1

        deletes, inserts, substitutes = costs   

        def substitute(row, col):
            return row - 1, col - 1
        def delete(row, col):
            return row - 1, col
        def insert(row, col):
            return row, col - 1 

        while(row > 0 and col > 0):   
            
            #operation_choice records all the possible operations at current point
            operation_choice = list()
            if(self.dist[row, col] == self.dist[row -1][col-1]):
                row, col = substitute(row, col)
            else:
                if(self.dist[row, col] == self.dist[row-1, col] + deletes):
                    operation_choice.append(delete)
        
                if(self.dist[row, col] == self.dist[row][col-1] + inserts):
                    operation_choice.append(insert)
                if(self.dist[row, col] == self.dist[row -1][col-1] + substitutes):
                    sub_row, sub_col = row, col     
                    operation_choice.append(substitute)

                #choice one direction randomly
                import random
                row, col = operation_choice[random.randint(0, len(operation_choice) - 1)](row, col)


        operation = None
        if(not row and col):
            s.insert(0, t[0])
            operation = 'insert'
        elif(not col and row):
            s.pop(0) 
            operation = 'delete'
        else:
            s[sub_row - 1] = t[sub_col - 1]
            operation = 'substitute'
        return s, operation

    

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('--ed_tokenizer', type=str, default="none", help='Size of each key')
    parser.add_argument('--ed_lowercase', default=True, action='store_true')
    parser.add_argument('--ed_char_level',default=False, help='number of centroids faiss should learn')
    parser.add_argument('--ed_remove_punct', default=False, action='store_true')

    args = parser.parse_args()
    edist = EditDistance(args)
    print(edist.get_first_modification("<<unk>> % s ", "<unk> % sd",costs = (1,1,2)))

    def longestCommonPrefix(strs):
        prefix=0
        for _,item in enumerate(zip(*strs)):
            if len(set(item))>1:
                return prefix
            else:
                prefix+=1
        return prefix
    print(longestCommonPrefix([['do','like'],['do','like']]))

