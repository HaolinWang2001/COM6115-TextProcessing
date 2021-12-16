import math
from collections import Counter  #used for counting the number of each term appears in the query

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        
        #precompute the value of idf for method tfidf:    
        if self.term_weighting == 'tfidf':
            self.idf_val = {term: math.log10(self.num_docs/len(docid)) 
                            for term, docid in self.index.items()}
        
        self.doc_vec = self.doc_vec_calculation()
        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
        
        
    def doc_vec_calculation(self):
        '''
        precompute the document vector size.
        -----------
        TF method: sum of document frequency^2 of different terms
        TFIDF method: sum of document frequency^2 of different terms
        Binary method(default): sum of weight 0 or 1
        -----------
        Returns <dict> : a dictionary of document ID and value of sqrt(df**2)
        e.g. {D01 : 1, D02 : 2,...}

        '''
        
        #create a dictionary to store the length of each document
        
        
        doc_freq_sum = dict()
        
        for term, docid_freq in self.index.items():
            #aviod duplicated calculations
            #query_term = query[term]  
            for docid, freq in docid_freq.items():
                #####################################
                if self.term_weighting in ['tf', 'tfidf']:
                    #for TF and TFIDF methods: sqrt(df^2) 
                    if docid in doc_freq_sum.keys():
                        doc_freq_sum[docid] += freq ** 2 
                    else:
                        doc_freq_sum[docid] = freq ** 2
                #####################################        
                else:
                    #for Binary method: 0 or 1
                    if docid in doc_freq_sum.keys():
                        doc_freq_sum[docid] += 1
                    else:
                        doc_freq_sum[docid] = 1
        #take square root of vector size:
        for docid, sq_sum in doc_freq_sum.items():
            doc_freq_sum[docid] = math.sqrt(sq_sum)
        return doc_freq_sum 
                        

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        '''
        ----------
        query <list> : The query to search for

        Returns <list> : 10 documents sorted by similarity with query, 
        from the most relevant to the least relevant
        -------

        '''
        #input the relavent documents only to reduce iterations
        related_docs = {query_key: self.index[query_key] 
                        for query_key in query 
                        if query_key in self.index.keys()}
        qd_vec = dict()
        sort_dict = dict()
        
        #Count number of each term appears in the query
        query_cnt = Counter()
        for term in query:
            query_cnt[term] += 1
                
        for query_i, docid_i in related_docs.items():
            for docid, freq in docid_i.items():
                if self.term_weighting in ['tf', 'tfidf']:
                    #tf method and preperation for tfidf method
                    if docid in qd_vec.keys():
                        qd_vec[docid] += query_cnt[query_i] * freq
                    else:
                        qd_vec[docid] = query_cnt[query_i] * freq
                ##############################################
                    if self.term_weighting == 'tfidf':
                        #tfidf method, which is tf * idf
                        idf = self.idf_val[query_i]
                        qd_vec[docid] *= idf
                ##############################################
                else:
                    #Binary method
                    if docid in qd_vec.keys():
                        qd_vec[docid] += 1
                    else:
                        qd_vec[docid] = 1
                        
        for docid in qd_vec.keys():
            sort_dict[docid] = qd_vec[docid]/self.doc_vec[docid]
        sort_orders = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)
        
        return [k[0] for k in sort_orders[:10]]
        
        
        
        
        
            
        


