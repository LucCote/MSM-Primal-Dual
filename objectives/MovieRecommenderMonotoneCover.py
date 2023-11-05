"""
@author: Adam Breuer
"""

import numpy as np
import pandas as pd



class MovieRecommenderMonotoneCover:  

    
    def __init__(self, movie_similarity_matrix, movie_titles, movie_genres,  movie_years, ratings_weight, cover_weight, genre_weight, year_weight, good_movie_rating):
        """
        class MovieRecommenderNonMonotoneEQ2:
            INPUTS and INSTANCE VARIABLES:
                movie_distance_matrix (becomes instance variable Sim for short): a 2D symmetric np.array of float64s
                movie_fnames: List of movie file names (strings) 
                movie_genres: List of ints, where each int represents the (single) genre of the movie of the corresponding index in movie_similarity_matrix
                movie_years: List of Ints, where each int is the year of movie release  of the movie of the corresponding index in movie_similarity_matrix
                ratings_weight: Float tuning parameter
                cover_weight: Float tuning parameter
                genre_weight: Float tuning parameter
                year_weight: Float tuning parameter
                good_movie_rating: Rating above which we consider a user to have given a movie a 'top rating'
        """

        self.groundset       = list( range(movie_similarity_matrix.shape[0]) )
        self.movie_titles    = movie_titles
        self.movie_genres    = movie_genres
        self.movie_years     = movie_years

        self.num_movies      = movie_similarity_matrix.shape[0]
        self.Sim             = movie_similarity_matrix
        self.Sim_cover       = movie_similarity_matrix >= good_movie_rating

        self.ratings_weight  = ratings_weight
        self.cover_weight    = cover_weight
        self.genre_weight    = genre_weight
        self.year_weight     = year_weight

       


    def value(self, S):
        ''' Coverage term + genre diversity term '''
        #return (1.0-self.alpha)*np.sum(self.Sim[list(set(S)), :]) + self.alpha*len(np.unique( self.movie_genres[list(set(S))] ))
        return self.ratings_weight * np.sum(self.Sim[list(set(S)), :]) + \
                self.cover_weight * np.sum( np.any(self.Sim_cover[S,:], 0) ) +\
                self.genre_weight*len(np.unique( self.movie_genres[list(set(S))] )) + \
                self.year_weight*len(np.unique( self.movie_years[list(set(S))] ))
    



    def marginalval(self, S, T):
        """ Speed-Optimized Marginal value of adding set S to current set T for value function above (EQ. 2) """
        # Fast approach
        # Coverage term: only change is that we add each of S's rows
        # Diversity term: only change is the if we increase the number of unique genres
        #return( self.value(S+T) - self.value(T) )
        if not len(S):
            return(0)
        if not len(T):
            return self.value(S)

        #cover   = (1.0-self.alpha) * np.sum(self.Sim[list(set(S)-set(T)),:])
        cover   = np.sum(self.Sim[list(set(S)-set(T)),:])
        gen_SuT = self.genre_weight * len(np.unique( self.movie_genres[list(set(S).union(T))]))
        gen_T   = self.genre_weight * len(np.unique( self.movie_genres[list(set(T))]))

        yea_SuT = self.year_weight * len(np.unique( self.movie_years[list(set(S).union(T))]))
        yea_T   = self.year_weight * len(np.unique( self.movie_years[list(set(T))]))

        return cover + gen_SuT - gen_T + yea_SuT - yea_T



    def printS(self, S):
        """ Print the names of the movies in the set (of indices) S """
        for idx in S:
            print(self.movie_titles[idx])




################################################################################
###   Some helper functions to load the movie ratings matrix and genre data  ###
################################################################################
# movies_dat_fname = 'data/ml-1m/movies.dat'

def load_movie_genres(movies_dat_fname):
    '''
    Load the movies metadata & extract the genres information.
    We say that an Action/Musical movie is a different genre than a Children's/Musical (so each movie has 1 genre)
    INPUTS:
    movies_dat_fname: a string filename of movies.dat metadata file from the movielens ml-1m data.
    OUTPUTS: 
    genres_idx: an np.array of ints, where int i corresponds to the genre of movie i.
    genres_dict: a dict where keys are the sorted index of genres and values are the genre strings
    genres_strings: an np.array of length corresponding to the number of unique movies, where each element is the string of the movie's genre
    genres_strings: an np.array of length corresponding to the number of unique movies, where each element is the string of the movie's title

    '''
    movies_df = pd.read_csv(movies_dat_fname, sep='::')
    movie_titles = movies_df['Title'].values
    movie_years = [ int((title.rsplit('(',1)[1]).replace(')','')) for title in movie_titles ]

    genres_strings = movies_df['Genres'].values
    genres_dict = {gi: idx for idx, gi in enumerate(set(genres_strings))}
    genres_idx = [genres_dict[gi] for gi in genres_strings]

    return np.array(genres_idx), genres_dict, np.array(genres_strings), np.array(movie_titles), np.array(movie_years)





# movie_ratings_mat_fname = "data/ml-1m/Movie_ratings_mat.csv"

def load_movie_user_matrix(movie_user_mat_fname):
    '''
    INPUTS:
    movie_user_mat_fname: a string filename of the SVD-Imputed user-movie ratings matrix where rows are users and columns are movies
    OUTPUTS: 
    Return a transposed 2d numpy array so that rows are movies
    '''
    # Load the movie ratings matrix. NOTE that USERS ARE ROWS (6040) and MOVIES ARE COLUMNS (3706) (we transpose this before returning)
    Sim = pd.read_csv(movie_user_mat_fname, header=None).values
    return Sim.T

















