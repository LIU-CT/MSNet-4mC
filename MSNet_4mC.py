"""
@author: liu chunting
Department of IST, Kyoto University
"""

import argparse
#import warnings

import prediction

if __name__ == "__main__":
    # MSNet_4mC

    #warnings.filterwarnings("ignore")
    #parser = argparse.ArgumentParser(description='generate feature')
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset", required=True, 
                        choices=["Lin_2017_Dataset", "Li_2020_Dataset", "User_Dataset"],
                        help="the dataset")   
    parser.add_argument("--Species", required=True, 
                        choices=["A.thaliana",     "C.elegans",
                                 "D.melanogaster", "E.coli",
                                 "G.pickeringii",  "G.subterraneus"],
                        help="the species type")
    parser.add_argument('--Fasta_file', type=str, help='input fasta file')
    args = parser.parse_args()
    
    prediction.prediction(dataset = args.Dataset, species = args.Species, fasta_file = args.Fasta_file)

