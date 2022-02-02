import os
import sqlite3
import pandas as pd


class DatabaseBuild:
    def __init__(self, path):
        if os.path.isdir(path):
            self._load_from_dir(path)
        elif os.path.isfile(path):
            self._load_from_db(path)
        else:
            raise ValueError("Path was not directory or file")
            
    def _load_from_dir(self, path):
        print("Loading PSMs...")
        self.psms = pd.read_csv(os.path.join(path, "psms.csv"))
        
        print("Loading Peptides...")
        self.peptides = pd.read_csv(os.path.join(path, "peptides.csv"))
        
        print("Loading Phosphosites")
        prot_data = pd.read_csv(os.path.join(path, "proteins.csv"), index_col="id")
        self.sites = pd.read_csv(os.path.join(path, "sites.csv"))
        self.sites = self.sites.join(prot_data, on="prot_id")
        
    def _load_from_db(self, path):
        chunksize = 100000
        conn = sqlite3.connect(path)
        
        print("Loading PSMs...")
        self.psms = pd.concat(
            [chunk for chunk in pd.read_sql("SELECT * FROM psm", conn, chunksize=chunksize)]
        )
        self.psms.columns = ["id", "pep_id", "qvalue", 
                             "sample_name", "scan_num", 
                             "precursor_charge", "precusor_mz"]
        
        print("Loading Peptides...")
        self.peptides = pd.read_sql("SELECT * FROM peptide", conn)
        self.peptides.columns = ["id", "qvalue", "sequence", "rt", 
                                 "nexamples", "error_rt",
                                 "z2", "z3", "z4", "z5", "z6"]
        
        print("Loading Phosphosites")
        self.sites = pd.read_sql("SELECT idSite, position, residue, fdr, accession, reference"
                                 " FROM site, protein WHERE site.idProtein = protein.idProtein", conn)