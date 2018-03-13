# Phylo-HMGP
Continuous-trait Phylogenetic Hidden Markov Gaussian Process Model

Part of the code is developed based on modification or reusing the source code of the hmmlearn package https://github.com/hmmlearn/hmmlearn.

The source code base_variant.py is developed based on modification of the file base.py from the hmmlearn package. The source code utils.py and _hmmc.c are reused from the hmmlearn package. Some functions in phylo_hmgp.py are used with modifications from the functions in the file hmm.py from the hmmlearn package. 

A statement on reusing the source code of the hmmlearn package, by the requirement of the copyright holders of hmmlearn, is available in the file folder statement_hmmlearn. A list of the authors of the hmmlearn package, as complementary to the statement, is also included in the file folder statement_hmmlearn.

The command to use Phylp-HMGP for evolutionary state estimation is as follows. 

python phylo_hmgp.py [Options]

The options:

- -n, --num_states : the number of states to estimate for Phylo-HMGP, default = 10

- -p, --root_path : root directory of the data files

- -r, --run_id : experiment id, default = 0

- -c, --cons_param : constraint parameter, default = 1

- -t, --method_mode : method_mode: 0- Phylo-HMGP-BM; 1- Phylo-HMGP-OU, default = 1

- -i, --initial_weight : initial weight for initial parameters, default = 0.1

- -j, --initial_magnitude : initial magnitude for initial parameters, default = 2

- -s, --version : dataset version, default = 1

Example: python phylo_hmgp.py -t 0 -n 20 -r 0 (using Phylo-HMGP-OU for estimating 20 states on the provided example data)

To use Phylo-HMGP, please first complie the C source file _hmmc.c into a Python module named _hmmc. The module _hmmc is needed and imported in the file base_variant.py. A sample setup.py is provided. Please use the following commands to compile the _hmmc module:

python setup.py build

python setup.py install

For the provided example, the input includes four files: edge.1.txt, branch_length.1.txt, sig.feature.1.txt, and sig.lenVec.1.txt. Please follow the descriptions of the example input files to prepare the input files for your own study. For the current version of Phylo-HMGP, please use the same file names as used in the example.

- edge.1.txt describes the topolopy of the phylogenetic tree. Phylo-HMGP uses binary trees. Please represent the phylogenetic tree in your study as a binary tree by showing the connectivity between a node and each of its children nodes. Please also add an initial ancestor node as the remote root node of the tree, in addition to the root node that exists in the phylogenetic tree of the clade of studied species. The nodes, including the remote root node, are numbered in the order of up-to-down and left-to-right and the index starts from 0. Therefore, node 0 represents the remote root node, and node 1 represents the root node of the phylogenetic tree of the clade of studied species. Each row of the edge.1.txt shows the indices of a pair of nodes, of which the first node is the parent node and the second node is one of its children. The provide example uses a phylogenetic tree with five leaf nodes (observed species). There are 10 nodes in the example tree including the remote root node.

- branch.length.1.txt shows the length of each branch of the phylogenetic tree (including the remote root node). The branches are numbered in the order of up-to-down and left-to-right, and the index starts from 0. The branch lengths are not used in the current released functions of the Phylo-HMGP, which infer equivalent transformed branch lengths that combine the temporal evolution time with the evolutionary rate or parameters (e.g., selection strength, Brownian-motion fluctuation) along this branch. Please use any nonnegative real values for the branch lengths if they are unknown to you. The branch lengths may be used in some of the functions of further improved Phylo-HMGP. 

- sig.feature.1.txt includes the observed signals (features) of the aligned genome regions of the studied species. Each row represents a genome region. Each column represents the observed signal (feature) of one species (one leaf node of the phylo-genetic tree). Please index the genome regions by the ascending order of their coordinates in the reference genomoe. The columns corresponding to the species are reverse-ordered by their indices in the phylo-genetic tree. For example, node 2,4,6,8,9 are leaf nodes in the provided examples. The columns of the input sig.feature.1.txt correspond to node 9,8,6,4,2, respectively.

- sig.lenVec.1.txt includes the lengths of the fragments of the continuous genome regions in sig.feature.1.txt. Each row is a integer showing the fragment length. Each fragment is a segment of continous genome regions. The segments are indexed by the ascending order of their coordinates in the reference genome. The sum of the values in sig.lenVec.1.txt is equal to the number of rows in sig.feature.1.txt.


************************************************************************************
# Required pre-installed packages
Phylo-HMGP requires the following packages to be installed:
- Python (tested on version 2.7.12)
- scikit-learn (tested on version 0.18.1)
- NumPy (tested on version 1.13.1)
- SciPy (tested on version 0.19.0)

You could install the Anaconda (avilable from https://www.continuum.io/downloads) for convenience, which provides a open souce collection of widely used data science packages including Python and numpy. PEP is tested using Anaconda 4.1.1.

