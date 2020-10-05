Explanation of All Downloadable Files
Polly Fordyce
August 12, 2010

Each transcription factor studied has a folder containing the following files:

1.  TF_AllData.txt file:
This file contains all relevant raw data measured for each transcription factor.

OligoNum: oligonucleotide number (1-1457; 0 signifies an empty well).
OligoSeq: oligonucleotide sequence (empty wells are represented by empty string).
Row: row position of data point within device (1-65).
Col: column position of data point within device (1-60).
Flag: flag value for data point (a non-zero value denotes a chamber flagged for low quality during initial data analysis).
Protein_button: background-subtracted BODIPY intensity measured beneath the button valve (proportional to number of surface-bound protein molecules).
DNA_button: background-subtracted Cy5 intensity measured beneath the button valve (proportional to number of DNA molecules bound by protein).
DNA_chamber: background-subtracted Cy5 intensity measured within the DNA chamber (proportional to concentration of soluble DNA).
Ratio: protein_button/DNA_button (proportional to fractional occupancy).
DNA_Final: background-subtracted Cy5 intensity measured beneath the button valve normalized such that the background Gaussian distribution is centered around 0.
Ratio_Final: intensity ratio normalized such that the background Gaussian distribution is centered around 0 and the maximum measured intensity is set to 1.

2.  TF_Ratio.txt file and TF_Seq.fas file:
The _Seq.fas file is a fasta-formatted file containing the nucleotide sequences for each data point, and the _Ratio.txt file is a text file containing the sequences and measured final ratios for each data point.  These are the files that were input directly into fREDUCE for motif analysis.

Within each folder, there is another folder labeled 'MatrixREDUCE' containing all calculated information for the top 3 highest-scoring motifs for each transcription factor.  For each motif, the following files are provided:

1.  TF_Motif.psam
This file contains the position-specific affinity matrix obtained running MatrixREDUCE on the data with the "Motif" sequence used as an initial seed.

2.  TF_Motif.eps and TF_Motif.png
These files contain AffinityLogo representations of the position-specific affinity matrix.

3.  TF_Motif_RC.eps and TF_Motif_RC.png
These files contain AffinityLogo representations of the reverse-complement of the position-specific affinity matrix.

4.  TF_Motif.affinity
This file contains the predicted affinity information for each oligonucleotide given the calculated psam.

5.  TF_Motif.affinity.png
This file contains a graph showing the relationship between predicted and measured ratios.  This graph can be replicated by plotting the TF_Ratio.txt file vs the TF_Motif.affinity file (above).

Please feel free to contact me if you have any questions or require additional data (polly@derisilab.ucsf.edu or fordyce@gmail.com).  Thank you for your interest!
