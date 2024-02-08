# Code structure:

## classes.py:
Contains two classes that are used in the code.  
***Individual*** for the individuals in the population; has properties **code** 
(representation of the individual) and **fitness** for part A and B as well as
**amount_of_bins** for part B.  
***Bin*** for the bins in part B; has properties **remaining_capacity** for the remaining
capacity in each bin and **items** for the items that are placed in this bin.

## strings.py:
Contains the code for the Part A (Initial search landscapes).  
Function ***calculate_fitness*** can call three different fitness 
functions (for 1.1, 1.2, 1.3).  
The best solution is displayed in the end.

## binpacking_bits.py:
Contains the code for the first solution of the Part B (Bin-packing Problem).  
Function ***read_items_from_file*** takes the information about items from the file;
the needed file can be chosen by passing the desired item from the 
***file_paths*** array.  
***bits_per_bin_assignment*** and ***start_bin_amount*** can be adjusted to 
start the algorithm with higher amount of bins.  
The best solution is displayed in the end.  

## binpacking_order.py:
Contains the code for the second solution of the Part B (Bin-packing Problem).  
Function ***read_items_from_file*** takes the information about items from the file;
the needed file can be chosen by passing the desired item from the 
***file_paths*** array.  
The best solution is displayed in the end.  

## Folder _plots_
Contains the plots of the performance of the genetic algorithms.

## Folder _problems_
Contains the .txt files of the different bin-packing problems.
