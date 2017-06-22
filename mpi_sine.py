#!/usr/bin/env python

import numpy as np 
from mpi4py import MPI
import math

def main (N):
    #Set up the communicator on each noce
    comm=MPI.COMM_WORLD
    #Size of the communicator
    size=comm.size
    rank = comm.Get_rank()

    if rank == 0:
        print '##################################################################'
        print 'An MPI program to act an array by splitting it into chunks.\n We are using {} processes'.format(size)
        print '##################################################################'

        #An array we're going to act on
        array = np.random.rand(N)
        #Where the output data will be stored
        outputData = np.zeros_like(array)

        #Split the array by the number of processes
        split = np.array_split(array,size,axis = 0) #Split input array by the number of available cores

        #Get the size of each split
        split_sizes = []

        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))

        #Each process needs to know the size of each chunk, as well as the diplacement
        #from the start of the array that the split happens.
        #Each chunk should be contiguous in memory, but need not be the same size
        split_sizes_input = split_sizes.copy()
        displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

        split_sizes_output = split_sizes.copy()
        displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]

    else:            
        #Create variables on other cores
        split_sizes_input = None
        displacements_input = None
        split_sizes_output = None
        displacements_output = None
        split = None
        array = None
        outputData = None


    #Broadcast the variables to each process
    split = comm.bcast(split, root=0) 
    split_sizes = comm.bcast(split_sizes_input, root = 0)
    displacements = comm.bcast(displacements_input, root = 0)
    split_sizes_output = comm.bcast(split_sizes_output, root = 0)
    displacements_output = comm.bcast(displacements_output, root = 0)

    #Create array to receive subset of data on each core, where rank specifies the core
    output_chunk = np.zeros(np.shape(split[rank])) 
    print("Rank %d has an empty array of shape %s" %(rank,output_chunk.shape))

    #Scatter the arrays to each core
    comm.Scatterv([array,split_sizes_input, displacements_input,MPI.DOUBLE],output_chunk,root=0)

    #perform some function on the array chunk that the process has
    sine_values = array_operation(output_chunk)

    #Make sure that all cores have finished what they're doing before we move on and gather things together again
    comm.Barrier()

    #Gather output data together
    comm.Gatherv(sine_values,[outputData,split_sizes_output,displacements_output,MPI.DOUBLE], root=0) 

    if rank == 0:    
        #If we're the master process, print the answer    
        print '{}'.format(outputData)


def array_operation(array):

    #Some operation we can do on an array  

    return np.sin(array)





if __name__ == "__main__":
    main(100000)
