# Julia_versus_OpenMP
-A comprehensive evaluation of Julia's parallel programming approaches in comparison to C/OpenMP, under [PolyBench/C](https://www.cs.colostate.edu/~pouchet/software/polybench/), the Polyhedral Benchmark suite;\
-Empirical data on performance characteristics across different computational patterns:\
 • Linear Algebra Kernels: 2mm, 3mm (matrix multiplications)\
 • Linear Algebra Solvers: cholesky (decomposition)\
 • Stencil Computations: jacobi-2D (stencil computation)\
 • Data Mining: correlation, covariance\
 • Dynamic Programming: dynprog (2D)\
-Guidelines for selecting optimal parallelization strategies based on problem characteristics(Polybench);\
-Insights into scaling behavior from workstation-class hardware to HPC environments;\
-Identification of potential improvements for Julia's parallel computing ecosystem;\

from workstation-class hardware:\
OsName                                                  : Microsoft Windows 10 Home.\
OsArchitecture                                          : 64-bit.\
CsNumberOfLogicalProcessors                             : 8.\
CsNumberOfProcessors                                    : 1.\
One physical CPU package on the hardware. The 8 logical processors likely come from having multiple cores and/or hyperthreading within that single physical processor.\
CsProcessors                           : {Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz}.\
CsTotalPhysicalMemory                                   : 8410091520 \
total physical RAM in bytes (approximately 7.83 GB).\
CsManufacturer                                          : HUAWEI.\
HyperVisorPresent                                       : True.\
CsSystemFamily                                          : MateBook.\
CsSystemSKUNumber                                       : C178.\
CsSystemType                                            : x64-based PC.\
BiosName                                                : 1.20.

HPC environments: [DAS-5 Clusters](https://www.cs.vu.nl/das5/).
| Cluster-Node | Type        | Speed   | Memory      | Storage    | Node HDDs | Network Accelerators                                  |
|--------------|-------------|---------|-------------|------------|-----------|-------------------------------------------------------|
| VU68         | dual 8-core | 2.4 GHz | 64 GB       | 128 TB     | 2*4TB     | IB and GbE, 16*TitanX+Titan/GTX980/K20/K40             |
| LU24         | dual 8-core | 2.4 GHz | 64 GB       | 128 TB     | 2*4TB     | IB and GbE                                            |
| UVA18        | dual 8-core | 2.4 GHz | 64 GB       | 128 TB     | 2*4TB     | IB and GbE, 4*TitanX                                    |
| TUDA8        | dual 8-core | 2.4 GHz | 64 GB       | 192 TB     | 2*4TB     | IB and GbE                                            |
| UVA-MN31     | dual 8-core | 2.4 GHz | 64 GB       | 128 TB     | 2*4TB     | IB and GbE                                            |
| ASTRON9      | dual 8/10/14-core | 2.6 GHz | 128/512 GB  | 96 TB      | 2*4TB     | IB, 40 GbE, and GbE, NVIDIA/AMD GPUs; Xeon Phi        |

"DAS-5 includes roughly 200 dual-eight-core compute nodes (primarily ASUS nodes with Intel E5-2630v3 CPUs), spread out over six clusters, located at five sites. The system has been built by ClusterVision. Like its predecessor, DAS-4, DAS-5 has a mostly homogeneous core.
However, every DAS-5 site also some special hardware extensions that focus on local research priorities.

Besides using the ubiquitous Ethernet (1 Gbit/s at the compute nodes, 10 Gbit/s on the head nodes), DAS-5 also employs the high speed InfiniBand interconnect technology. FDR InfiniBand (IB) is used as an internal high-speed interconnect. To connect with the other DAS-5 clusters, a second ethernet adapter of every compute node is used that can communicate over dedicated Ethernet lightpaths by means of an OpenFlow switch at every site. These lightpaths are implemented via a fully optical (DWDM) backbone in the Netherlands, SURFnet7. See also: DAS-5 Connectivity.

The operating system the DAS-5 runs is Rocky Linux. Cluster management is done using the OpenHPC cluster management suite. In addition, software from many sources is available to support research on DAS-5: the SLURM resource management system, various MPI implementations (e.g., OpenMPI and MPICH), optimizing compilers, visualization packages, performance analysis tools, etc."


