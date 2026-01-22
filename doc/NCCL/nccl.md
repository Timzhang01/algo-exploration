NCCL_SOCKET_IFNAME set by environment to eth0,bond0,bond4

NCCL 通信库已通过环境变量 `NCCL_SOCKET_IFNAME`设置了用于通信的网络接口，其值为 `eth0,bond0,bond4`。该值以逗号分隔，表示 NCCL 将按顺序尝试使用这些接口进行通信。

Comm config Blocking set to 1

表示 NCCL 通信库已配置为**阻塞模式**，即 NCCL 的通信操作（如 `ncclAllReduce`）会等待操作完成才返回，而不是立即返回。

Initialized NET plugin IB

Assigned NET plugin IB to comm

DMA-BUF is available on GPU device 1

GPU 1 支持 DMA-BUF 机制，可用于实现设备间的零拷贝数据传输。

ncclCommInitRankConfig comm 0x35e69880 rank 1 nranks 16 cudaDev 1 nvmlDev 1 busId 38000 commId 0x93bd8130993d7c36 - Init START

在**初始化通信器**时打印的状态信息，表明一个包含16个进程（nranks 16）的通信组正在建立连接

NCCL INFO Bootstrap timings total 0.289828 (create 0.000023, send 0.000238, recv 0.288147, ring 0.000862, delay 0.000000)

记录了NCCL通信库在**Bootstrap网络初始化阶段**各环节所花费的时间。Bootstrap网络是NCCL在正式进行集合通信（如AllReduce）之前，为所有参与进程（Ranks）建立控制面连接的关键步骤

NCCL INFO MNNVL busId 0x59000 fabric UUID 0.0 cliqueId 0x0 state 3 healthMask 0x80

对 MNNVL（Multi-Node NVLink）通信域进行拓扑检测和状态报告。

NCCL INFO NCCL_NET_GDR_LEVEL set by environment to PXB

启用GPUDirect RDMA，允许网卡通过PCIe交换机直接访问GPU显存，绕过CPU内存。

NCCL INFO Setting affinity for GPU 6 to 64-127,192-255

将GPU的计算任务绑定到特定的CPU核心上，减少跨CPU核心调度的开销，提升通信稳定性。

NVLS multicast support is available

当前GPU支持NVLink Sharring（NVLS）技术，可用于更高效的多播通信。

NCCL INFO Trees [0] -1/-1/-1->7->6 [1] 0/-1/-1->7->6 [2] 0/-1/-1->7->6 [3] 0/-1/-1->7->6 [4] 0/-1/-1->7->6 [5] 0/-1/-1->7->6 [6] 0/-1/-1->7->6
[7] 0/15/-1->7->-1 [8] -1/-1/-1->7->6 [9] 0/-1/-1->7->6 [10] 0/-1/-1->7->6 [11] 0/-1/-1->7->6 [12] 0/-1/-1->7->6 [13] 0/-1/-1->7->6 [14] 0/-1/-1->7->6 [15] 0/15/-1->7->-1 [16] -1/-1/-1->7->6 [17] 0/-1/-1->7->6 [18] 0/-1/-1->7->6 [19] 0/-1/-1->7->6 [20] 0/-1/-1->7->6 [21] 0/-1/-1->7->6 [22] 0/-1/-1->7->6 [23] 0/-1/-1->7->15 [24] -1/-1/-1->7->6 [25] 0/-1/-1->7->6 [26] 0/-1/-1-
7->6 [27] 0/-1/-1->7->6 [28] 0/-1/-1->7->6 [29] 0/-1/-1->7->6 [30] 0/-1/-1->7->6 [31] 0/-1/-1->7->15

展示了用于集合通信（如All-Reduce）的树状拓扑结构，描述了数据在GPU间的传输路径。

P2P Chunksize set to 131072
设置点对点（Peer-to-Peer）通信中每次传输的数据块大小为128KB。
