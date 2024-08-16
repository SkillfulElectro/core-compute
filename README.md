# core-compute
fast , simple and cross-platform parallel computing library

# Getting started
- first you will write your kernel code in shading languages which wgpu supports ( wgsl is recommended )
- create variable of type compute kernel and set x , y , z of it . think about x y z here like how you would in CUDA
- create variable of type info with datas you want to send to gpu side (for now set bind and group to the same value thik of it as id which wgpu uses to find the right data)
- call compute!(compute_kernel ,&mut info , ...)
- after computing , the compute macro will replace data field of infos with new data which gpu set to them
- and done !

check out : 
https://docs.rs/core-compute/latest/core_compute/
