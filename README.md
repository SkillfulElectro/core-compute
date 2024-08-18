# core-compute
fast , simple and cross-platform parallel computing library

# Getting started
- first you will write your kernel code in shading languages which wgpu supports ( wgsl is recommended , default shader entry point is set to main so your kernel code in wgsl must contain main function )
- create variable of type compute kernel and set x , y , z of it . think about x y z here like how you would in CUDA
- create variable of type info with datas you want to send to gpu side (for now set bind and group to the same value think of it as what wgpu uses to find the right data) ( since v0.4.0 bind and group can be set to any value )
- call compute!(compute_kernel ,&mut info , ...)
- after computing , the compute macro will replace data field of infos with new data which gpu set to them
- and done !

check out : 
https://docs.rs/core-compute/latest/core_compute/
