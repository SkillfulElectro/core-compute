

/// for starting your computing mission on gpu
/// first you need to write your kernel code 
/// in wgsl (recommended) or any other shader codes 
/// which wgpu supports , and then create compute_kernel variable
/// with that code .
///
/// x , y , z fields can be used to specify how many 
/// blocks of gpu is needed for your work
///
/// amounts of threads used = @workgroup_size (which you specify in your shader code) * x * y * z
#[derive(Debug , Clone)]
pub struct compute_kernel{
    /// sets number of GPU blocks in x direction
    pub x : u32 ,
    /// sets number of GPU blocks in y direction
    pub y : u32 , 
    /// sets number of GPU blocks in z direction
    pub z : u32 ,
    /// the default entry point is set to main function , so your kernel code in wgsl must 
    /// contain main function
    pub code : String,
}

impl compute_kernel{
    /// with your kernel code creates compute_kernel with x y and z set to 1
    fn new(code : String) -> Self{
        compute_kernel{
            x : 1,
            y : 1,
            z : 1,
            code : code,
        }
    }
}

/// with info struct you pass data to 
/// gpu side , for now set bind and group to the
/// same value !
/// think of it as id of your variable in wgsl side , 
/// wgpu uses it to find out where to copy data to in gpu 
/// side
///
/// in data field you should use vec! of your data 
/// the rest of variable types are not tested yet
///
/// ** since v0.4.0 bind and group can be any value **
#[derive(Debug , Clone)]
pub struct info<T>{
    /// sets binding index of variable in your kernel code (for now it must be same as your group)
    pub bind : u32,
    /// sets group index of variable in your kernel code (for now it must be same as your binding)
    pub group : u32,
    /// the data which you want grant access to GPU for rw
    pub data : T,
}

/// when you use compute! macro to start 
/// your computing , the default compute_config will 
/// be generated . but for customizing it you can 
/// create your own compute_config and pass it directly to 
/// compute_ext! macro as its first arg . _entry_point is set to main by default
/// change that to use another function for entry point of your kernel program
#[derive(Debug)]
pub struct compute_config{
    /// your custom wgpu instance
    pub _wgpu_instance : wgpu::Instance,
    /// your custom wgpu adapter
    pub _wgpu_adapter : wgpu::Adapter,
    /// your custom wgpu queue
    pub _wgpu_queue : wgpu::Queue,
    /// your custom wgpu device
    pub _wgpu_device : wgpu::Device,
    /// by default it will be set to main function in your wgsl kernel code
    pub _entry_point : String,
}

/// the default configuration tries to work on most of the devices 
impl Default for compute_config{
    /// it is used by compute macro to set defaults
    fn default() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance
                .request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("ERROR : failed to get adapter");
        let (device, queue) = pollster::block_on(adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                        memory_hints: wgpu::MemoryHints::MemoryUsage,
                    },
                    None,
                ))
                .expect("ERROR : Adapter could not find the device");

        Self {
            _wgpu_instance : instance ,
            _wgpu_adapter : adapter ,
            _wgpu_queue : queue ,
            _wgpu_device : device ,
            _entry_point : "main".to_string() ,
        }
    }
}

/// if you want to do have customized config for wgpu
/// create compute_config and pass it as first arg 
/// to this macro
#[macro_export]
macro_rules! compute_ext {
    ($config:expr , $kernel:expr, $($data:expr),*) => {
        {
            use wgpu::util::DeviceExt;
            use std::collections::HashMap;



            let instance = $config._wgpu_instance;

            let adapter = $config._wgpu_adapter;
            let device = $config._wgpu_device;
            let queue = $config._wgpu_queue;

            
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl($kernel.code.into()),
            });


            let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &shader,
                entry_point: &$config._entry_point ,
                compilation_options: Default::default(),
                cache: None,
            });



            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

            let mut staging_buffers : Vec<wgpu::Buffer> = Vec::new();
            let mut sizes : Vec<wgpu::BufferAddress> = Vec::new();
            let mut storage_buffers : Vec<wgpu::Buffer> = Vec::new();
            
            #[derive(Debug)]
            struct buf_index {
                index: usize ,
                bind : u32 ,
            }

            let mut grouponized : HashMap<u32 , Vec<buf_index>> = HashMap::new();

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });




                $(
                    if !grouponized.contains_key(&$data.group){
                        grouponized.insert($data.group , Vec::new());
                    }
                    let refr = $data.data.as_slice();
                    let size = std::mem::size_of_val(refr) as wgpu::BufferAddress;

                    sizes.push(size);

                    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size : sizes[sizes.len() - 1],
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    staging_buffers.push(staging_buffer);


                    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Storage Buffer"),
                        contents: bytemuck::cast_slice(refr),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                    });
                    storage_buffers.push(storage_buffer);

                    grouponized.get_mut(&$data.group).expect("ERROR : smth went wrong !").push(buf_index{
                        index : sizes.len() - 1,
                        bind : $data.bind
                    });

                )*


                    for group in grouponized.keys(){
                        let bind_group_layout = compute_pipeline.get_bind_group_layout(group.clone());

                        let mut entries : Vec<wgpu::BindGroupEntry> = Vec::new();
                        let data = grouponized.get(&group).expect("ERROR : smth went wrong !");
                        for GroupEntry in data {
                            entries.push(wgpu::BindGroupEntry{
                                binding : GroupEntry.bind , 
                                resource : storage_buffers[GroupEntry.index].as_entire_binding(),
                            });
                        }
                        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &bind_group_layout,
                            entries: entries.as_slice() ,
                        });

                        cpass.set_pipeline(&compute_pipeline);
                        cpass.set_bind_group(group.clone(), &bind_group, &[]);


                    }
                    

                    



                    
                    cpass.insert_debug_marker("debug_marker");
                    cpass.dispatch_workgroups($kernel.x, $kernel.y, $kernel.z); 
            }

            for (index, storage_buffer) in storage_buffers.iter().enumerate() {


                encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffers[index], 0, sizes[index]);
            }

            queue.submit(Some(encoder.finish()));




            let mut index = 0;
            $(
                let buffer_slice = staging_buffers[index].slice(..);
                let (sender, receiver) = flume::bounded(1);
                buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());


                device.poll(wgpu::Maintain::wait()).panic_on_timeout();

                if let Ok(Ok(())) = pollster::block_on(receiver.recv_async()) {
                    let data = buffer_slice.get_mapped_range();
                    let casted_data = bytemuck::cast_slice(&data).to_vec();

                    for (i, &value) in casted_data.iter().enumerate() {
                        $data.data[i] = value;
                    }


                    drop(data);
                    staging_buffers[index].unmap(); 
                } else {
                    panic!("failed to run compute on gpu!")
                }

                index += 1;
                )*



        }
    };
}

/// compute macro is used to start your computing
/// compute!(compute_kernel , &mut info , ...)
///
/// compute macro starts the computing and when it finished
/// it will change the data fields to new data which gpu did set
/// to them , this way you can get results of the computing
#[macro_export]
macro_rules! compute {
    ($kernel:expr, $($data:expr),*) => {
        {
            let config = core_compute::compute_config::default(); 
            core_compute::compute_ext!(config , $kernel, $($data),*);  
        }
    };
}
