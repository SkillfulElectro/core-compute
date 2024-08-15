

fn main() {
    let mut kernel = core_compute::compute_kernel{
        x : 4,
        y : 1,
        z : 1,
        code : r#"@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // this is used as both input and output for convenience


@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = v_indices[global_id.x] + 1;


}
"#.to_string(), 
    };


    let mut param1 = core_compute::info{
        bind : 0,
        group : 0,
        data : vec![1 ,2 ,3 , 4 ],
    };

   core_compute::compute!(kernel , &mut param1);

    println!("{:?}" , param1);
}
