use spirv_builder::{Capability, MetadataPrintout, SpirvBuilder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    SpirvBuilder::new("shader", "spirv-unknown-spv1.5")
        .print_metadata(MetadataPrintout::Full)
        .relax_struct_store(true)
        .relax_logical_pointer(true)
        .capability(Capability::Float16)
        .build()?;
    Ok(())
}