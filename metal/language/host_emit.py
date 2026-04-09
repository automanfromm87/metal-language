"""Host Emit - Generate Objective-C host code from TileKernel

Generates an ObjC program with Metal runtime compilation, buffer management,
and kernel dispatch. Data is transferred between Python and the Metal runner
via stdin/stdout.
"""

def _escape_metal_source(metal_code):
    """Escape Metal source code for use in an ObjC NSString literal"""
    return metal_code.replace("\\", "\\\\").replace('"', '\\"').replace("\n", '\\n"\n"')


def emit_host(tile_kernel, metal_code, n_runs=1):
    """Generate ObjC host code from a TileKernel and Metal source

    Args:
        tile_kernel: TileKernel
        metal_code: str, Metal shader source
        n_runs: int, number of kernel dispatches (>1 enables GPU timing)
    Returns:
        str, complete ObjC source
    """
    tk = tile_kernel
    escaped_metal = _escape_metal_source(metal_code)

    n_bufs = len(tk.buf_params)
    n_scalars = len(tk.scalar_params)

    grid_x, grid_y, grid_z = tk.grid
    block_x, block_y, block_z = tk.block

    # Buffer creation (read from stdin)
    buffer_lines = []
    buffer_lines.append("    // Read buffer metadata")
    buffer_lines.append(f"    uint32_t n_buffers = {n_bufs};")
    buffer_lines.append("    uint32_t buffer_sizes[32];")
    buffer_lines.append("    fread(buffer_sizes, sizeof(uint32_t), n_buffers, stdin);")
    buffer_lines.append("")

    for i, (name, _is_const) in enumerate(tk.buf_params):
        buffer_lines.append(
            f"    id<MTLBuffer> buffer_{name} = "
            f"[device newBufferWithLength:buffer_sizes[{i}] "
            f"options:MTLResourceStorageModeShared];"
        )
        buffer_lines.append(
            f"    fread(buffer_{name}.contents, 1, buffer_sizes[{i}], stdin);"
        )
    buffer_code = "\n".join(buffer_lines)

    # Scalar parameter reading
    scalar_lines = []
    for name in tk.scalar_params:
        scalar_lines.append(f"    uint32_t scalar_{name};")
        scalar_lines.append(f"    fread(&scalar_{name}, sizeof(uint32_t), 1, stdin);")
    scalar_code = "\n".join(scalar_lines)

    # Output: write back all buffers
    output_lines = []
    for i, (name, _is_const) in enumerate(tk.buf_params):
        output_lines.append(
            f"    fwrite(buffer_{name}.contents, 1, buffer_sizes[{i}], stdout);"
        )
    output_code = "\n".join(output_lines)

    # Generate encoder setup lines (buffer/scalar binding)
    def _gen_encoder_setup(indent):
        p = " " * indent
        lines = []
        for i, (name, _is_const) in enumerate(tk.buf_params):
            lines.append(f"{p}[encoder setBuffer:buffer_{name} offset:0 atIndex:{i}];")
        for i, name in enumerate(tk.scalar_params):
            lines.append(f"{p}[encoder setBytes:&scalar_{name} length:sizeof(uint32_t) atIndex:{n_bufs + i}];")
        return lines

    # Dispatch code: single run or benchmark loop
    if n_runs > 1:
        p2, p3 = " " * 8, " " * 12
        dispatch_lines = [f"{p2}for (int run = 0; run < {n_runs}; run++) {{"]
        dispatch_lines.append(f"{p3}id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];")
        dispatch_lines.append(f"{p3}id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];")
        dispatch_lines.append(f"{p3}[encoder setComputePipelineState:pipeline];")
        dispatch_lines.extend(_gen_encoder_setup(12))
        dispatch_lines.append(f"{p3}[encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];")
        dispatch_lines.append(f"{p3}[encoder endEncoding];")
        dispatch_lines.append(f"{p3}[commandBuffer commit];")
        dispatch_lines.append(f"{p3}[commandBuffer waitUntilCompleted];")
        dispatch_lines.append(f'{p3}if (commandBuffer.error) {{ fprintf(stderr, "GPU error\\n"); return 1; }}')
        dispatch_lines.append(f"{p3}double gpu_time = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;")
        dispatch_lines.append(f"{p3}fwrite(&gpu_time, sizeof(double), 1, stderr);")
        dispatch_lines.append(f"{p2}}}")
        dispatch_code = "\n".join(dispatch_lines)
    else:
        p2 = " " * 8
        dispatch_lines = []
        dispatch_lines.append(f"{p2}id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];")
        dispatch_lines.append(f"{p2}id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];")
        dispatch_lines.append(f"{p2}[encoder setComputePipelineState:pipeline];")
        dispatch_lines.extend(_gen_encoder_setup(8))
        dispatch_lines.append(f"{p2}[encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];")
        dispatch_lines.append(f"{p2}[encoder endEncoding];")
        dispatch_lines.append(f"{p2}[commandBuffer commit];")
        dispatch_lines.append(f"{p2}[commandBuffer waitUntilCompleted];")
        dispatch_lines.append(f"{p2}if (commandBuffer.error) {{")
        dispatch_lines.append(f'{p2}    fprintf(stderr, "GPU error: %s\\n",')
        dispatch_lines.append(f"{p2}            [[commandBuffer.error localizedDescription] UTF8String]);")
        dispatch_lines.append(f"{p2}    return 1;")
        dispatch_lines.append(f"{p2}}}")
        dispatch_code = "\n".join(dispatch_lines)

    return f"""#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

int main(int argc, const char* argv[]) {{
    @autoreleasepool {{
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {{
            fprintf(stderr, "Error: Metal not supported\\n");
            return 1;
        }}

        NSError* error = nil;
        NSString* metalSrc = @"{escaped_metal}";
        id<MTLLibrary> library = [device newLibraryWithSource:metalSrc
                                                     options:nil
                                                       error:&error];
        if (!library) {{
            fprintf(stderr, "Shader compile error: %s\\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }}

        id<MTLFunction> function = [library newFunctionWithName:@"{tk.name}"];
        if (!function) {{
            fprintf(stderr, "Kernel '{tk.name}' not found\\n");
            return 1;
        }}

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {{
            fprintf(stderr, "Pipeline error: %s\\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }}

        // Create buffers (read data from stdin)
{buffer_code}

        // Read scalar parameters
{scalar_code}

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        MTLSize gridSize = MTLSizeMake({grid_x}, {grid_y}, {grid_z});
        MTLSize threadgroupSize = MTLSizeMake({block_x}, {block_y}, {block_z});

{dispatch_code}

        // Write back all buffers
{output_code}

        return 0;
    }}
}}
"""
