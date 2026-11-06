#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test production buffer host barriers with Vulkan validation (lavapipe supported).

Requires glslangValidator, spirv-val, Vulkan validation layers, vulkan==1.3.275.1,
numpy and pillow. Pass --build <CMake build with compile_commands.json> --out <dir>.
"""

import argparse
import ctypes
import json
from pathlib import Path
import shlex
import subprocess

from test_fsr3_shaders import VulkanTest
from vulkan import *


def build_probe(build, out):
    commands = json.loads((build / "compile_commands.json").read_text())
    entry = next(c for c in commands if c["file"].endswith("renderer_vulkan/vk_buffer_cache.cpp"))
    source = Path(entry["file"]).read_text()
    body = source.split("void BufferCacheRuntime::FinishForHostRead() {", 1)[1].split("\n}", 1)[0]
    calls = ["scheduler.RequestOutsideRenderPassOperationContext();",
             "scheduler.Record(HostReadBarrier);", "scheduler.Finish();"]
    assert "".join(body.split()) == "".join("".join(calls).split()), body
    command = shlex.split(entry["command"])
    index = command.index("-o")
    del command[index:index + 2]
    command.remove("-c")
    command.remove(entry["file"])
    library = out.resolve() / "buffer-host-probe.so"
    command += ["-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                "-ffunction-sections", "-fdata-sections", "-shared",
                "-Wl,--gc-sections,--no-undefined",
                str(Path(__file__).with_name("buffer_host_visibility_probe.cpp").resolve()),
                "-o", str(library)]
    subprocess.run(command, cwd=entry["directory"], check=True)
    probe = ctypes.CDLL(str(library))
    probe.CheckBufferHostReadProbe.restype = ctypes.c_bool
    assert probe.CheckBufferHostReadProbe(), "production legacy/sync2 host barrier masks"
    print("PASS production runtime record-before-finish wiring and legacy/sync2 barrier masks")
    return probe


def run(build, out):
    out.mkdir(parents=True, exist_ok=True)
    probe = build_probe(build, out)
    barrier = probe.BufferHostReadProbe
    barrier.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    native = ctypes.CDLL("libvulkan.so.1")
    shader = out / "host-write.comp"
    shader.write_text("""#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) buffer A { uint a[]; };
layout(set=0,binding=1,std430) buffer B { uint b[]; };
void main() { uint i=gl_GlobalInvocationID.x; a[i]=0x12340000u+i; b[i]=0xabcd0000u+i; }
""")
    binary = out / "host-write.spv"
    subprocess.run(["glslangValidator", "-V", str(shader), "-o", str(binary)], check=True)
    subprocess.run(["spirv-val", str(binary)], check=True)
    vk = VulkanTest()
    try:
        flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT

        def buffer():
            handle = vkCreateBuffer(vk.device, VkBufferCreateInfo(
                size=1024, usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                sharingMode=VK_SHARING_MODE_EXCLUSIVE), None)
            memory = vk.allocate(vkGetBufferMemoryRequirements(vk.device, handle), flags)
            vkBindBufferMemory(vk.device, handle, memory, 0)
            vk.cleanup.append(lambda: vkDestroyBuffer(vk.device, handle, None))
            mapped = vkMapMemory(vk.device, memory, 0, 1024, 0)
            vk.cleanup.append(lambda: vkUnmapMemory(vk.device, memory))
            return handle, mapped

        a, a_map = buffer()
        b, b_map = buffer()
        staging, staging_map = buffer()
        bindings = [VkDescriptorSetLayoutBinding(binding=i, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptorCount=1, stageFlags=VK_SHADER_STAGE_COMPUTE_BIT) for i in range(2)]
        layout = vkCreateDescriptorSetLayout(vk.device, VkDescriptorSetLayoutCreateInfo(
            pBindings=bindings), None)
        vk.cleanup.append(lambda: vkDestroyDescriptorSetLayout(vk.device, layout, None))
        pool = vkCreateDescriptorPool(vk.device, VkDescriptorPoolCreateInfo(maxSets=1,
            pPoolSizes=[VkDescriptorPoolSize(type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=2)]), None)
        vk.cleanup.append(lambda: vkDestroyDescriptorPool(vk.device, pool, None))
        descriptors = vkAllocateDescriptorSets(vk.device, VkDescriptorSetAllocateInfo(
            descriptorPool=pool, pSetLayouts=[layout]))
        writes = [VkWriteDescriptorSet(dstSet=descriptors[0], dstBinding=i,
                  descriptorCount=1, descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  pBufferInfo=[VkDescriptorBufferInfo(buffer=buf, offset=0, range=1024)])
                  for i, buf in enumerate((a, b))]
        vkUpdateDescriptorSets(vk.device, len(writes), writes, 0, None)
        pipeline_layout = vkCreatePipelineLayout(vk.device, VkPipelineLayoutCreateInfo(
            pSetLayouts=[layout]), None)
        vk.cleanup.append(lambda: vkDestroyPipelineLayout(vk.device, pipeline_layout, None))
        code = binary.read_bytes()
        module = vkCreateShaderModule(vk.device, VkShaderModuleCreateInfo(codeSize=len(code), pCode=code), None)
        vk.cleanup.append(lambda: vkDestroyShaderModule(vk.device, module, None))
        stage = VkPipelineShaderStageCreateInfo(
            stage=VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main")
        pipeline = vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1,
            [VkComputePipelineCreateInfo(stage=stage, layout=pipeline_layout)], None)[0]
        vk.cleanup.append(lambda: vkDestroyPipeline(vk.device, pipeline, None))

        for compute in (False, True):
            for mixed in (False, True):
                for already_submitted in (False, True):
                    def write(cmd):
                        if compute:
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                    pipeline_layout, 0, 1, descriptors, 0, None)
                            vkCmdDispatch(cmd, 4, 1, 1)
                        else:
                            vkCmdFillBuffer(cmd, a, 0, 1024, 0x12345678)
                            vkCmdFillBuffer(cmd, b, 0, 1024, 0xabcdef01)

                    def readback(cmd):
                        if not already_submitted:
                            write(cmd)
                        if mixed:
                            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                                [VkMemoryBarrier(srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT,
                                                 dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT)],
                                0, None, 0, None)
                            vkCmdCopyBuffer(cmd, b, staging, 1, [VkBufferCopy(size=1024)])
                        barrier(int(ffi.cast("uintptr_t", cmd)),
                                ctypes.cast(native.vkCmdPipelineBarrier, ctypes.c_void_p))

                    if already_submitted:
                        vk.command(write)
                    vk.command(readback)
                    actual_a = memoryview(a_map).cast("I").tolist()
                    actual_b = memoryview(staging_map if mixed else b_map).cast("I").tolist()
                    assert actual_a == ([0x12340000 + i for i in range(256)] if compute else [0x12345678] * 256)
                    assert actual_b == ([0xabcd0000 + i for i in range(256)] if compute else [0xabcdef01] * 256)
                    print(f"PASS compute={compute} mixed={mixed} prior_submission={already_submitted}")
        assert not vk.errors, vk.errors
        print("PASS eight coherent mapped readback cases; zero Vulkan validation errors")
    finally:
        vk.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run(args.build, args.out)
