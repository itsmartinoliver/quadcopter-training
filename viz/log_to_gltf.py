import json
import struct
import base64
import math
import ast

def convert_logs_to_gltf(log_file, output_file="quadcopter_anim.gltf", fps=100.0, episodes_to_include=None):
    """
    Parses a text log of 2D quadcopter states and generates a 3D glTF animation file.
    """
    episodes = []
    current_ep = []
    
    # 1. Parse the text logs
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_ep:
                    episodes.append(current_ep)
                    current_ep = []
                continue
            
            try:
                # ast.literal_eval safely parses stringified lists like "[0.0, 1.0, ...]"
                state = ast.literal_eval(line)
                if len(state) >= 3:
                    current_ep.append(state)
            except (ValueError, SyntaxError):
                pass
                
    if current_ep:
        episodes.append(current_ep)
        
    print(f"Found {len(episodes)} total episodes.")
    
    # 2. Filter episodes if requested
    if episodes_to_include is not None:
        try:
            episodes = [episodes[i] for i in episodes_to_include]
            print(f"Filtered down to {len(episodes)} episodes based on your selection.")
        except IndexError as e:
            print(f"Warning: Check your episode indices. {e}")

    # 3. Setup glTF data structures
    binary_data = bytearray()
    buffer_views = []
    accessors = []
    nodes = []
    samplers = []
    channels = []

    # Helper function to pack binary data and create glTF accessors
    def add_buffer_data(bin_data, count, type_str, comp_type, min_val=None, max_val=None):
        offset = len(binary_data)
        binary_data.extend(bin_data)
        
        # 4-byte padding alignment (required by glTF spec)
        while len(binary_data) % 4 != 0:
            binary_data.append(0)
            
        buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(bin_data)
        })
        
        accessor = {
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": comp_type, # 5126 = FLOAT
            "count": count,
            "type": type_str
        }
        if min_val is not None: accessor["min"] = min_val
        if max_val is not None: accessor["max"] = max_val
        
        accessors.append(accessor)
        return len(accessors) - 1

    dt = 1.0 / fps

    # 4. Process each episode into geometry and animation data
    for ep_idx, ep in enumerate(episodes):
        num_frames = len(ep)
        if num_frames == 0: continue
        
        times = [i * dt for i in range(num_frames)]
        times_bin = struct.pack(f"<{num_frames}f", *times)
        
        translations = []
        rotations = []
        
        # Track min/max for translation bounding boxes
        pos_min = [float('inf')] * 3
        pos_max = [float('-inf')] * 3
        
        for state in ep:
            y, z, rot = state[0], state[1], state[2]
            tx, ty, tz = 0.0, y, z  # Mapped to 3D space
            
            translations.extend([tx, ty, tz])
            pos_min = [min(pos_min[0], tx), min(pos_min[1], ty), min(pos_min[2], tz)]
            pos_max = [max(pos_max[0], tx), max(pos_max[1], ty), max(pos_max[2], tz)]
            
            # Convert 2D rotation to 3D quaternion around the X-axis
            qx = math.sin(rot / 2.0)
            qy = 0.0
            qz = 0.0
            qw = math.cos(rot / 2.0)
            rotations.extend([qx, qy, qz, qw])
            
        # Pack everything into bytes
        trans_bin = struct.pack(f"<{num_frames * 3}f", *translations)
        rot_bin = struct.pack(f"<{num_frames * 4}f", *rotations)
        
        # Add accessors for this episode
        t_acc = add_buffer_data(times_bin, num_frames, "SCALAR", 5126, [times[0]], [times[-1]])
        trans_acc = add_buffer_data(trans_bin, num_frames, "VEC3", 5126, pos_min, pos_max)
        rot_acc = add_buffer_data(rot_bin, num_frames, "VEC4", 5126)
        
        # Create a glTF Node for this episode
        node_idx = len(nodes)
        nodes.append({
            "name": f"Episode_{ep_idx}",
            "translation": [0.0, ep[0][0], ep[0][1]],
            "rotation": [math.sin(ep[0][2]/2.0), 0.0, 0.0, math.cos(ep[0][2]/2.0)]
        })
        
        # Link samplers and channels to animate the Node
        sampler_trans = len(samplers)
        samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": trans_acc})
        
        sampler_rot = len(samplers)
        samplers.append({"input": t_acc, "interpolation": "LINEAR", "output": rot_acc})
        
        channels.append({"sampler": sampler_trans, "target": {"node": node_idx, "path": "translation"}})
        channels.append({"sampler": sampler_rot, "target": {"node": node_idx, "path": "rotation"}})

    # 5. Assemble and save the final glTF JSON payload
    b64_data = base64.b64encode(binary_data).decode('ascii')
    
    gltf = {
        "asset": {"version": "2.0", "generator": "CleanRL Log to glTF"},
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "scene": 0,
        "nodes": nodes,
        "animations": [{
            "name": "QuadCopter_Playback",
            "samplers": samplers,
            "channels": channels
        }],
        "buffers": [{
            "byteLength": len(binary_data),
            "uri": f"data:application/octet-stream;base64,{b64_data}"
        }],
        "bufferViews": buffer_views,
        "accessors": accessors
    }

    with open(output_file, 'w') as f:
        json.dump(gltf, f, indent=2)
        
    print(f"Successfully wrote {output_file}")

# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    # Example 1: Convert EVERYTHING
    # convert_logs_to_gltf("logs.txt", "all_episodes.gltf", fps=100.0)
    
    # Example 2: Convert only specific episodes (e.g., the 1st, 10th, and 50th)
    convert_logs_to_gltf(
        "../logs/replay.txt",
        "cropped_episodes.gltf",
        fps=100.0,
        episodes_to_include=range(0, 50)
        )
