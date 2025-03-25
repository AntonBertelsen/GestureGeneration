from utils.bvh_processing.bvh_converter3 import OffsetBVHParser

# Step 1: Load the BVH file
input_bvh = "bvh_tests/bvh/trn_2023_v0_000_main-agent.bvh"
parser = OffsetBVHParser(input_bvh)

# Print basic information
joints = parser.get_all_joints()
print(f"Loaded BVH with {len(joints)} joints")
print(f"First few joints: {', '.join(joints[:5])}")

# Step 2: Extract features (channels)
features = parser.extract_channels()
print(f"Extracted features shape: {features.shape}")
print(f"First 5 values of first frame: {features[0, :5]}")

# Step 3: Convert features back to BVH
output_bvh = "bvh_tests/results/converted_output.bvh"
parser.update_motion_data(features)
parser.write_bvh(output_bvh)

print(f"Successfully converted BVH to features and back")
print(f"Original: {input_bvh}")
print(f"Result: {output_bvh}")

# Optional: Verify the result by loading the output BVH
result_parser = OffsetBVHParser(output_bvh)
result_features = result_parser.extract_channels()

# Compare shapes
print(f"Original features shape: {features.shape}")
print(f"Result features shape: {result_features.shape}")

# Compare a few values
import numpy as np
difference = np.abs(features - result_features).mean()
print(f"Mean absolute difference: {difference}")