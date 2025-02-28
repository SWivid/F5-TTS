# The following code will only execute
# successfully when compression is complete

import kagglehub
kagglehub.login()
#{"username":"surajprasai","key":"b7bffbbace14f49a7de3c185a6cb8f9f"}
# Download latest version
out_path = "/home/prasais/projects/xttsv2/F5-TTS/data/indic_r_char"  # Change this to your desired directory
path = kagglehub.dataset_download("surajprasai/audio-dataset", path=out_path)

print("Path to dataset files:", path)