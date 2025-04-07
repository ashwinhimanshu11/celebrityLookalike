import pickle
import os

# Load image file paths
with open('artifacts/pickle_format_data/img_pickle_file.pkl', 'rb') as f:
    filenames = pickle.load(f)

# Extract unique celebrity folder names
celebs = set()
for file in filenames:
    celeb_name = os.path.basename(os.path.dirname(file))
    celebs.add(celeb_name)

# Print all detected celebrities
print("Celebrities in dataset:")
for celeb in sorted(celebs):
    print(celeb.replace("_", " "))
